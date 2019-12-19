#include "mysql_manager.h"
#include "base.h"
#include "app/base_disk.h"
#include "os/qt_base.h"
#include "log_config.h"

POMutex g_sqlite_mutex;

//////////////////////////////////////////////////////////////////////////
#if (defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE))
QMutex CMySQLManager::m_s_db_mutex;
QHash<QThread*, QSqlDatabase> CMySQLManager::m_s_db_conns;

bool CMySQLManager::getDatabase(QThread* thread_ptr, QSqlDatabase& db)
{
	QMutexLocker locker(&m_s_db_mutex);

	// if we have a connection for this thread, return it
	if (m_s_db_conns.contains(thread_ptr))
	{
		db = m_s_db_conns[thread_ptr];
		return true;
	}

	// otherwise, create a new connection for this thread
	QString new_db_name  = QString("%1").arg((i64)thread_ptr);
	db = QSqlDatabase::cloneDatabase(QSqlDatabase::database(), new_db_name);

	// open the database connection
	// initialize the database connection
	if (!db.open())
	{
		printlog_lv1("Unable to open the new database connection.");
		return false;
	}
	m_s_db_conns.insert(thread_ptr, db);
	return true;
}

void CMySQLManager::removeDatabase(QThread* thread_ptr)
{
	QMutexLocker locker(&m_s_db_mutex);
	if (m_s_db_conns.contains(thread_ptr))
	{
		QSqlDatabase db = m_s_db_conns.take(thread_ptr);
		db.close();
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
CMySQLLogger::CMySQLLogger()
{
	m_is_inited = false;
	m_is_thread_cancel = false;
}

CMySQLLogger::~CMySQLLogger()
{
	exitInstance();
}

void CMySQLLogger::initInstance(DBConfig* db_param_ptr)
{
	exitInstance();
	if (!m_is_inited)
	{
		singlelog_lv0("Logger Module InitInstance");

		m_db_param_ptr = db_param_ptr;
		m_is_thread_cancel = false;

		QThreadStart();
		m_is_inited = true;
	}
}

void CMySQLLogger::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0("Logger Module ExitInstance");

		m_is_inited = false;
		m_is_thread_cancel = true;
		QThreadStop1(1000);
	}
}

void CMySQLLogger::run()
{
	singlelog_lv0("The MySQLLogger thread is");
	QElapsedTimer elasped_timer;
	elasped_timer.start();

#if defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE)
	QSqlDatabase database;
	QThread *thread_ptr = QThread::currentThread();
	if (!CMySQLManager::getDatabase(thread_ptr, database))
	{
		return;
	}

	i64 dt = 0;
	POInfoVec info_vec;
	POAlarmVec alarm_vec;
	POActionVec action_vec;
	bool is_cancel = false, is_processed = false;
	bool has_alarm, has_action, has_info;

	while (true)
	{
		{
			QMutexLocker l(&m_alarm_mutex);
			has_alarm = m_alarm_queue.getAlarmLogVec(alarm_vec);
		}
		{
			QMutexLocker l(&m_action_mutex);
			has_action = m_action_queue.getActionLogVec(action_vec);
		}
		{
			QMutexLocker l(&m_info_mutex);
			has_info = m_info_queue.getInfoLogVec(info_vec);
		}
		if (m_is_thread_cancel && !has_action && !has_alarm && !has_info)
		{
			//if app hasn't any alram or action or info log and current thread is recived quit signal,
			//it will be break...
			break;
		}

		is_processed = has_alarm | has_action | has_info;
		if (is_processed)
		{
			g_sqlite_mutex.lock();
			database.transaction();
		}
        if (has_alarm)
        {
#if defined(POR_WITH_MYSQL)
            writeAlarmToSQL(database, alarm_vec);
#elif defined(POR_WITH_SQLITE)
			writeAlarmToSQLite(database, alarm_vec);
#endif
        }
        if (has_action)
        {
#if defined(POR_WITH_MYSQL)
            writeActionToSQL(database, action_vec);
#elif defined(POR_WITH_SQLITE)
			writeActionToSQLite(database, action_vec);
#endif
        }
        if (has_info)
        {
#if defined(POR_WITH_MYSQL)
            writeInfoToSQL(database, info_vec);
#elif defined(POR_WITH_SQLITE)
			writeInfoToSQLite(database, info_vec);
#endif
        }
		if (is_processed)
		{
			database.commit();
			g_sqlite_mutex.unlock();
		}

		if (m_is_thread_cancel)
		{
			if (is_cancel)
			{
				//if writing delay time is much, break thread loop force
				i32 delay_time = elasped_timer.elapsed() - dt;
				if (delay_time > 500)
				{
					printlog_lvs2(QString("Alarm & Action writing is delayed(%1ms)").arg(delay_time), LOG_SCOPE_DB);
					alarmlog2(kPOErrDBProcessBlocked, QString("%1").arg(elasped_timer.elapsed() - dt));
					break;
				}
			}
			else
			{
				dt = elasped_timer.elapsed();
				is_cancel = true;
			}
		}
		msleep(1);
	}

	CMySQLManager::removeDatabase(thread_ptr);
#endif
}

void CMySQLLogger::addActionLog(const POAction& action)
{
	if (m_is_inited)
	{
		singlelog_lvs4("ActionLog add", LOG_SCOPE_DB);
		QMutexLocker l(&m_action_mutex);
		m_action_queue.setActionLog(action);
	}
}

void CMySQLLogger::addInfoLog(const POInfo& info)
{
	if (m_is_inited)
	{
		singlelog_lvs4("InfoLog add", LOG_SCOPE_DB);
		QMutexLocker l(&m_info_mutex);
		m_info_queue.setInfoLog(info);
	}
}
void CMySQLLogger::addAlarmLog(POECode errcode, const QString& strvalue)
{
	if (m_is_inited)
	{
		singlelog_lvs4("AlarmLog add", LOG_SCOPE_DB);
		QMutexLocker l(&m_alarm_mutex);
		m_alarm_queue.setAlarmLog(POAlarm(errcode, strvalue));
	}
}

#if defined(POR_WITH_SQLITE)
bool CMySQLLogger::writeAlarmToSQLite(QSqlDatabase& database, POAlarmVec& alarm_vec)
{
	i32 i, count = (i32)alarm_vec.size();
	if (!CPOBase::isCount(count))
	{
		return false;
	}

    QSqlQuery query(database);

	query.prepare("SELECT `alarm_index`, `alarm_limit` FROM `tbl_management`");
	if (!query.exec())
	{
		printlog_lvs2(QString("Write alarm error(step1): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
	}

	i32 alarm_index = 0;
	i32 alarm_limit = 1;
	if (!query.next())
	{
		printlog_lvs2("Write alarm error(step2): No record", LOG_SCOPE_DB);
		return false;
	}

	QString qsql;
	alarm_index = query.value(0).toInt();
	alarm_limit = query.value(1).toInt();

	for (i = 0; i < count; i++)
	{
		const POAlarm& alarm = alarm_vec[i];
		alarm_index = (alarm_index + 1) % po::_max(alarm_limit, 1);
		qsql = QString("INSERT OR REPLACE INTO `tbl_alarm`(`id`, `errcode`, `errcontent`, `created_time`) " \
						"VALUES(%1, %2, '%3', '%4')")
					.arg(alarm_index).arg(alarm.err)
					.arg(alarm.str_value).arg(QTBase::convertToString(alarm.dtm));

		query.prepare(qsql);
		if (!query.exec())
		{
			printlog_lvs2(qsql, LOG_SCOPE_DB);
			printlog_lvs2(QString("Write alarm error(step3): ") + query.lastError().text(), LOG_SCOPE_DB);
			return false;
		}
	}

	qsql = QString("UPDATE `tbl_management` SET `alarm_index` = %1").arg(alarm_index);
	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("Write alarm error(step4): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
    }
	return true;
}

bool CMySQLLogger::writeActionToSQLite(QSqlDatabase& database, POActionVec& action_vec)
{
	i32 i, count = (i32)action_vec.size();
	if (!CPOBase::isCount(count))
	{
		return false;
	}

	QSqlQuery query(database);

	QString qsql = "SELECT `action_index`, `action_limit` FROM `tbl_management`";
	query.prepare(qsql);
	if (!query.exec())
    {
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("Write action error(step1): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
	}

	i32 action_index = 0;
	i32 action_limit = 1;
	if (!query.next())
	{
		printlog_lvs2("Write action error(step2): No record", LOG_SCOPE_DB);
		return false;
	}

	action_index = query.value(0).toInt();
	action_limit = query.value(1).toInt();
	for (i = 0; i < count; i++)
	{
		const POAction& action = action_vec[i];
		action_index = (action_index + 1) % po::_max(action_limit, 1);

		qsql = QString("INSERT OR REPLACE INTO `tbl_action`(`id`, `action`, `pexpr`, `cexpr`, `content`, `created_time`) " \
						"VALUES(%1, %2, %3, %4, '%5', '%6')")
					.arg(action_index).arg(action.mode).arg(action.pexpr).arg(action.cexpr)
					.arg(action.string_value).arg(QTBase::convertToString(action.dtm));

		//write information
		query.prepare(qsql);
		if (!query.exec())
		{
			printlog_lvs2(qsql, LOG_SCOPE_DB);
			printlog_lvs2(QString("Write action error(step3): ") + query.lastError().text(), LOG_SCOPE_DB);
			return false;
		}
	}

	qsql = QString("UPDATE `tbl_management` SET `action_index` = %1").arg(action_index);
	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("Write action error(step4): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
    }
	return true;
}

bool CMySQLLogger::writeInfoToSQLite(QSqlDatabase& database, POInfoVec& info_vec)
{
	i32 i, count = (i32)info_vec.size();
	if (!CPOBase::isCount(count))
	{
		return false;
	}

	QSqlQuery query(database);

	QString qsql = "SELECT `info_index`, `info_limit` FROM `tbl_management`";
	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("Write info error(step1): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
	}

	i32 info_index = 0;
	i32 info_limit = 1;
	if (!query.next())
	{
		printlog_lvs2("Write info error(step2): No record", LOG_SCOPE_DB);
		return false;
	}

	info_index = query.value(0).toInt();
	info_limit = query.value(1).toInt();

	for (i = 0; i < count; i++)
	{
		const POInfo& info = info_vec[i];
		info_index = (info_index + 1) % po::_max(info_limit, 1);

		qsql = QString("INSERT OR REPLACE INTO `tbl_info`(`id`, `info`, `content`, `created_time`) " \
						"VALUES(%1, %2, '%3', '%4')")
					.arg(info_index).arg(info.mode).arg(info.string_value).arg(QTBase::convertToString(info.dtm));

		//write information
		query.prepare(qsql);
		if (!query.exec())
		{
			printlog_lvs2(qsql, LOG_SCOPE_DB);
			printlog_lvs2(QString("Write info error(step3): ") + query.lastError().text(), LOG_SCOPE_DB);
			return false;
		}
	}

	qsql = QString("UPDATE `tbl_management` SET `info_index` = %1").arg(info_index);
	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("Write info error(step4): ") + query.lastError().text(), LOG_SCOPE_DB);
		return false;
	}
	return true;
}
#endif

#if defined(POR_WITH_MYSQL)
bool CMySQLLogger::writeAlarmToSQL(QSqlDatabase& database, POAlarm& alarm)
{
	QSqlQuery query(database);

	QString qsql = QString("CALL add_alarm_log(%1, %2, %3)")
						.arg(alarm.err).arg(alarm.str_value).arg(alarm.dtm);
	
	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("DB Alaram Error: ") + query.lastError().text());
		return false;
	}
	return true;
}

bool CMySQLLogger::writeActionToSQL(QSqlDatabase& database, POAction& action)
{
	QSqlQuery query(database);

	QString qsql = QString("CALL add_action_log(%1, %2, %3, %4, %5)")
						.arg(action.mode).arg(action.pexpr).arg(action.cexpr).arg(action.string_value).arg(action.dtm);

	query.prepare(qsql);
	if (!query.exec())
	{
		printlog_lvs2(qsql, LOG_SCOPE_DB);
		printlog_lvs2(QString("DB Action Error: ") + query.lastError().text());
		return false;
	}
	return true;
}
#endif

//////////////////////////////////////////////////////////////////////////
CMySQLManager::CMySQLManager()
{
	m_is_inited = false;
	m_is_thread_cancel = false;

	m_db_param_ptr = NULL;
}

CMySQLManager::~CMySQLManager()
{
	exitInstance(false);
}

bool CMySQLManager::initInstance(DBConfig* db_param_ptr, CPODisk* disk_mgr_ptr, bool is_admin, bool is_loop)
{
	if (m_is_inited || !db_param_ptr)
	{
		return false;
	}

	singlelog_lv0("The MySQL Manager Module InitInstance");
	m_db_param_ptr = db_param_ptr;

#if defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE)
	if (is_admin)
	{
#if defined(POR_WITH_MYSQL)
		m_database = QSqlDatabase::addDatabase("QMYSQL3");
		m_database.setHostName(db_param_ptr->host_name.c_str());
		m_database.setDatabaseName(db_param_ptr->database_name.c_str());
		m_database.setUserName(db_param_ptr->username.c_str());
		m_database.setPassword(db_param_ptr->password.c_str());
		m_database.setPort(db_param_ptr->db_port);

		m_database.setConnectOptions("MYSQL_OPT_RECONNECT=1");

		if (!m_database.open())
		{
			m_database.setConnectOptions();
			printlog_lv1(QString("DBOpen Error: ") + m_database.lastError().text());
			return false;
		}

#elif defined(POR_WITH_SQLITE)
		potstring database_path = disk_mgr_ptr->getDatabasePath(db_param_ptr->host_name);
		m_database = QSqlDatabase::addDatabase("QSQLITE");
		m_database.setDatabaseName(QString::fromTCharArray(database_path.c_str()));

        m_database.setConnectOptions("QSQLITE_BUSY_TIMEOUT = 10000"); //10s

		if (!m_database.open())
		{
			m_database.setConnectOptions();
			printlog_lv1(QString("DBOpen Error: path[%1].").arg(QString::fromTCharArray(database_path.c_str()))
						+ m_database.lastError().text());
			return false;
		}

		QSqlQuery query(m_database);
		query.exec("PRAGMA cache_size = 4000;");
		query.exec("PRAGMA page_size = 4096;");
		query.exec("PRAGMA synchronous = OFF;");
        query.exec("PRAGMA count_changes = OFF;");
        query.exec("PRAGMA locking_mode = NORMAL;");
		query.exec("PRAGMA temp_store = MEMORY;");
		query.exec("PRAGMA journal_mode = WAL;");
		query.exec("PRAGMA auto_vacuum = NONE;");
#endif

		//insert base connection
        {
            QMutexLocker q(&m_s_db_mutex);
            m_s_db_conns.insert(QThread::currentThread(), m_database);
        }

		//update SQLManager
        updateSQLManager();
	}
	else
	{
        QMutexLocker q(&m_s_db_mutex);
		if (m_s_db_conns.size() <= 0)
		{
			printlog_lv1(QString("DBClient Connection Error: %1").arg(m_database.lastError().text()));
			return false;
		}
	}
#endif

	if (is_loop)
	{
		QThreadStart();
	}
	m_is_inited = true;
	m_is_thread_cancel = false;
	return true;
}

void CMySQLManager::exitInstance(bool is_admin)
{
	if (m_is_inited)
	{
		singlelog_lv0("The MySQL Manager Module ExitInstance");

		m_is_inited = false;
		m_is_thread_cancel = true;
		QThreadStop1(1000);
	}

#if (defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE))
	if (is_admin)
	{
		if (m_database.isOpen())
		{
			m_database.close();
		}

		{
			QMutexLocker q(&m_s_db_mutex);
			QHash<QThread*, QSqlDatabase>::Iterator iter;
			for (iter = m_s_db_conns.begin(); iter != m_s_db_conns.end(); iter++)
			{
				iter->close();
			}
			m_s_db_conns.clear();
		}
	}
#endif
}

void CMySQLManager::updateSQLManager()
{
	printlog_lv1("Invalid: Based updateSQLManager");
}

void CMySQLManager::run()
{
	singlelog_lv0("The MySQLManager thread is");

#if defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE)
	QSqlDatabase database;
	QThread* thread_ptr = QThread::currentThread();
	if (!CMySQLManager::getDatabase(thread_ptr, database))
	{
		printlog_lv1("Can't clone database connection");
		return;
	}

	while (!m_is_thread_cancel)
	{
		onCallbackDBThread((void*)&database);
		QThread::msleep(1);
	}
	removeDatabase(thread_ptr);
#endif
}

void CMySQLManager::executeQuery(DBConfig* db_param_ptr, QString& str_query)
{
	if (!db_param_ptr)
	{
		return;
	}

#if defined(POR_WITH_MYSQL)
	if (!m_is_inited)
	{
		m_database = QSqlDatabase::addDatabase("QMYSQL3");
		m_database.setHostName(db_param_ptr->host_name.c_str());
		m_database.setUserName(db_param_ptr->username.c_str());
		m_database.setPassword(db_param_ptr->password.c_str());
		m_database.setPort(db_param_ptr->db_port);

		m_database.setConnectOptions("MYSQL_OPT_RECONNECT=1");
		if (!m_database.open())
		{
			printlog_lv1(QString("DB OpenError: ") + m_database.lastError().text());
			return;
		}
	}

	//execute normal query
	if (m_database.transaction())
	{
		QSqlQuery query(m_database);
		query.exec(str_query);
		if (query.lastError().type() != QSqlError::NoError)
		{
			//rollback the transaction if there is any problem
			printlog_lv1(QString("DB SQLExcute Error: ") + query.lastError().text());
			m_database.rollback();
		}
		m_database.commit();
	}

	if (!m_is_inited && m_database.isOpen())
	{
		m_database.close();
	}
#endif
}

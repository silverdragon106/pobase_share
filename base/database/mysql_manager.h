#pragma once

#include "define.h"
#include "struct.h"
#include <QMutex>
#include <QThread>
#include <QElapsedTimer>

#if (defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE))
#include <QtSql>
#endif

const i32 kDBMaxRecordCount = 100000;

class CMySQLLogger : public QThread
{
	Q_OBJECT

public:
	CMySQLLogger();
	virtual ~CMySQLLogger();

	void						initInstance(DBConfig* db_param_ptr);
	void						exitInstance();

	void						addInfoLog(const POInfo& info);
    void						addActionLog(const POAction& action);
	void						addAlarmLog(POECode errcode, const QString& strvalue);

protected:
	virtual void				run() Q_DECL_OVERRIDE;

private:
#if defined(POR_WITH_SQLITE)
	bool						writeAlarmToSQLite(QSqlDatabase& database, POAlarmVec& alarm_vec);
	bool						writeActionToSQLite(QSqlDatabase& database, POActionVec& action_vec);
	bool						writeInfoToSQLite(QSqlDatabase& database, POInfoVec& info_vec);
#elif defined(POR_WITH_MYSQL)
	bool						writeAlarmToSQL(QSqlDatabase& database, POAlarm& alarm);
	bool						writeActionToSQL(QSqlDatabase& database, POAction& action);
#endif

private:
	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;
    DBConfig*                   m_db_param_ptr;

    AlarmQueue                  m_alarm_queue;
    ActionQueue                 m_action_queue;
	InfoQueue					m_info_queue;

	QMutex						m_alarm_mutex;
	QMutex						m_action_mutex;
	QMutex						m_info_mutex;
};

class CPODisk;
class CMySQLManager : public QThread
{
	Q_OBJECT

public:
	CMySQLManager();
	virtual ~CMySQLManager();

public:
	bool						initInstance(DBConfig* db_config_ptr, CPODisk* disk_mgr_ptr, bool is_admin, bool is_loop);
	void						exitInstance(bool is_admin);

	void						executeQuery(DBConfig* db_param_ptr, QString& str_query);

	virtual void				onCallbackDBThread(void*) = 0;

protected:
	virtual void				run() Q_DECL_OVERRIDE;
	virtual void				updateSQLManager();

protected:
	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;

    DBConfig*                   m_db_param_ptr;

#if (defined(POR_WITH_MYSQL) || defined(POR_WITH_SQLITE))
public:
	static bool					getDatabase(QThread* thread_ptr, QSqlDatabase& db);
	static void					removeDatabase(QThread* thread_ptr);

public:
	QSqlDatabase				m_database;
	static QMutex				m_s_db_mutex;
	static QHash<QThread*, QSqlDatabase> m_s_db_conns;
#endif
};

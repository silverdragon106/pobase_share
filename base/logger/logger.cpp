#include "logger.h"
#include "base.h"
#include "app/base_disk.h"
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QElapsedTimer>
#include <QDateTime>

#if defined(POR_WITH_LOG)
	#define POR_WITH_FILELOG
  #if !defined(POR_PRODUCT)
	#define POR_WITH_CONSOLELOG
  #endif
#endif

QElapsedTimer		g_elapsed_timer;
QMap<u64, u32>		g_depth_map;
POMutex				g_logger_mutex;

CLogger				g_debug_logger;
CTimeLogger			g_time_logger;
CCounterFps			g_counter_fps1;
CCounterFps			g_counter_fps2;
CCounterFps			g_counter_fps3;

inline u32 getThreadDepth(u64 thread_id)
{
	exlock_guard(g_logger_mutex);
	if (g_depth_map.contains(thread_id))
	{
		return g_depth_map[thread_id];
	}
	return 0;
}

inline void setThreadDepth(u64 thread_id, u32 depth)
{
	exlock_guard(g_logger_mutex);
    g_depth_map[thread_id] = depth;
}

//////////////////////////////////////////////////////////////////////////
CLogger::CLogger()
{
	m_is_inited = false;
	m_is_threadcancel = false;
	m_log_level = LOG_LV2;
	m_log_scope = LOG_SCOPE_ALL;
	m_mode = kLogModeQueued;

#ifdef POR_WITH_FILELOG
	m_log_stpos = 0;
	m_log_edpos = 0;

	m_filename = "";
	m_file_ptr = NULL;
	m_filesize = 0;
	m_queue_semaphore_ptr = NULL;
#endif
}

CLogger::~CLogger()
{
	exitInstance();

#ifdef POR_WITH_FILELOG
	POSAFE_DELETE(m_queue_semaphore_ptr);
#endif
}

void CLogger::initInstance(const postring& filename, LogModeType mode)
{
	g_elapsed_timer.start();
	
	if (!m_is_inited)
	{
		m_mode = mode;
		m_is_inited = true;
		m_is_threadcancel = false;
		m_log_level = LOG_LV2;
		m_log_scope = LOG_SCOPE_ALL;

#ifdef POR_WITH_FILELOG
		m_filename = filename;
		m_log_stpos = 0;
		m_log_edpos = 0;
		m_queue_semaphore_ptr = po_new QSemaphore(LOG_QUEUE_COUNT - 1);

		QThreadStart();
#endif
	}
}

void CLogger::exitInstance()
{
	if (m_is_inited)
	{
		m_is_threadcancel = true;

#ifdef POR_WITH_FILELOG
		QThreadStop();
		POSAFE_DELETE(m_queue_semaphore_ptr);
#endif

		//exitInstance
		m_is_inited = false;
	}
}

void CLogger::printLog(const QString& message, bool is_console_only)
{
#if defined(POR_WITH_CONSOLELOG) || defined(POR_WITH_FILELOG)
	if (!m_is_inited)
	{
		return;
	}

	//output message in console with color
	u64 thread_id = (u64)QThread::currentThreadId();
	QDateTime dtm = QDateTime::currentDateTime();
	QDate dm = dtm.date();
	QTime tm = dtm.time();

	char log_message[LOG_MAX_PATH];
	po_sprintf(log_message, LOG_MAX_PATH, "%02d:%02d-%02d:%02d:%02d(%03d):%05d|%*s%s",
            dm.month(), dm.day(), tm.hour(), tm.minute(), tm.second(), tm.msec(),
			(u16)(thread_id % 100000), 2 * getThreadDepth(thread_id), "", message.toUtf8().data());

#if defined(POR_WITH_CONSOLELOG)
	qDebug() << log_message;
#endif
	if (is_console_only)
	{
		return;
	}

#if defined(POR_WITH_FILELOG)
	switch (m_mode)
	{
		case kLogModeQueued:
		{
			if (m_queue_semaphore_ptr)
			{
				i32 free_queue_size = m_queue_semaphore_ptr->available();
				if (free_queue_size <= 0)
				{
					printf("LogQueue Full...");
				}
				m_queue_semaphore_ptr->tryAcquire(1, 2000);
			}

			QMutexLocker l(&m_queue_mutex);
			m_log_edpos = (m_log_edpos + 1) % LOG_QUEUE_COUNT;
			strcpy(m_log_message[m_log_edpos], log_message);
			break;
		}
		case kLogModeDirect:
		{
			logFileOpen();
			if (m_file_ptr)
			{
				//output log message to file
				fprintf(m_file_ptr, "%s\n", log_message);
				fflush(m_file_ptr);

				m_filesize += strlen(log_message);
			}
			break;
		}
        default:
        {
            break;
        }
    }
#endif
#endif
}

void CLogger::printLogWithArg(i32 log_level, const char* format, ...)
{
	char log_message[LOG_MAX_PATH];
	va_list args;
	va_start(args, format);
	vsnprintf(log_message, LOG_MAX_PATH, format, args);
	va_end(args);

	if (isAvailableLevel(log_level))
	{
		printLog(log_message);
	}
}

void CLogger::run()
{
#ifdef POR_WITH_FILELOG
	i32 records = 0;
	char log_message[LOG_QUEUE_COUNT][LOG_MAX_PATH];
	
	logFileOpen();

	while (!m_is_threadcancel)
	{
		records = 0;
		{
			QMutexLocker l(&m_queue_mutex);
			while (m_log_stpos != m_log_edpos)
			{
				m_log_stpos = (m_log_stpos + 1) % LOG_QUEUE_COUNT;
				strcpy(log_message[records], m_log_message[m_log_stpos]);
				records++;
			}
			if (m_queue_semaphore_ptr)
			{
				m_queue_semaphore_ptr->release(records);
				
				i32 available = m_queue_semaphore_ptr->available();
				if (available > LOG_QUEUE_COUNT - 1)
				{
					printf("queue_semaphore available is %d", available);
					assert(false);
				}
			}
		}
		
		if (records)
		{
			logFileOpen();
			if (m_file_ptr)
			{
				for (i32 i = 0; i < records; i++)
				{
					//output log message to file
					fprintf(m_file_ptr, "%s\n", log_message[i]);
					m_filesize += strlen(log_message[i]);
				}
				fflush(m_file_ptr);
			}
		}
		QThread::msleep(10); //idle-level-callback
	}

	logFileClose();
#endif
}

bool CLogger::setLogLevel(i32 level)
{
	if (!m_is_inited || !CPOBase::checkRange(level, LOG_LV0, LOG_LV4))
	{
		return false;
	}

	m_log_level = level;
	return true;
}

bool CLogger::setLogScope(i32 scope)
{
	if (!m_is_inited)
	{
		return false;
	}

	m_log_scope = scope;
	return true;
}

void CLogger::addLogScope(i32 scope)
{
	if (!m_is_inited)
	{
		return;
	}

	m_log_scope |= scope;
}

void CLogger::removeLogScope(i32 scope)
{
	if (!m_is_inited)
	{
		return;
	}

	m_log_scope &= ~scope;
}


void CLogger::logFileOpen(i32 call_lv)
{
	if (call_lv > 2)
	{
		return;
	}

#ifdef POR_WITH_FILELOG
	if (m_file_ptr)
	{
		if (m_filesize > PO_LOG_FILESIZE)
		{
			fflush(m_file_ptr);
			fclose(m_file_ptr);
			m_file_ptr = NULL;
			m_filesize = 0;
			backupLogFile();
		}
	}

	if (!m_file_ptr && m_filename.length() > 0)
	{
		m_file_ptr = fopen(m_filename.c_str(), "a+");
		if (m_file_ptr)
		{
			fseek(m_file_ptr, 0, SEEK_END);
			m_filesize = ftell(m_file_ptr);
			logFileOpen(call_lv + 1);
		}
	}
#endif
}

void CLogger::logFileClose()
{
#ifdef POR_WITH_FILELOG
	if (m_file_ptr)
	{
		fflush(m_file_ptr);
		fclose(m_file_ptr);
		m_file_ptr = NULL;
		m_filesize = 0;
	}
#endif
}

void CLogger::backupLogFile()
{
#ifdef POR_WITH_FILELOG
	if (m_file_ptr)
	{
		fflush(m_file_ptr);
		fclose(m_file_ptr);
	}

	if (isAvailableLevel(LOG_LV3))
	{
		// LV3이상의 로그레벨에서는 로그백업파일을 생성한다.
		postring backup_filename = CPODisk::getNonExtPath(m_filename) + "_" + CPOBase::getDateTimeFileName() + ".txt";
		CPODisk::rename(m_filename, backup_filename);
	}
	else
	{
		// LV3이하의 본로그레벨에서는 로그백업파일을 생성하지 않는다.
		CPODisk::deleteFile(m_filename);
	}
#endif
}

//////////////////////////////////////////////////////////////////////////
CSingleLogger::CSingleLogger(const QString& str)
{
	m_log_message = str;
	if (m_log_message.length() > 0)
	{
		printlog_lv0(m_log_message + " start.");
		u64 thread_id = (u64)QThread::currentThreadId();
		setThreadDepth(thread_id, getThreadDepth(thread_id) + 1);
	}
}

CSingleLogger::~CSingleLogger()
{
	if (m_log_message.length() > 0)
	{
		u64 thread_id = (u64)QThread::currentThreadId();
		setThreadDepth(thread_id, getThreadDepth(thread_id) - 1);
		printlog_lv0(m_log_message + " done.");
	}
}

//////////////////////////////////////////////////////////////////////////
TMTick::TMTick()
{
	reset();
}

void TMTick::reset()
{
	tick_name = "";
	clear();
}

void TMTick::clear()
{
	tick_count = 0;
	start_tick_time = -1;
	avg_tick_time = 0;
	min_tick_time = 0;
	max_tick_time = 0;
	last_tick_time = 0;
}

//////////////////////////////////////////////////////////////////////////
CTimeLogger::CTimeLogger()
{
	m_is_inited = false;
	m_time_tick_vec.resize(LOG_MAX_TIMER);
	g_elapsed_timer.start();

#ifdef POR_WITH_FILELOG
	m_filename = "";
	m_file_ptr = NULL;
#endif
}

CTimeLogger::~CTimeLogger()
{
	exitInstance();
}

void CTimeLogger::initInstance(const postring& log_filename)
{
	m_time_tick_vec.resize(LOG_MAX_TIMER);
	for (i32 i = 0; i < LOG_MAX_TIMER; i++)
	{
		m_time_tick_vec[i].reset();
	}

#ifdef POR_WITH_FILELOG
	m_filename = log_filename;
	if (m_filename.length())
	{
		m_file_ptr = fopen(m_filename.c_str(), "w");
	}
#endif
	m_is_inited = true;
}

void CTimeLogger::exitInstance()
{
#ifdef POR_WITH_FILELOG
	if (m_file_ptr)
	{
		fflush(m_file_ptr);
		fclose(m_file_ptr);
	}
#endif
	m_time_tick_vec.clear();
	m_is_inited = false;
}

TMTick* CTimeLogger::getTickData(i32 index)
{
	if (!CPOBase::checkIndex(index, LOG_MAX_TIMER))
	{
		return NULL;
	}
	return m_time_tick_vec.data() + index;
}

void CTimeLogger::timeName(i32 index, const char* timer_name)
{
	if (CPOBase::checkIndex(index, LOG_MAX_TIMER))
	{
		TMTick* tick_ptr = m_time_tick_vec.data() + index;
		tick_ptr->tick_name = timer_name;
	}
}

void CTimeLogger::timeStart(i32 index)
{
	if (CPOBase::checkIndex(index, LOG_MAX_TIMER))
	{
		TMTick* tick_ptr = m_time_tick_vec.data() + index;
		tick_ptr->start_tick_time = g_elapsed_timer.elapsed();
	}
}

void CTimeLogger::timeStop(i32 index)
{
	if (CPOBase::checkIndex(index, LOG_MAX_TIMER))
	{
		TMTick* tick_ptr = m_time_tick_vec.data() + index;
		if (tick_ptr->start_tick_time < 0)
		{
			return;
		}

		i64 dt = g_elapsed_timer.elapsed() - tick_ptr->start_tick_time;

		tick_ptr->tick_count++;
		if (tick_ptr->tick_count == 1)
		{
			tick_ptr->avg_tick_time = dt;
			tick_ptr->min_tick_time = dt;
			tick_ptr->max_tick_time = dt;
			tick_ptr->last_tick_time = dt;
		}
		else
		{
			tick_ptr->avg_tick_time = (tick_ptr->avg_tick_time * (tick_ptr->tick_count - 1) + dt) / tick_ptr->tick_count;
			tick_ptr->min_tick_time = po::_min(tick_ptr->min_tick_time, dt);
			tick_ptr->max_tick_time = po::_max(tick_ptr->max_tick_time, dt);
			tick_ptr->last_tick_time = dt;
		}
	}
}

void CTimeLogger::clear(i32 st_index)
{
	i32 i, count = (i32)m_time_tick_vec.size();
	for (i = st_index; i < count; i++)
	{
		m_time_tick_vec[i].clear();
	}
}

void CTimeLogger::debugOutput()
{
#if defined(POR_WITH_FILELOG)
	if (!m_file_ptr || !m_is_inited)
	{
		return;
	}

	fseek(m_file_ptr, 0, SEEK_SET);

	TMTick* tick_ptr;
	TMTick* tick_data_ptr = m_time_tick_vec.data();
	for (i32 i = 0; i < LOG_MAX_TIMER; i++)
	{
		tick_ptr = tick_data_ptr + i;
		fprintf(m_file_ptr, "%02d: count=%d, avg=%d, min=%d, max=%d, cur=%d \t[%s]\n",
			i, tick_ptr->tick_count,
			(i32)(tick_ptr->avg_tick_time), (i32)(tick_ptr->min_tick_time),
			(i32)(tick_ptr->max_tick_time), (i32)(tick_ptr->last_tick_time),
			tick_ptr->tick_name.c_str());
	}
	fflush(m_file_ptr);

#else
	if (!m_is_inited)
	{
		return;
	}

	TMTick* tick_ptr;
	TMTick* tick_data_ptr = m_time_tick_vec.data();
	char log_message[LOG_MAX_PATH];

	for (i32 i = 0; i < LOG_MAX_TIMER; i++)
	{
		tick_ptr = tick_data_ptr + i;
		po_sprintf(log_message, LOG_MAX_PATH, "%02d: count=%d, avg=%d, min=%d, max=%d, cur=%d \t[%s]\n",
			i, tick_ptr->tick_count,
			(i32)(tick_ptr->avg_tick_time), (i32)(tick_ptr->min_tick_time),
			(i32)(tick_ptr->max_tick_time), (i32)(tick_ptr->last_tick_time),
			tick_ptr->tick_name.c_str());

		qDebug() << log_message;
	}
#endif
}

///////////////////////////////////////////////////////////////////////////////
CCounterFps::CCounterFps()
{
	m_interval = 1000;
	m_counter = 0;
	m_time_stamp = 0;
	m_elapsed_timer.start();
}

CCounterFps::~CCounterFps()
{
}

f32 CCounterFps::fps()
{
	i32 dt = m_elapsed_timer.elapsed()-m_time_stamp;
	if (dt > m_interval)
	{
		f32 fps = (f32)(m_counter + 1) * 1000 / dt;
		m_time_stamp = m_elapsed_timer.elapsed();
		m_counter = 0;
		return fps;
	}
	m_counter++;
	return -1;
}

f32 CCounterFps::fps(i32 interval)
{
	i32 dt = m_elapsed_timer.elapsed()-m_time_stamp;
	if (dt > interval && interval > 0)
	{
		f32 fps = (f32)(m_counter + 1) * 1000 / dt;
		m_time_stamp = m_elapsed_timer.elapsed();
		m_counter = 0;
		return fps;
	}
	m_counter++;
	return -1;
}

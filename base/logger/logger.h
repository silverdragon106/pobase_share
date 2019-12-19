#pragma once

#include "types.h"
#include "define.h"
#include <QString>
#include <QElapsedTimer>
#include <QThread>
#include <QSemaphore>
#include <QMutex>

#if defined(POR_WITH_LOG)
	#define printlog(a, b)			if (g_debug_logger.isAvailableLevel(b)) { g_debug_logger.printLog(a); };
	#define printlog_lv0(a)			if (g_debug_logger.isAvailableLevel(LOG_LV0)) { g_debug_logger.printLog(a); };
	#define printlog_lv1(a)			if (g_debug_logger.isAvailableLevel(LOG_LV1)) { g_debug_logger.printLog(a); };
	#define printlog_lv2(a)			if (g_debug_logger.isAvailableLevel(LOG_LV2)) { g_debug_logger.printLog(a); };
	#define printlog_lv3(a)			if (g_debug_logger.isAvailableLevel(LOG_LV3)) { g_debug_logger.printLog(a); };
	#define printlog_lv4(a)			if (g_debug_logger.isAvailableLevel(LOG_LV4)) { g_debug_logger.printLog(a); };
	
	#define printlog_lvs(a, b)		if (g_debug_logger.isAvailableLevel(b)) { g_debug_logger.printLog(a); };
	#define printlog_lvs0(a, b)		if (g_debug_logger.isAvailable(LOG_LV0, b)) { g_debug_logger.printLog(a); };
	#define printlog_lvs1(a, b)		if (g_debug_logger.isAvailable(LOG_LV1, b)) { g_debug_logger.printLog(a); };
	#define printlog_lvs2(a, b)		if (g_debug_logger.isAvailable(LOG_LV2, b)) { g_debug_logger.printLog(a); };
	#define printlog_lvs3(a, b)		if (g_debug_logger.isAvailable(LOG_LV3, b)) { g_debug_logger.printLog(a); };
	#define printlog_lvs4(a, b)		if (g_debug_logger.isAvailable(LOG_LV4, b)) { g_debug_logger.printLog(a); };

	#define single_log(a, b)		CSingleLogger q(g_debug_logger.isAvailableLevel(b) ? (a) : "");
	#define singlelog_lv0(a)		CSingleLogger q(g_debug_logger.isAvailableLevel(LOG_LV0) ? (a) : "");
	#define singlelog_lv1(a)		CSingleLogger q(g_debug_logger.isAvailableLevel(LOG_LV1) ? (a) : "");
	#define singlelog_lv2(a)		CSingleLogger q(g_debug_logger.isAvailableLevel(LOG_LV2) ? (a) : "");
	#define singlelog_lv3(a)		CSingleLogger q(g_debug_logger.isAvailableLevel(LOG_LV3) ? (a) : "");
	#define singlelog_lv4(a)		CSingleLogger q(g_debug_logger.isAvailableLevel(LOG_LV4) ? (a) : "");

	#define singlelog_lvs0(a, b)	CSingleLogger q(g_debug_logger.isAvailable(LOG_LV0, b) ? (a) : "");
	#define singlelog_lvs1(a, b)	CSingleLogger q(g_debug_logger.isAvailable(LOG_LV1, b) ? (a) : "");
	#define singlelog_lvs2(a, b)	CSingleLogger q(g_debug_logger.isAvailable(LOG_LV2, b) ? (a) : "");
	#define singlelog_lvs3(a, b)	CSingleLogger q(g_debug_logger.isAvailable(LOG_LV3, b) ? (a) : "");
	#define singlelog_lvs4(a, b)	CSingleLogger q(g_debug_logger.isAvailable(LOG_LV4, b) ? (a) : "");
	
	#define simplelog(a)			g_debug_logger.printLog(a)
	#define debug_log(a)			g_debug_logger.printLog(a, true)
	#define get_loglevel()			g_debug_logger.getLogLevel()
	#define set_loglevel(x)			g_debug_logger.setLogLevel(x)
	#define get_logscope()			g_debug_logger.getLogScope()
	#define set_logscope(x)			g_debug_logger.setLogScope(x)
	#define add_logscope(x)			g_debug_logger.addLogScope(x)
	#define remove_logscope(x)		g_debug_logger.removeLogScope(x)
	#define chk_logscope(x)			g_debug_logger.isAvailableScope(x)
	#define chk_loglevel(x)			g_debug_logger.isAvailableLevel(x)
	#define chk_logcond(x, y)		g_debug_logger.isAvailable(x, y)

	#define keep_time(x)			g_time_logger.timeStart(x)
	#define leave_time(x)			g_time_logger.timeStop(x)
	#define tm_time(x)				g_time_logger.getTickData(x)
	#define get_cur_time(x)			g_time_logger.getTickData(x)->last_tick_time
	#define clear_time(x)			g_time_logger.getTickData(x)->reset
	#define write_all_time			g_time_logger.debugOutput();
	#define write_clear_time(x)		g_time_logger.debugOutput(); g_time_logger.clear(x);
	#define pop_all_time()			g_time_logger.debugOutput(); g_time_logger.clear();

	#define name_chktime(a, b)		g_time_logger.timeName(a, b)

#else
	#define printlog_lv0(a)
	#define printlog_lv1(a)
	#define printlog_lv2(a)
	#define printlog_lv3(a)
	#define printlog_lv4(a)

	#define printlog_lvs0(a, b)
	#define printlog_lvs1(a, b)
	#define printlog_lvs2(a, b)
	#define printlog_lvs3(a, b)
	#define printlog_lvs4(a, b)

	#define single_log(a, b)
	#define singlelog_lv0(a)
	#define singlelog_lv1(a)
	#define singlelog_lv2(a)
	#define singlelog_lv3(a)
	#define singlelog_lv4(a)

	#define singlelog_lvs0(a, b)
	#define singlelog_lvs1(a, b)
	#define singlelog_lvs2(a, b)
	#define singlelog_lvs3(a, b)
	#define singlelog_lvs4(a, b)

	#define debug_log(x)
	#define simplelog(a)			
	#define get_loglevel()			0	
	#define set_loglevel(x)			
	#define get_logscope()			0
	#define set_logscope(x)			
	#define add_logscope(x)			
	#define remove_logscope(x)		
	#define chk_logscope(x)			0

	#define keep_time(x)
	#define leave_time(x)
	#define tm_time(x)
	#define get_cur_time(x)
	#define clear_time(x)
	#define write_all_time
	#define write_clear_time(x)

	#define name_chktime(a, b)
#endif

#define LOG_LV0						0x00		//ring0: 체계의 구성부분의 초기, 해체, 스레드의 본체부 
#define LOG_LV1						0x01		//ring1: 특수경우의 사건흐름, 체계준위의 오유
#define LOG_LV2						0x02		//ring2: 체계준위의 사건흐름, 모듈준위의 오유
#define LOG_LV3						0x03		//ring3: 모듈준위의 사건흐름
#define LOG_LV4						0x04		//ring4: 실시간자료

#define LOG_SCOPE_APP				0x0001
#define LOG_SCOPE_IPC				0x0002
#define LOG_SCOPE_NET				0x0004
#define LOG_SCOPE_CAM				0x0008
#define LOG_SCOPE_ENCODE			0x0010
#define LOG_SCOPE_IO				0x0020
#define LOG_SCOPE_DB				0x0040
#define LOG_SCOPE_FTP				0x0080
#define LOG_SCOPE_TAG				0x0100
#define LOG_SCOPE_IVS				0x0200
#define LOG_SCOPE_COMM				0x0400
#define LOG_SCOPE_OPC				0x1000
#define LOG_SCOPE_OVX				0x2000
#define LOG_SCOPE_ENGINE			0x4000
#define LOG_SCOPE_RESERVED3			0x8000

#define LOG_SCOPE_ALL				0xFFFF
#define LOG_SCOPE_NONE				0x0000

#define ERR_PREPARE(n)				{_err_rep##n = 0; _err_limit##n = 1;}
#define ERR_UNOCCUR(n)				{_err_rep##n = 0; _err_limit##n = 1;}
#define ERR_DEFINE(n)				i32 _err_rep##n, _err_limit##n;

#define ERR_OCCUR(n, a)				if (_err_rep##n > 10000000) \
									{ \
										_err_rep##n = 0; _err_limit##n = 1; \
									} \
									if (++_err_rep##n >= _err_limit##n) \
									{ \
										_err_limit##n *= 10; a; \
									}

#define ERR_OCCUR2(n, a, b)			if (_err_rep##n > 10000000) \
									{ \
										_err_rep##n = 0; _err_limit##n = 1; \
									} \
									if (++_err_rep##n >= _err_limit##n) \
									{ \
										_err_limit##n *= 10; a; b; \
									}

#define LOG_MAX_TIMER				30
#define LOG_MAX_PATH				512
#define LOG_QUEUE_COUNT				256

#define sys_keep_time				u64 qtm = g_elapsed_timer.elapsed()
#define sys_chk_time				qtm = g_elapsed_timer.elapsed()
#define sys_cur_time				g_elapsed_timer.elapsed()
#define sys_get_time_ms				g_elapsed_timer.elapsed()-qtm
#define sys_get_time				(f32)(g_elapsed_timer.elapsed()-qtm)/1000

#define get_fps1(x)					g_counter_fps1.fps(x);
#define get_fps2(x)					g_counter_fps2.fps(x);
#define get_fps3(x)					g_counter_fps3.fps(x);

//////////////////////////////////////////////////////////////////////////
enum LogModeType
{
	kLogModeDirect = 0,
	kLogModeQueued,
	kLogModeCount
};

class CLogger : public QThread
{
public:
	CLogger();
	virtual ~CLogger();

	void					initInstance(const postring& log_filename, LogModeType mode = kLogModeQueued);
	void					exitInstance();
	
	void					printLog(const QString& str, bool is_console_only = false);
	void					printLogWithArg(i32 log_level, const char* format, ...);

	bool					setLogLevel(i32 level);
	bool					setLogScope(i32 scope);
	void					addLogScope(i32 scope);
	void					removeLogScope(i32 scope);

	inline i32				getLogScope() { return m_log_scope; };
	inline i32				getLogLevel() { return m_log_level; };
	inline bool				isAvailableLevel(i32 lv) { return lv <= m_log_level; };
	inline bool				isAvailableScope(i32 scope) { return (m_log_scope & scope) == scope; };
	inline bool				isAvailable(i32 lv, i32 scope) { return isAvailableLevel(lv) && isAvailableScope(scope); };

private:
	void					run() Q_DECL_OVERRIDE;

	void					logFileOpen(i32 call_lv = 0);
	void					logFileClose();
	void					backupLogFile();

public:
	LogModeType				m_mode;
	bool					m_is_inited;
	std::atomic<bool>		m_is_threadcancel;
	std::atomic<i32>		m_log_level;
	std::atomic<i32>		m_log_scope;

	i64						m_filesize;
	postring				m_filename;
	FILE*					m_file_ptr;

	i32						m_log_stpos;
	i32						m_log_edpos;
	char					m_log_message[LOG_QUEUE_COUNT][LOG_MAX_PATH];
	
	QMutex					m_queue_mutex;
	QSemaphore*				m_queue_semaphore_ptr;
};

class CSingleLogger
{
public:
	CSingleLogger(const QString& str);
	virtual ~CSingleLogger();

public:
	QString					m_log_message;
};
extern CLogger g_debug_logger;

//////////////////////////////////////////////////////////////////////////
struct TMTick
{
	postring				tick_name;
	i32						tick_count;
	i64						start_tick_time;
	f32						avg_tick_time;
	f32						min_tick_time;
	f32						max_tick_time;
	f32						last_tick_time;

public:
	TMTick();

	void					reset();
	void					clear();
};
typedef std::vector<TMTick> TimeTickVec;

class CTimeLogger
{
public:
	CTimeLogger();
	~CTimeLogger();

	void					initInstance(const postring& log_filename);
	void					exitInstance();
	TMTick*					getTickData(i32 index);

	void					timeName(i32 index, const char* timer_name);
	void					timeStart(i32 index);
	void					timeStop(i32 index);
	void					debugOutput();
	void					clear(i32 st_index = 0);

public:
	bool					m_is_inited;
	TimeTickVec				m_time_tick_vec;

	postring				m_filename;
	FILE*					m_file_ptr;
};

//////////////////////////////////////////////////////////////////////////
class CCounterFps
{
public:
	CCounterFps();
	~CCounterFps();

	f32						fps();
	f32						fps(i32 interval);

private:
	i32						m_interval;
	i32						m_counter;
	u64						m_time_stamp;
	QElapsedTimer			m_elapsed_timer;
};

extern CTimeLogger			g_time_logger;
extern QElapsedTimer		g_elapsed_timer;
extern CCounterFps			g_counter_fps1;
extern CCounterFps			g_counter_fps2;
extern CCounterFps			g_counter_fps3;

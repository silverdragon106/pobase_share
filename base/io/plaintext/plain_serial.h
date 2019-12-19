#pragma once

#include <QThread>
#include <QMutex>
#include <QElapsedTimer>

#include "define.h"
#include "logger/logger.h"
#include "libserialport/serial.h"

class CPlainDevParam;
class CPlainTextSerial : public QThread
{
	Q_OBJECT
	ERR_DEFINE(0)
	ERR_DEFINE(1)

public:
	CPlainTextSerial();
	virtual ~CPlainTextSerial();

	bool						initInstance(CPlainDevParam* dev_param_ptr);
	void						exitInstance();

	bool						openDevice(serial::Serial* port, const CPlainDevParam& dev_param);
	void						closeDevice(serial::Serial* port);
	bool						restartDevice();

	bool						writeData(u8* buffer_ptr, i32 buffer_size);
	virtual bool				onReadData(u8*& buffer_ptr, i32 buffer_size);
	virtual void				onResetDevice();

	inline bool					isRunning() { return m_is_inited && QThread::isRunning(); };

protected:
	virtual void				run();

signals:
	void						reopenedDevice();
	void						deviceError(i32 subdev, i32 errtype, i32 value);

public slots:
	void						onReOpenedDevice();

public:
	CPlainDevParam*				m_dev_param_ptr;

	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;
	std::atomic<bool>			m_is_reopen_device;

	u8*							m_write_buffer_ptr;
	i32							m_write_buffer_size;
	POMutex						m_write_mutex;
};


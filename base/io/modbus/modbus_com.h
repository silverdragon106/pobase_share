#pragma once

#include <QThread>
#include <QMutex>
#include <QElapsedTimer>

#include "define.h"
#include "logger/logger.h"
#include "libserialport/serial.h"
#include "modbus_packet.h"

class CModbusDevParam;
class CModBusCom : public QThread
{
	Q_OBJECT
	ERR_DEFINE(0)
	ERR_DEFINE(1)

public:
	CModBusCom();
	virtual ~CModBusCom();

	bool						initInstance(CModbusDevParam* modbus_param_ptr, bool is_master, i32 scan_interval = MODBUS_INTERVAL);
	void						exitInstance();

	bool						openDevice(serial::Serial* port, const CModbusDevParam& dev_param);
	void						closeDevice(serial::Serial* port);
	bool						restartDevice();
	void						writeCmdToDevice(CModBusPacket* pak);
	void						writeErrResponse(CModBusPacket* pak, i32 errtype);

	bool						onAcceptData(u8*& buffer_ptr, i32 len, i32 dev, i32 cmd, bool& is_complete);
	virtual void				onReadPacket(CModBusPacket* pak) = 0;
	virtual void				onScanDeviceTimer();
	virtual void				onResetDevice();
	virtual CModBusPacket*		getClosedPacket();

	bool						hasQueue();
	inline bool					isRun() { return m_is_inited && QThread::isRunning(); };

protected:
	virtual void				run();

signals:
	void						reopenedDevice();
	void						deviceError(i32 subdev, i32 errtype, i32 value);

public slots:
	void						onReopenedDevice();

public:
	bool						m_is_master;
	i32							m_scan_interval;
	CModbusDevParam*			m_modbus_param_ptr;

	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;
	std::atomic<bool>			m_is_reopen_device;

	i32							m_buffer_pos1;
	i32							m_buffer_pos2;
	u8							m_cmd_len[POIO_CMDQUEUE];
	u8*							m_cmd_buffer_ptr;
	CModBusPacket*				m_packet_ptr;

	QMutex						m_queue_mutex;
};


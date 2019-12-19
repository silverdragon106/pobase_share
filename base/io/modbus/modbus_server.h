#pragma once

#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <QTcpServer>
#include <QTcpSocket>
#include <QUdpSocket>

#include "define.h"
#include "modbus_packet.h"
#include "logger/logger.h"

class CModbusDevParam;
class CModBusServer;
class CModBusTcpThread : public QThread
{
	Q_OBJECT
public:
	explicit CModBusTcpThread(i32 id, QObject *parent = 0);
	virtual ~CModBusTcpThread();

	bool						onAcceptData(u8*& buffer_ptr, i32 len, bool& bcomplete);
	void						onWritePacket(CModNetPacket* pak);
	void						cancelThread();

protected:
	void						run() Q_DECL_OVERRIDE;

public slots:
	void						readyRead();
	void						disConnected();

private:
	u8*							m_buffer_ptr;
	u8*							m_buffer_last_ptr;
	u8*							m_buffer_read_ptr;

	CModNetPacket*				m_packet_ptr;
	QTcpSocket*					m_socket_ptr;
	CModBusServer*				m_modbus_server_ptr;

	i32							m_socket_descriptor;
	std::atomic<bool>			m_is_thread_cancel;
};
typedef std::list<CModBusTcpThread*> ModbusTcpThreadList;

class CModBusUdpThread : public QThread
{
	Q_OBJECT
	ERR_DEFINE(0)

public:
	explicit CModBusUdpThread(QObject *parent = 0);
	virtual ~CModBusUdpThread();

	void						onWritePacket(CModNetPacket* pak, QHostAddress& sender, u16 sport);
	void						cancelThread();

protected:
	void						run() Q_DECL_OVERRIDE;

public slots:
	void						readyRead();

public:
	CModNetPacket*				m_paket_ptr;
	QUdpSocket*					m_socket_ptr;
	CModBusServer*				m_modbus_server_ptr;

	CModbusDevParam*			m_modbus_param_ptr;
	std::atomic<bool>			m_is_thread_cancel;
};

class CModBusServer : public QTcpServer
{
	Q_OBJECT

public:
	CModBusServer();
	virtual ~CModBusServer();

	bool						initInstance(i32 mode, CModbusDevParam* modbus_param_ptr);
	void						exitInstance();
	void						removeConnection(CModBusTcpThread* connection_ptr);

	virtual CModNetPacket*		onReadPacket(CModNetPacket* pak) = 0;
	
protected:
	void						incomingConnection(qintptr socket_descriptor) Q_DECL_OVERRIDE;   //This is where we deal with incoming connections

public:
	bool						m_is_inited;
	i32							m_net_device;
	CModbusDevParam*			m_modbus_param_ptr;

	CModBusUdpThread*			m_udp_thread_ptr;
	ModbusTcpThreadList			m_tcp_thread_list;
	QMutex						m_thread_mutex;
};
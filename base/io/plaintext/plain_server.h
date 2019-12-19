#pragma once

#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <QTcpServer>
#include <QTcpSocket>
#include <QUdpSocket>

#include "define.h"
#include "logger/logger.h"

class CPlainDevParam;
class CPlainTextServer;
class CPlainTcpThread : public QThread
{
	Q_OBJECT
public:
	explicit CPlainTcpThread(i32 id, QObject *parent = 0);
	virtual ~CPlainTcpThread();

	bool						writeData(u8* buffer_ptr, i32 buffer_size);
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

	QTcpSocket*					m_socket_ptr;
	CPlainTextServer*			m_plain_server_ptr;

	i32							m_socket_descriptor;
	std::atomic<bool>			m_is_thread_cancel;
};
typedef std::list<CPlainTcpThread*> PlainTcpThreadList;

class CPlainUdpThread : public QThread
{
	Q_OBJECT
	ERR_DEFINE(0)

public:
	explicit CPlainUdpThread(QObject *parent = 0);
	virtual ~CPlainUdpThread();

	bool						writeData(u8* buffer_ptr, i32 buffer_size);
	void						cancelThread();

protected:
	void						run() Q_DECL_OVERRIDE;

public slots:
	void						readyRead();

public:
	QUdpSocket*					m_socket_ptr;
	CPlainTextServer*			m_plain_server_ptr;

	CPlainDevParam*				m_plain_param_ptr;
	std::atomic<bool>			m_is_thread_cancel;

	POMutex						m_conn_mutex;
	bool						m_conn_isvalid;
	QHostAddress				m_conn_address;
	i32							m_conn_port;
};

class CPlainTextServer : public QTcpServer
{
	Q_OBJECT

public:
	CPlainTextServer();
	virtual ~CPlainTextServer();

	bool						initInstance(i32 mode, CPlainDevParam* plain_param_ptr);
	void						exitInstance();

	CPlainTcpThread*			getFirstTcpThread();
	void						removeConnection(CPlainTcpThread* connection_ptr);

	bool						writeData(u8* buffer_ptr, i32 buffer_size);
	virtual bool				onReadData(u8*& buffer_ptr, i32 buffer_size);

protected:
	void						incomingConnection(qintptr socket_descriptor) Q_DECL_OVERRIDE;   //This is where we deal with incoming connections

public:
	bool						m_is_inited;
	i32							m_net_device;
	CPlainDevParam*				m_plain_param_ptr;

	CPlainUdpThread*			m_udp_thread_ptr;
	PlainTcpThreadList			m_tcp_thread_list;
	QMutex						m_thread_mutex;
};
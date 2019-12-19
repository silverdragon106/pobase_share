#pragma once

#include "define.h"
#include <QObject>
#include "server.h"

struct Packet;
class CCmdServer : public CServer
{
	Q_OBJECT

public:
	CCmdServer();
	~CCmdServer();

	void					netClose();

	void					onReadPacket(Packet* packet_ptr, CConnection* connection_ptr); // override. read packet_ptr callback.
	void					onNewHLConnection(i32 conn, i32 conn_mode);
	void					onLostConnection(CConnection* connection_ptr);
	void					onCreatedServer();

	bool					isLocalConnection();
	bool					isAvailableConnect(i32 conn, i32 conn_mode, bool is_remotable = false);
	inline i32				getConnection() { return m_connection; };

signals:
	void					serverInited(i32 mode, i32 port);
	void					networkPreConnect(Packet* packet_ptr, i32 conn);
	void					networkConnect(Packet* packet_ptr, i32 conn);
	void					networkReadPacket(Packet* packet_ptr, i32 conn_mode);
	void					networkLostConnection();

public:
	std::atomic<i32>		m_connection;
	std::atomic<i32>		m_conn_mode;
	POMutex					m_conn_mutex;
};


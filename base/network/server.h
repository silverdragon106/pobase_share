#pragma once
#include <QMap>
#include <QMutex>
#include <QThread>
#include <QElapsedTimer>
#include "netlink/socket_group.h"
#include "define.h"
#include "connection.h"
#include "packet.h"

using namespace NL;

#define SERVER_MAX_ACCEPTS		100
#define SERVER_DEFAULT_PORT		0

class CServer : public QThread, public CConnectionListener
{
	Q_OBJECT

	friend class OnReadListenerServer;
	friend class OnAcceptListenerServer;
	friend class OnDisconnectListenerServer;

public:
	CServer(bool is_pak_omit = false);
	virtual ~CServer();

	void						netStart(i32 accept = SERVER_MAX_ACCEPTS, i32 port = SERVER_DEFAULT_PORT);
																			// start listening.
	virtual void				netClose();									// stop listening.

	void						sendPacket(i32 conn, Packet* packet_ptr);
	void						sendPacket(PacketQueueVector& packet_vec);
	void						clearOmitMap();
	i32							getConnections();

	void						serverListen(i32 time_ms);
	void						removeConnection(CConnection* connection_ptr);
	void						removeAllConnections();
	void						setMaxAccept(i32 count);
	void						checkConnections(i64 cur_time);
	void						registerOmitCmd(i32vector& omitcmds);
	void						ping();
	bool						isOmitable(i32 cmd);

	CConnection*				_findConnectionByID(i32 id);
	void						_removeConnection(CConnection* connection_ptr, bool remove_map);

	virtual void				onReadPacket(Packet* packet_ptr, CConnection* connection_ptr);		// ready to read packet_ptr.
	virtual void				onNewConnection(CConnection* connection_ptr);					// new connection was established.
	virtual void				onLostConnection(CConnection* connection_ptr);					// old connection was closed.
	virtual void				onCreatedServer() = 0;

private:
	void						run() Q_DECL_OVERRIDE;

	//////////////////////////////////////////////////////////////////////////
	// called functions in this Thread.
	void						onRead(Packet* packet_ptr, CConnection* connection_ptr);
	void						onConnected(CConnection* connection_ptr);
	void						onDisconnected(CConnection* connection_ptr);
	void						onError(ConnectionErrorType err, CConnection* connection_ptr);

protected:
	i32							m_port;
	i32							m_max_accept;
	bool*						m_cmd_cull_ptr;
	std::atomic<bool>			m_is_thread_cancel;
	
	SocketGroup					m_socket_group;
	QMap<i32, CConnection*>		m_connections;
	QElapsedTimer				m_elapsed_timer;
	POMutex						m_mutex_conn;

	CConnection*				m_last_conn_ptr;
};

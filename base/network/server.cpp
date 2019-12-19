#include "server.h"
#include "connection.h"
#include "base.h"

const i32 kPOServerPingInterval = 1500;
//////////////////////////////////////////////////////////////////////////

class OnAcceptListenerServer : public NL::SocketGroupCmd 
{
public:
	OnAcceptListenerServer()
	{
		m_max_accept = 0;
	}

	void exec(NL::Socket* socket, NL::SocketGroup* group_ptr, void* reference_ptr)
	{
		CServer* server_ptr = (CServer*)reference_ptr;
		if (m_max_accept > 0 && m_max_accept < server_ptr->m_socket_group.size())
		{
			return;
		}

		NL::Socket* new_connection_ptr = socket->accept();
        if (new_connection_ptr)
        {
            printlog_lv1("Network listener is accepted new socket");
            CConnection* conn_ptr = po_new CConnection(new_connection_ptr, server_ptr);
        }
	}

	void setMaxAccept(i32 count)
	{
		m_max_accept = count;
	}

public:
	i32	m_max_accept;
};

class OnDisconnectListenerServer : public NL::SocketGroupCmd 
{
	void exec(NL::Socket* socket, NL::SocketGroup* group_ptr, void* reference_ptr)
	{
		CServer* server_ptr = (CServer*)reference_ptr;
		server_ptr->onDisconnected((CConnection*)socket->getData());
	}
};

class OnReadListenerServer : public NL::SocketGroupCmd
{
	void exec(NL::Socket* socket, NL::SocketGroup* group_ptr, void* reference_ptr)
	{
		CServer* server_ptr = (CServer*)reference_ptr;
		CConnection* connection_ptr = (CConnection*)socket->getData();
		if (connection_ptr)
		{
			connection_ptr->onRead();
		}
	}
};

class OnWriteListenerServer : public NL::SocketGroupCmd
{
	void exec(NL::Socket* socket, NL::SocketGroup* group_ptr, void* reference_ptr)
	{
		CServer* server_ptr = (CServer*)reference_ptr;
		CConnection* connection_ptr = (CConnection*)socket->getData();
		if (connection_ptr)
		{
			if (!connection_ptr->onWrite())
			{
				server_ptr->_removeConnection(connection_ptr, true);
			}
		}
	}
};

//////////////////////////////////////////////////////////////////////////
CServer::CServer(bool use_packet_omit)
{
	m_max_accept = -1;
	m_is_thread_cancel = false;
	m_port = PO_NETWORK_PORT;
	
	m_last_conn_ptr = NULL;
	m_cmd_cull_ptr = NULL;

	if (use_packet_omit)
	{
		m_cmd_cull_ptr = po_new bool[kPOCmdCount];
	}

	clearOmitMap();
	CConnection::initNL();
}

CServer::~CServer()
{
	netClose();
	POSAFE_DELETE_ARRAY(m_cmd_cull_ptr);
}

// start listening.
void CServer::netStart(i32 accept, i32 port)
{
	if (isRunning())
	{
		return;
	}

	m_port = port;
	m_max_accept = accept;
	m_is_thread_cancel = false;
	start();
}

// stop listening.
void CServer::netClose()
{
	if (isRunning())
	{
		m_is_thread_cancel = true;
		wait();
	}
	{
		POMutexLocker l(m_mutex_conn);
		QMap<i32, CConnection*>::Iterator iter;
		for (iter = m_connections.begin(); iter != m_connections.end(); iter++)
		{
			CConnection* connection_ptr = *iter;
			Socket* sock_ptr = connection_ptr->getSocket();
			m_socket_group.remove(sock_ptr);
			connection_ptr->setListener(NULL);
			connection_ptr->setSocket(NULL);
			POSAFE_DELETE(sock_ptr);

			onLostConnection(connection_ptr);
		}

		clearOmitMap();
		m_socket_group.clear();
		m_connections.clear();
	}
}

void CServer::clearOmitMap()
{
	if (!m_cmd_cull_ptr)
	{
		return;
	}
	memset(m_cmd_cull_ptr, 0, kPOCmdCount);
}

// ready to read packet_ptr.
void CServer::onReadPacket(Packet* packet_ptr, CConnection* connection_ptr)
{
}

// new connection was established.
void CServer::onNewConnection(CConnection* connection_ptr)
{
	printlog_lv1("Network ready command send");
	i32 conn = connection_ptr->getID();
	Packet* pak = po_new Packet(kPOCmdReady, kPOPacketRespOK);
	sendPacket(conn, pak);
}

// old connection was closed.
void CServer::onLostConnection(CConnection* connection_ptr)
{
	if (m_last_conn_ptr == connection_ptr)
	{
		m_last_conn_ptr = NULL;
	}
	POSAFE_DELETE(connection_ptr);
}

void CServer::run()
{
	singlelog_lv0("The Server thread is");

	try 
	{
		i64 cur_time = 0;
		i64 last_time = 0;
		OnReadListenerServer onRead;
		OnWriteListenerServer onWrite;
		OnAcceptListenerServer onAccept;
		OnDisconnectListenerServer onDisconnect;
		Socket server_socket(m_port);

		onAccept.setMaxAccept(m_max_accept);

		m_socket_group.setCmdOnAccept(&onAccept);
		m_socket_group.setCmdOnRead(&onRead);
		m_socket_group.setCmdOnWrite(&onWrite);
		m_socket_group.setCmdOnDisconnect(&onDisconnect);

		m_port = server_socket.portFrom();
		this->onCreatedServer();
		
		m_socket_group.add(&server_socket);

		while (!m_is_thread_cancel)
		{
			//read and action
			serverListen(1);

			// send ping packet_ptr to all connections for checking physical network connection
			cur_time = sys_cur_time;
			if (cur_time - last_time > kPOServerPingInterval)
			{
				ping();
				last_time = cur_time;

#if defined(POR_PRODUCT)
				//check disconnect all connection
				checkConnections(cur_time);
#endif
			}
			QThread::msleep(1);
		}
	}
	catch (NL::Exception e)
	{
		printlog_lv1(QString("Server Exception: code%1, %2").arg(e.code()).arg(e.what()));
	}
}

void CServer::onRead(Packet* packet_ptr, CConnection* connection_ptr)
{
	if (!packet_ptr || !connection_ptr)
	{
		return;
	}
	onReadPacket(packet_ptr, connection_ptr);
}

void CServer::onConnected(CConnection* connection_ptr)
{
	if (connection_ptr && connection_ptr->getSocket())
	{	
		m_mutex_conn.lock();
		Socket* sock_ptr = connection_ptr->getSocket();
		if (m_connections.contains(connection_ptr->getID()))
		{
			m_mutex_conn.unlock();

			// error. duplicate connection id.
			connection_ptr->setListener(NULL);
			POSAFE_DELETE(sock_ptr);
			POSAFE_DELETE(connection_ptr);
			printlog_lv1(QString("Error. duplicated connection id. %1").arg(m_connections.size()));
		}
		else
		{
			m_socket_group.add(sock_ptr);
			m_connections.insert(connection_ptr->getID(), connection_ptr);
			connection_ptr->setListener(this);
			m_mutex_conn.unlock();

			// you must set id to connection_ptr in OnNewConnection() call.
			onNewConnection(connection_ptr);
		}
	}
	else
	{
		printlog_lv1("Error. connection socket is invalid.");
	}
}

void CServer::onDisconnected(CConnection* connection_ptr)
{
	if (connection_ptr)
	{
		Socket* sock_ptr = connection_ptr->getSocket();
		m_socket_group.remove(sock_ptr);
		{
			POMutexLocker l(m_mutex_conn);
			m_connections.remove(connection_ptr->getID());
		}
		connection_ptr->setListener(NULL);
		connection_ptr->setSocket(NULL);
		POSAFE_DELETE(sock_ptr);

		onLostConnection(connection_ptr);
	}
}

void CServer::onError(ConnectionErrorType err, CConnection* connection_ptr)
{
	if (m_last_conn_ptr == connection_ptr)
	{
		m_last_conn_ptr = NULL;
	}
}

i32 CServer::getConnections()
{
	POMutexLocker l(m_mutex_conn);
	return m_connections.size();
}

CConnection* CServer::_findConnectionByID(i32 id)
{
	if (m_last_conn_ptr && m_last_conn_ptr->getID() == id)
	{
		return m_last_conn_ptr;
	}

	if (m_connections.contains(id))
	{
		m_last_conn_ptr = m_connections[id];
		return m_last_conn_ptr;
	}
	return NULL;
}

void CServer::serverListen(i32 time_ms)
{
	m_socket_group.listen(time_ms, this);
}

void CServer::ping()
{
	POMutexLocker l(m_mutex_conn);
	QMap<i32, CConnection*>::Iterator iter;
	for (iter = m_connections.begin(); iter != m_connections.end(); iter++)
	{
		CConnection* connection_ptr = *iter;
		Packet* pak = po_new Packet(kPOCmdPing, kPOPacketRespOK);
		sendPacket(connection_ptr->getID(), pak);
	}
}

void CServer::checkConnections(i64 cur_time)
{
	POMutexLocker l(m_mutex_conn);
	QMap<i32, CConnection*>::Iterator iter;
	for (iter = m_connections.begin(); iter != m_connections.end();)
	{
		CConnection* connection_ptr = *iter;
		if (connection_ptr->m_last_read_time > 0 &&
			cur_time - connection_ptr->m_last_read_time > 5 * kPOServerPingInterval)
		{
			iter = m_connections.erase(iter);
			_removeConnection(connection_ptr, false);
		}
		else
		{
			iter++;
		}
	}
}

void CServer::removeConnection(CConnection* connection_ptr)
{
	POMutexLocker l(m_mutex_conn);
	_removeConnection(connection_ptr, true);
}

void CServer::_removeConnection(CConnection* connection_ptr, bool remove_map)
{
	if (!connection_ptr)
	{
		return;
	}

	Socket* sock_ptr = connection_ptr->getSocket();
	printlog_lvs2(QString("Disconnect Network connection %1: timeout").arg(sock_ptr->hostFrom().c_str()),
					LOG_SCOPE_NET);

	if (remove_map)
	{
		m_connections.remove(connection_ptr->getID());
	}

	m_socket_group.remove(sock_ptr);
	connection_ptr->setListener(NULL);
	connection_ptr->setSocket(NULL);
	POSAFE_DELETE(sock_ptr);
	onLostConnection(connection_ptr);
}

void CServer::removeAllConnections()
{
	POMutexLocker l(m_mutex_conn);
	QMap<i32, CConnection*>::Iterator iter;
	for (iter = m_connections.begin(); iter != m_connections.end(); iter++)
	{
		CConnection* connection_ptr = *iter;
		_removeConnection(connection_ptr, false);
	}
	m_connections.clear();
}

void CServer::setMaxAccept(i32 count)
{
	m_max_accept = count;
}

void CServer::sendPacket(PacketQueueVector& packet_vec)
{
	i32 i, count = (i32)packet_vec.size();
	if (!CPOBase::isPositive(count))
	{
		return;
	}

	POMutexLocker l(m_mutex_conn);
	CConnection* conn_ptr = _findConnectionByID(packet_vec[0].conn);
	if (conn_ptr)
	{
		conn_ptr->writeLock();
		for (i = 0; i < count; i++)
		{
			PacketQueueItem& item = packet_vec[i];
			conn_ptr->sendPacket(item.pak, isOmitable(item.pak->getCmd()), false);
		}
		conn_ptr->writeUnlock();
	}
	else
	{
		//free unsent packets
		for (i = 0; i < count; i++)
		{
			PacketQueueItem& item = packet_vec[i];
			POSAFE_DELETE(item.pak);
		}
	}
}

void CServer::sendPacket(i32 conn, Packet* packet_ptr)
{
	if (!packet_ptr || !conn)
	{
		POSAFE_DELETE(packet_ptr);
		return;
	}

	POMutexLocker l(m_mutex_conn);
	CConnection* conn_ptr = _findConnectionByID(conn);
	if (!conn_ptr)
	{
		//free unsent packet
		POSAFE_DELETE(packet_ptr);
		return;
	}
	
	//testpoint
	//if (cmd == 581)
	//{
	//	debug_log(QString("cmd:%1, len:%2").arg(packet_ptr->getCmd()).arg(packet_ptr->getDataLen()));
	//}
	conn_ptr->sendPacket(packet_ptr, isOmitable(packet_ptr->getCmd()));
}

void CServer::registerOmitCmd(i32vector& omit_cmd_vec)
{
	if (!m_cmd_cull_ptr)
	{
		return;
	}

	clearOmitMap();

	i32 i, omit_cmd;
	i32 count = (i32)omit_cmd_vec.size();
	for (i = 0; i < count; i++)
	{
		omit_cmd = omit_cmd_vec[i];
		if (CPOBase::checkIndex(omit_cmd, kPOCmdCount))
		{
			m_cmd_cull_ptr[omit_cmd] = true;
		}
	}
}

bool CServer::isOmitable(i32 cmd)
{
	if (!m_cmd_cull_ptr || !CPOBase::checkIndex(cmd, kPOCmdCount))
	{
		return false;
	}
	return m_cmd_cull_ptr[cmd];
}

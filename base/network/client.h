#pragma once

#include "connection.h"
#include "netlink/socket_group.h"

#include <QMutex>
#include <QString>
#include <QThread>
#include <QElapsedTimer>
class CClient : public QThread, public CConnectionListener
{
	friend class OnReadListenerClient;
	friend class OnDisconnectListenerClient;
public:
	CClient();
	virtual ~CClient();

	void						setConnectionRetryCnt(int n);
	void						connect(postring ip_addr, int port);		// connect to the host.
	void						disconnect();							// disconnect from the host.

	bool						ping();
	virtual void				send(Packet* packet);					// send packet
	bool						isConnected();							// is connected
	bool						hasToStop();
	void						setStop(bool bStop);

	virtual void				onPacketReceived(Packet* packet);
	virtual void				onConnected();
	virtual void				onDisconnected();

	void						netStart();
	void						netClose();

	int							getLocalIP();
	int							getLocalPort();
	const postring&				getHostIP();
	int							getHostPort();
private:
	void						run();

	virtual void				onRead(Packet* packet, CConnection* connection);			// ready to read packet.
	virtual void				onConnected(CConnection* connection);
	virtual void				onDisconnected(CConnection* connection);
	virtual void				onError(int err);
private:
	std::atomic<bool>			m_has_to_stop;

	u64							m_last_time;
	QElapsedTimer				m_elapsed_timer;

	CConnection*				m_connection;
	SocketGroup					m_socket_group;
	postring					m_host_ip;
	int							m_host_port;
	int							m_connect_retry_count;

	postring					m_local_ip;
	int							m_local_port;
};

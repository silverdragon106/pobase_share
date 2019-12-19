#include "client.h"
#include "connection.h"
#include "packet.h"
#include "base.h"
#include "logger.h"
#include "define.h"

#include <QtCore>

#define POCLIENT_PINGINTERVAL			1500
//////////////////////////////////////////////////////////////////////////
class OnDisconnectListenerClient : public NL::SocketGroupCmd 
{
	void exec(NL::Socket* socket, NL::SocketGroup* group, void* reference)
	{
		group->remove(socket);

		CClient* client = (CClient*)reference;
		client->onDisconnected(client->m_connection);
		//str << "DisConnected " << socket->hostTo() << ":" << socket->portTo() << endl;
	}
};

class OnReadListenerClient : public NL::SocketGroupCmd
{
public:
	void exec(NL::Socket* , NL::SocketGroup* , void* reference)
	{
		CClient* client = (CClient*)reference;
		CConnection* connection = client->m_connection;
		if (connection)
		{
			connection->onRead();
		}
	}
};

//////////////////////////////////////////////////////////////////////////
CClient::CClient()
{
	m_connection = NULL;
	m_has_to_stop = false;
	m_connect_retry_count = -1;
	CConnection::initNL();
	m_last_time = 0;
	m_elapsed_timer.start();
}

CClient::~CClient()
{
	disconnect();
}

// connect to the host
void CClient::connect(postring ip_addr, int port)
{
	m_host_ip = ip_addr;
	m_host_port = port;

	netStart();
}

// disconnect from the host
void CClient::disconnect()
{
	netClose();

	POSAFE_DELETE(m_connection);
}

// ready to read buffer
void CClient::onRead(Packet* packet, CConnection* )
{
	onPacketReceived(packet);
}

void CClient::onConnected(CConnection* )
{
	setStop(false);
	onConnected();
}
void CClient::onPacketReceived(Packet*)
{

}

void CClient::onConnected()
{
}

void CClient::onDisconnected()
{
}


// start listening.
void CClient::netStart()
{
	if (isRunning())
	{
		return;
	}

	m_has_to_stop = false;
	start();
}

// stop listening.
void CClient::netClose()
{
	if (isRunning())
	{
		setStop(true);
		wait();
	}
	m_socket_group.clear();
}


void CClient::onDisconnected(CConnection* )
{
	if (isConnected())
	{
		setStop(true);
		onDisconnected();
	}
}

void CClient::onError(int )
{

}
bool CClient::ping()
{
	u64 currTime = m_elapsed_timer.elapsed();

	if (currTime - m_last_time < POCLIENT_PINGINTERVAL)
		return true;

	m_last_time = currTime;
	Packet *p = new Packet(kPOCmdPing, kPOPacketRequest);
	p->setReservedi64(0, currTime);

	send(p);
	return true;
}

// send string
void CClient::send(Packet* packet)
{
	if (isConnected())
	{
		m_connection->sendPacket(packet, false);
	}
	else
	{
		POSAFE_DELETE(packet);
	}
}

// is connected
bool CClient::isConnected()
{
	return (m_connection != NULL ? true : false);
}

bool CClient::hasToStop()
{
	return m_has_to_stop;
}

void CClient::setStop(bool bStop)
{
	m_has_to_stop = bStop;
}

void CClient::setConnectionRetryCnt(int n)
{
	m_connect_retry_count = n;
}

void CClient::run()
{
	int retry_cnt = m_connect_retry_count;
	while (!hasToStop())
	{
		try
		{
			SocketGroup socket_group;
			Socket socket(m_host_ip, m_host_port);// , 1000); // async connect

            OnReadListenerClient onRead;
			OnDisconnectListenerClient onDisconnect;

			socket_group.add(&socket);
			socket_group.setCmdOnRead(&onRead);
			socket_group.setCmdOnDisconnect(&onDisconnect);

			m_local_ip = socket.hostFrom().data();
			m_local_port = socket.portFrom();

			//socket.blocking(false);

			m_connection = new CConnection(&socket, this);
			while (isConnected() && !hasToStop())
			{
				socket_group.listen(1, this);

 				if (!ping())
 					break;

				m_connection->onWrite();

                SLEEP(1);
			}

			POSAFE_DELETE(m_connection);
		}
		catch (NL::Exception e)
		{
			simplelog("Error occured in CmdClient Socket.");
			POSAFE_DELETE(m_connection);
			onError(kConnectionErrUnknown);
		}

		if (!hasToStop())
		{
			if (retry_cnt > 0)
			{
				retry_cnt--;
			}
			else if (retry_cnt <= 0)
			{
				break;
			}

			SLEEP(500);
		}
	}

	/* Process connection timeout */
	if (retry_cnt <= 0 && !hasToStop())
	{
		onError(kConnectionTimeout);
	}
}

const postring& CClient::getHostIP()
{
	return m_host_ip;
}

int CClient::getLocalIP()
{
	return po::IpA2N(m_local_ip.data());
}

int CClient::getLocalPort()
{
	return m_local_port;
}

int CClient::getHostPort()
{
	return m_host_port;
}

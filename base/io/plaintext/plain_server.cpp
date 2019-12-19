#include "plain_server.h"
#include "struct.h"
#include "base.h"

CPlainTextServer::CPlainTextServer() : QTcpServer(NULL)
{
	m_is_inited = false;
	m_net_device = kPOCommNone;
	m_plain_param_ptr = NULL;
	m_udp_thread_ptr = NULL;
	m_tcp_thread_list.clear();
}

CPlainTextServer::~CPlainTextServer()
{
	exitInstance();
}

bool CPlainTextServer::initInstance(i32 mode, CPlainDevParam* plain_param_ptr)
{
	if (!plain_param_ptr)
	{
		return false;
	}

	if (!m_is_inited)
	{
		m_net_device = mode;
		m_plain_param_ptr = plain_param_ptr; //init before communication init

		switch (mode)
		{
			case kPOCommPlainTcp:
			{
				if (!listen(QHostAddress::Any, plain_param_ptr->getTcpPort()))
				{
					printlog_lvs2(QString("Unable to start PlainTCP, port:%1").arg(errorString()), LOG_SCOPE_FTP);
					return false;
				}
				break;
			}
			case kPOCommPlainUdp:
			{
				m_udp_thread_ptr = po_new CPlainUdpThread(this);
				connect(m_udp_thread_ptr, SIGNAL(finished()), m_udp_thread_ptr, SLOT(deleteLater()));
				m_udp_thread_ptr->start();
			}
		}
		
		m_is_inited = true;
	}
	return true;
}

void CPlainTextServer::exitInstance()
{
	if (m_is_inited)
	{
		switch (m_net_device)
		{
			case kPOCommPlainTcp:
			{
				close(); //close listening
				{
					QMutexLocker l(&m_thread_mutex);

					PlainTcpThreadList::iterator iter;
					CPlainTcpThread* tcp_thread_ptr;
					for (iter = m_tcp_thread_list.begin(); iter != m_tcp_thread_list.end(); iter++)
					{
						tcp_thread_ptr = *iter;
						tcp_thread_ptr->cancelThread();
					}
					m_tcp_thread_list.clear();
				}
			
				break;
			}
			case kPOCommPlainUdp:
			{
				if (m_udp_thread_ptr)
				{
					m_udp_thread_ptr->cancelThread();
					m_udp_thread_ptr = NULL;
				}
				break;
			}
		}
		m_is_inited = false;
	}
}

void CPlainTextServer::removeConnection(CPlainTcpThread* connection_ptr)
{
	connection_ptr->cancelThread();

	QMutexLocker l(&m_thread_mutex);
	m_tcp_thread_list.remove(connection_ptr);
}

void CPlainTextServer::incomingConnection(qintptr socket_descriptor)
{
	if (m_net_device != kPOCommPlainTcp)
	{
		return;
	}

	//Every new connection will be run in a newly created thread
	CPlainTcpThread* thread_ptr = po_new CPlainTcpThread(socket_descriptor, this);
	{
		QMutexLocker l(&m_thread_mutex);
		m_tcp_thread_list.push_back(thread_ptr);
	}

	//once a thread is not needed, it will be deleted later
	connect(thread_ptr, SIGNAL(finished()), thread_ptr, SLOT(deleteLater()));
	thread_ptr->start();
}

bool CPlainTextServer::onReadData(u8*& buffer_ptr, i32 buffer_size)
{
	return false;
}

bool CPlainTextServer::writeData(u8* buffer_ptr, i32 buffer_size)
{
	if (!m_is_inited || !buffer_ptr || buffer_size <= 0)
	{
		return false;
	}
	switch (m_net_device)
	{
		case kPOCommPlainTcp:
		{
			CPlainTcpThread* tcp_thread_ptr = getFirstTcpThread();
			if (!tcp_thread_ptr)
			{
				return false;
			}
			return tcp_thread_ptr->writeData(buffer_ptr, buffer_size);
		}
		case kPOCommPlainUdp:
		{
			if (!m_udp_thread_ptr)
			{
				return false;
			}
			return m_udp_thread_ptr->writeData(buffer_ptr, buffer_size);
		}
	}
	return false;
}

CPlainTcpThread* CPlainTextServer::getFirstTcpThread()
{
	if (m_tcp_thread_list.size() <= 0)
	{
		return NULL;
	}
	return m_tcp_thread_list.front();
}

//////////////////////////////////////////////////////////////////////////
CPlainTcpThread::CPlainTcpThread(i32 id, QObject* parent_ptr) : QThread(parent_ptr)
{
	m_socket_ptr = NULL;
	m_socket_descriptor = id;
	m_plain_server_ptr = (CPlainTextServer*)parent_ptr;

	m_buffer_read_ptr = po_new u8[POIO_READSIZE];
	m_buffer_ptr = m_buffer_read_ptr;
	m_buffer_last_ptr = m_buffer_read_ptr;

	m_is_thread_cancel = false;
}

CPlainTcpThread::~CPlainTcpThread()
{
	POSAFE_DELETE_ARRAY(m_buffer_read_ptr);
}

void CPlainTcpThread::run()
{
	singlelog_lv0("The PlainTCP thread is");

	m_socket_ptr = po_new QTcpSocket();
	m_socket_ptr->moveToThread(this);

	if (m_socket_ptr->setSocketDescriptor(m_socket_descriptor))
	{
		m_socket_ptr->setSocketOption(QAbstractSocket::LowDelayOption, 1);
		m_socket_ptr->setSocketOption(QAbstractSocket::KeepAliveOption, 1);
		connect(m_socket_ptr, SIGNAL(readyRead()), this, SLOT(readyRead()), Qt::DirectConnection);
		connect(m_socket_ptr, SIGNAL(disconnected()), this, SLOT(disConnected()));
	}
	exec();

	m_socket_ptr->close();
	m_socket_ptr->deleteLater();
}

void CPlainTcpThread::readyRead()
{
	i32 read_bytes, max_read_bytes;
	max_read_bytes = POIO_READSIZE - (m_buffer_ptr - m_buffer_read_ptr);
	read_bytes = m_socket_ptr->read((char*)m_buffer_ptr, max_read_bytes);
	if (read_bytes > 0)
	{
		m_buffer_ptr += read_bytes;
		m_plain_server_ptr->onReadData(m_buffer_last_ptr, m_buffer_ptr - m_buffer_last_ptr);

		//[start] read_buffer--[readed packet_ptr]--last_buffer--[reading packet_ptr]--buffer--[end]
		i32 lastlen = m_buffer_ptr - m_buffer_last_ptr;
		if (lastlen >= POIO_HALFREADSIZE) //check input data size
		{
			m_buffer_last_ptr = m_buffer_ptr - POIO_SMALLREADSIZE;
			lastlen = POIO_SMALLREADSIZE;
		}
		if (m_buffer_last_ptr - m_buffer_read_ptr > POIO_HALFREADSIZE) //check input data position
		{
			CPOBase::memCopy(m_buffer_read_ptr, m_buffer_last_ptr, lastlen);
			m_buffer_last_ptr = m_buffer_read_ptr;
			m_buffer_ptr = m_buffer_read_ptr + lastlen;
			printlog_lv1(QString("Plain NetModule ReadBuffer was swaped."));
		}
	}
}

bool CPlainTcpThread::writeData(u8* buffer_ptr, i32 buffer_size)
{
	if (!buffer_ptr || buffer_size <= 0)
	{
		return false;
	}
	if (!m_socket_ptr)
	{
		return false;
	}
	i32 wbytes = m_socket_ptr->write((char*)buffer_ptr, buffer_size);
	return (wbytes == buffer_size);
}

void CPlainTcpThread::disConnected()
{
	((CPlainTextServer*)parent())->removeConnection(this);
}

void CPlainTcpThread::cancelThread()
{
	m_is_thread_cancel = true;
	QEventLoopStop();
}

//////////////////////////////////////////////////////////////////////////
CPlainUdpThread::CPlainUdpThread(QObject *parent) : QThread(parent)
{
	ERR_PREPARE(0);
	m_is_thread_cancel = false;

	m_socket_ptr = NULL;
	m_plain_server_ptr = (CPlainTextServer*)parent;
	m_plain_param_ptr = m_plain_server_ptr->m_plain_param_ptr;

	{
		exlock_guard(m_conn_mutex);
		m_conn_isvalid = false;
		m_conn_address.clear();
		m_conn_port = 0;
	}
}

CPlainUdpThread::~CPlainUdpThread()
{
}

void CPlainUdpThread::run()
{
	singlelog_lv0("The PlainUDP thread is");
	if (!m_plain_param_ptr)
	{
		return;
	}

	m_socket_ptr = po_new QUdpSocket();
	m_socket_ptr->moveToThread(this);

	while (!m_is_thread_cancel)
	{
		m_conn_port = m_plain_param_ptr->getUdpPort();
		if (!m_socket_ptr->bind(m_conn_port))
		{
			ERR_OCCUR(0, printlog_lvs2(QString("Unable to UDPbind: %1, port:%2, [%3]")
							.arg(m_socket_ptr->errorString()).arg(m_conn_port).arg(_err_rep0), LOG_SCOPE_COMM));

			QThread::msleep(1000);
			continue;
		}
		ERR_UNOCCUR(0);
		break;
	}

	connect(m_socket_ptr, SIGNAL(readyRead()), this, SLOT(readyRead()), Qt::DirectConnection);
	exec();

	m_socket_ptr->close();
	m_socket_ptr->deleteLater();
}

void CPlainUdpThread::readyRead()
{
	QByteArray datagram;
	QHostAddress sender;
	quint16 sport;

	while (m_socket_ptr->hasPendingDatagrams())
	{
		datagram.resize(m_socket_ptr->pendingDatagramSize());
		m_socket_ptr->readDatagram(datagram.data(), datagram.size(), &sender, &sport);
		if (datagram.size() > 0)
		{
			{
				exlock_guard(m_conn_mutex);
				m_conn_isvalid = true;
				m_conn_address = sender;
			}

			bool is_complete = false;
			u8* buffer_ptr = (u8*)datagram.data();
			i32 buffer_size = datagram.size();
			m_plain_server_ptr->onReadData(buffer_ptr, buffer_size);
		}
	}
}

bool CPlainUdpThread::writeData(u8* buffer_ptr, i32 buffer_size)
{
	if (!buffer_ptr || buffer_size <= 0)
	{
		return false;
	}
	if (!m_socket_ptr)
	{
		return false;
	}

	i32 wbytes = 0;
	bool is_valid = false;
	QHostAddress address;
	{
		exlock_guard(m_conn_mutex);
		is_valid = m_conn_isvalid;
		address = m_conn_address;
	}
	if (is_valid)
	{
		wbytes = m_socket_ptr->writeDatagram((char*)buffer_ptr, buffer_size, address, m_conn_port);
	}
	return (wbytes == buffer_size);
}

void CPlainUdpThread::cancelThread()
{
	m_is_thread_cancel = true;
	if (m_socket_ptr)
	{
		QEventLoopStop();
	}
	{
		exlock_guard(m_conn_mutex);
		m_conn_isvalid = false;
	}
}
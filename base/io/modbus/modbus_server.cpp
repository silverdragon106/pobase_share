#include "modbus_server.h"
#include "struct.h"
#include "base.h"

CModBusServer::CModBusServer() : QTcpServer(NULL)
{
	m_is_inited = false;
	m_net_device = kPOCommNone;
	m_modbus_param_ptr = NULL;
	m_udp_thread_ptr = NULL;
	m_tcp_thread_list.clear();
}

CModBusServer::~CModBusServer()
{
	exitInstance();
}

bool CModBusServer::initInstance(i32 mode, CModbusDevParam* modbus_param_ptr)
{
	if (!modbus_param_ptr)
	{
		return false;
	}

	if (!m_is_inited)
	{
		m_net_device = mode;
		m_modbus_param_ptr = modbus_param_ptr; //init before communication init

		switch (mode)
		{
			case kPOCommModbusTcp:
			{
				if (!listen(QHostAddress::Any, modbus_param_ptr->getTcpPort()))
				{
					printlog_lvs2(QString("Unable to start ModbusTCP, port:%1").arg(errorString()), LOG_SCOPE_FTP);
					return false;
				}
				break;
			}
			case kPOCommModbusUdp:
			{
				m_udp_thread_ptr = po_new CModBusUdpThread(this);
				connect(m_udp_thread_ptr, SIGNAL(finished()), m_udp_thread_ptr, SLOT(deleteLater()));
				m_udp_thread_ptr->start();
			}
		}
		
		m_is_inited = true;
	}
	return true;
}

void CModBusServer::exitInstance()
{
	if (m_is_inited)
	{
		switch (m_net_device)
		{
			case kPOCommModbusTcp:
			{
				close(); //close listening
				{
					QMutexLocker l(&m_thread_mutex);

					ModbusTcpThreadList::iterator iter;
					CModBusTcpThread* tcp_thread_ptr;
					for (iter = m_tcp_thread_list.begin(); iter != m_tcp_thread_list.end(); iter++)
					{
						tcp_thread_ptr = *iter;
						tcp_thread_ptr->cancelThread();
					}
					m_tcp_thread_list.clear();
				}
			
				break;
			}
			case kPOCommModbusUdp:
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

void CModBusServer::removeConnection(CModBusTcpThread* connection_ptr)
{
	connection_ptr->cancelThread();

	QMutexLocker l(&m_thread_mutex);
	m_tcp_thread_list.remove(connection_ptr);
}

void CModBusServer::incomingConnection(qintptr socket_descriptor)
{
	if (m_net_device != kPOCommModbusTcp)
	{
		return;
	}

	//Every new connection will be run in a newly created thread
	CModBusTcpThread* thread_ptr = po_new CModBusTcpThread(socket_descriptor, this);
	{
		QMutexLocker l(&m_thread_mutex);
		m_tcp_thread_list.push_back(thread_ptr);
	}

	//once a thread is not needed, it will be deleted later
	connect(thread_ptr, SIGNAL(finished()), thread_ptr, SLOT(deleteLater()));
	thread_ptr->start();
}

//////////////////////////////////////////////////////////////////////////
CModBusTcpThread::CModBusTcpThread(i32 id, QObject* parent_ptr) :
QThread(parent_ptr)
{
	m_socket_ptr = NULL;
	m_socket_descriptor = id;
	m_modbus_server_ptr = (CModBusServer*)parent_ptr;

	m_buffer_read_ptr = po_new u8[POIO_READSIZE];
	m_buffer_ptr = m_buffer_read_ptr;
	m_buffer_last_ptr = m_buffer_read_ptr;

	m_packet_ptr = po_new CModNetPacket();
	m_is_thread_cancel = false;
}

CModBusTcpThread::~CModBusTcpThread()
{
	POSAFE_DELETE(m_packet_ptr);
	POSAFE_DELETE_ARRAY(m_buffer_read_ptr);
}

void CModBusTcpThread::run()
{
	singlelog_lv0("The ModbusTCP thread is");

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

void CModBusTcpThread::readyRead()
{
	bool is_complete;
	i32 read_bytes, max_read_bytes;
	max_read_bytes = m_buffer_read_ptr - m_buffer_ptr + POIO_READSIZE;
	read_bytes = m_socket_ptr->read((char*)m_buffer_ptr, max_read_bytes);
	if (read_bytes > 0)
	{
		is_complete = false;
		m_buffer_ptr += read_bytes;
		while (onAcceptData(m_buffer_last_ptr, m_buffer_ptr - m_buffer_last_ptr, is_complete));

		//[start] read_buffer--[readed packet_ptr]--last_buffer--[reading packet_ptr]--buffer--[end]
		i32 lastlen = m_buffer_ptr - m_buffer_last_ptr;
		if (lastlen >= POIO_HALFREADSIZE)
		{
			m_buffer_last_ptr = m_buffer_ptr - POIO_SMALLREADSIZE;
			lastlen = POIO_SMALLREADSIZE;
		}
		if (m_buffer_last_ptr - m_buffer_read_ptr >= POIO_HALFREADSIZE && is_complete)
		{
			CPOBase::memCopy(m_buffer_read_ptr, m_buffer_last_ptr, lastlen);
			m_buffer_last_ptr = m_buffer_read_ptr;
			m_buffer_ptr = m_buffer_read_ptr + lastlen;
			printlog_lv1(QString("ModBus NetModule ReadBuffer was swaped."));
		}
	}
}

bool CModBusTcpThread::onAcceptData(u8*& buffer_ptr, i32 len, bool& is_complete)
{
	//if readed buffer is empty, wait to read next data
	if (len == 0)
	{
		is_complete = true;
		return false;
	}

	//check current buffer
	is_complete = false;
	if (!m_packet_ptr->isModNetPacket(false, buffer_ptr, len))
	{
		return false;
	}

	//if current packet_ptr is invalid, increase buffer point when buffer has some error bytes...
	if (!m_packet_ptr->isValid())
	{
		buffer_ptr++;
		return true;
	}

	if (m_modbus_server_ptr)
	{
		CModNetPacket* pak = m_modbus_server_ptr->onReadPacket(m_packet_ptr);
		if (pak)
		{
			onWritePacket(pak);
			POSAFE_DELETE(pak);
		}
	}

	buffer_ptr += m_packet_ptr->m_len;
	is_complete = true;
	return true;
}

void CModBusTcpThread::onWritePacket(CModNetPacket* pak)
{
	if (!pak)
	{
		return;
	}

	m_socket_ptr->write((char*)pak->m_buffer_ptr, pak->m_len);
}

void CModBusTcpThread::disConnected()
{
	((CModBusServer*)parent())->removeConnection(this);
}

void CModBusTcpThread::cancelThread()
{
	m_is_thread_cancel = true;
	QEventLoopStop();
}

//////////////////////////////////////////////////////////////////////////
CModBusUdpThread::CModBusUdpThread(QObject *parent) : QThread(parent)
{
	ERR_PREPARE(0);
	m_is_thread_cancel = false;

	m_socket_ptr = NULL;
	m_modbus_server_ptr = (CModBusServer*)parent;
	m_modbus_param_ptr = m_modbus_server_ptr->m_modbus_param_ptr;
	m_paket_ptr = po_new CModNetPacket();
}

CModBusUdpThread::~CModBusUdpThread()
{
	POSAFE_DELETE(m_paket_ptr);
}

void CModBusUdpThread::run()
{
	singlelog_lv0("The ModbusUDP thread is");
	if (!m_modbus_param_ptr)
	{
		return;
	}

	m_socket_ptr = po_new QUdpSocket();
	m_socket_ptr->moveToThread(this);

	while (!m_is_thread_cancel)
	{
		i32 net_port = m_modbus_param_ptr->getUdpPort();
		if (!m_socket_ptr->bind(net_port))
		{
			ERR_OCCUR(0, printlog_lvs2(QString("Unable to UDPbind: %1, port:%2, [%3]")
							.arg(m_socket_ptr->errorString()).arg(net_port).arg(_err_rep0), LOG_SCOPE_COMM));

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

void CModBusUdpThread::readyRead()
{
	QByteArray datagram;
	QHostAddress sender;
	quint16 sport;

	while (m_socket_ptr->hasPendingDatagrams())
	{
		datagram.resize(m_socket_ptr->pendingDatagramSize());
		m_socket_ptr->readDatagram(datagram.data(), datagram.size(), &sender, &sport);
		if (m_paket_ptr->isModNetPacket(false, (u8*)datagram.data(), datagram.size()) && m_paket_ptr->isValid())
		{
			CModNetPacket* pak = m_modbus_server_ptr->onReadPacket(m_paket_ptr);
			if (pak)
			{
				onWritePacket(pak, sender, sport);
				POSAFE_DELETE(pak);
			}
		}
	}
}

void CModBusUdpThread::onWritePacket(CModNetPacket* pak, QHostAddress& sender, u16 port)
{
	if (!pak)
	{
		return;
	}

	m_socket_ptr->writeDatagram((char*)pak->m_buffer_ptr, pak->m_len, sender, port);
}

void CModBusUdpThread::cancelThread()
{
	m_is_thread_cancel = true;
	if (m_socket_ptr)
	{
		QEventLoopStop();
	}
}
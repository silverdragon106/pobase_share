#include "cmd_server.h"
#include "packet.h"
#include "os/qt_base.h"

CCmdServer::CCmdServer() : CServer(true)
{
	m_connection = 0;
	m_conn_mode = kPOConnNone;
}

CCmdServer::~CCmdServer()
{
}

void CCmdServer::netClose()
{
	CServer::netClose();
	m_connection = 0;
	m_conn_mode = kPOConnNone;
}

void CCmdServer::onReadPacket(Packet* packet_ptr, CConnection* connection_ptr)
{
	if (!packet_ptr || !connection_ptr)
	{
		POSAFE_DELETE(packet_ptr);
		return;
	}

	if (packet_ptr->getHeaderType() != kPOPacketRequest || packet_ptr->getDeviceType() != PO_CUR_DEVICE)
	{
		printlog_lv1(QString("Read Command Error: invalid header or device type, header:%1, device:%2")
						.arg(packet_ptr->getHeaderType()).arg(packet_ptr->getDeviceType()));
		POSAFE_DELETE(packet_ptr);
		_removeConnection(connection_ptr, true);
		return;
	}

	bool is_available = false;
	i32 cmd = packet_ptr->getCmd();
	i32 conn = connection_ptr->getID();
	i32 code = kPOErrCmdFail;

	switch (cmd)
	{
		case kPOCmdPing:
		{
			code = kPOSuccess;
			break;
		}
		case kPOCmdPreConnect:
		{
			is_available = true;
			emit networkPreConnect(packet_ptr, connection_ptr->getID());
			break;
		}
		case kPOCmdConnect:
		{
			is_available = true;
			emit networkConnect(packet_ptr, connection_ptr->getID());
			break;
		}
		default:
		{
			{
				POMutexLocker q(m_conn_mutex);
				is_available = (conn == m_connection);
			}

			if (is_available)
			{
				emit networkReadPacket(packet_ptr, m_conn_mode);
			}
			else
			{
				printlog_lv1(QString("---skip command[%1]").arg(cmd));
			}
			break;
		}
	}

	if (!is_available)
	{
		POSAFE_DELETE(packet_ptr);
	}
}

bool CCmdServer::isLocalConnection()
{
	POMutexLocker l(m_conn_mutex);
	return (m_connection == LOCALHOST);
}

bool CCmdServer::isAvailableConnect(i32 conn, i32 conn_mode, bool is_remotable)
{
	printlog_lvs2(QString("Available connection check. %1, %2, %3")
					.arg(conn).arg(conn_mode).arg(is_remotable), LOG_SCOPE_NET);

	POMutexLocker q(m_conn_mutex);
	if (conn == m_connection || conn == 0)
	{
		return false;
	}

	if (is_remotable)
	{
		/* 현재의 접속방식이 원래의 접속방식보다 우위여야 한다. */
		/* 외부접속이 가능한 경우, 원래의 접속이 없다면 접속가능 */
		if (conn_mode > m_conn_mode || m_connection == 0)
		{
			return true;
		}
		/* 원래접속이 있다면 이경우에는 외부->로컬, 로컬->외부이여야 한다.*/
		if (conn == LOCALHOST || m_connection == LOCALHOST)
		{
			return true;
		}
	}
	else
	{
		/* 외부접속이 불가능한 경우 로컬접속이여야 하며 현재의 접속방식이 원래의 접속방식보다 우위여야 한다. */
		if (conn == LOCALHOST && conn_mode > m_conn_mode)
		{
			return true;
		}
	}
	return false;
}

void CCmdServer::onNewHLConnection(i32 conn, i32 conn_mode)
{
	POMutexLocker l(m_mutex_conn);
	{
		//check current connection validation
		POMutexLocker q(m_conn_mutex);
		if (m_connection)
		{
			_removeConnection(_findConnectionByID(m_connection), true);
			printlog_lv1(QString("The current connection will be disconnect for new connection."));
		}

		m_connection = conn;
		m_conn_mode = conn_mode;
		if (!_findConnectionByID(conn))
		{
			m_conn_mode = kPOConnNone;
			m_connection = 0;
		}
	}
}

void CCmdServer::onLostConnection(CConnection* connection_ptr)
{
	{
		POMutexLocker q(m_conn_mutex);
		if (m_connection == connection_ptr->getID())
		{
			m_connection = 0;
			emit networkLostConnection();
		}
	}
	CServer::onLostConnection(connection_ptr);
}

void CCmdServer::onCreatedServer()
{
	emit serverInited(kPOServerCmd, m_port);
}

#include "ivs_server.h"
#include "packet.h"

CIVSServer::CIVSServer()
{
	m_login_conn = 0;
	m_can_update = false;
	m_is_hl_embedded = false;
}

CIVSServer::~CIVSServer()
{
}

void CIVSServer::onNewConnection(CConnection* conn_ptr)
{
	i32 conn = conn_ptr->getID();
	emit signalIVSChange(kIVSConnected, conn);
}

void CIVSServer::onLostConnection(CConnection* conn_ptr)
{
	i32 conn = conn_ptr->getID();
	CServer::onLostConnection(conn_ptr);
	emit signalIVSChange(kIVSDisconnected, conn);
}

void CIVSServer::onCreatedServer()
{
	emit serverInited(kPOServerIVS, m_port);
}

void CIVSServer::onReadPacket(Packet* packet_ptr, CConnection* conn_ptr)
{
	if (!packet_ptr || !conn_ptr )
	{
		POSAFE_DELETE(packet_ptr);
		return;
	}
	if (packet_ptr->getHeaderType() != kPOPacketRequest)
	{
		printlog_lv1("Read IVS Error: PacketHeaderType is not POR_PACKET_REQ");
		POSAFE_DELETE(packet_ptr);
		return;
	}

	emit signalIVSReadPacket(conn_ptr->getID(), packet_ptr);
}

void CIVSServer::uploadReturnPacket(i32 conn, i32 cmd, i32 code)
{
	Packet* pak = po_new Packet(cmd, kPOPacketRespOK);
	pak->setSubCmd(code);
	sendPacket(conn, pak);

	if (code >= kIVSReturnDevStatusErr)
	{
		printlog_lv1(QString("IVS response of command is failed. error code is %1").arg(code));
	}
}

void CIVSServer::updateState(i32 state, i64 data)
{
	POMutexLocker l(m_mutex_conn);

	i32 conn;
	QMap<i32, CConnection*>::Iterator iter;
	for (iter = m_connections.begin(); iter != m_connections.end(); iter++)
	{
		CConnection* conn_ptr = *iter;
		if (!conn_ptr)
		{
			continue;
		}
		
		Packet* pak = po_new Packet(kIVSCmdStatus, kPOPacketRespOK);
		pak->setSubCmd(state);
		pak->setReservedi64(0, data);
		pak->setReservedb8(8, m_can_update);
		pak->setReservedb8(9, m_is_hl_embedded);

		conn = conn_ptr->getID();
		if (state == kIVSStateLogin && conn != data)
		{
			data = 0;
			state = kIVSStateLogOut;
			pak->setSubCmd(state);
			pak->setReservedi64(0, data);
		}
		sendPacket(conn, pak);
		printlog_lvs2(QString("IVS[%1] update state[%2], data[%3].").arg(conn).arg(state).arg(data), LOG_SCOPE_NET);
	}
}

i32 CIVSServer::isLoginConn(i32 conn)
{
	QMutexLocker l(&m_login_mutex);
	if (m_login_conn == conn)
	{
		return kIVSLoginSelf;
	}
	if (m_login_conn != 0)
	{
		return kIVSLoginOther;
	}
	return kIVSLoginNone;
}

bool CIVSServer::setLoginConn(i32 conn)
{
	QMutexLocker l(&m_login_mutex);
	if (m_login_conn != conn && m_login_conn != 0)
	{
		return false;
	}

	m_login_conn = conn;
	updateState(kIVSStateLogin, conn);
	return true;
}

bool CIVSServer::setLogoutConn(i32 conn)
{
	QMutexLocker l(&m_login_mutex);
	if (m_login_conn != conn)
	{
		return false;
	}

	m_login_conn = 0;
	updateState(kIVSStateLogin, 0);
	return true;
}

void CIVSServer::setLogoutAll()
{
	QMutexLocker l(&m_login_mutex);
	m_login_conn = 0;
}

void CIVSServer::setHLEmbedded(bool is_hl_embedded, bool can_be_update)
{
	m_can_update = can_be_update;
	m_is_hl_embedded = is_hl_embedded;
}
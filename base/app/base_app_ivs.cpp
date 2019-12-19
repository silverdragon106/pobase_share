#include "base_app_ivs.h"
#include "base_app.h"
#include "base.h"

CBaseAppIVS::CBaseAppIVS()
{
}

CBaseAppIVS::~CBaseAppIVS()
{
}

CPOBaseApp* CBaseAppIVS::getBaseApp()
{
	return dynamic_cast<CPOBaseApp*>(this);
}

void CBaseAppIVS::onRecivedIVSChange(i32 mode, i32 conn)
{
	i64 send_data = 0;
	i32 state = kIVSStateNetFree;
	CPOBaseApp* base_app_ptr = getBaseApp();
	CIPInfo* ip_info_ptr = base_app_ptr->getIPInfo();
	if (!base_app_ptr || !ip_info_ptr)
	{
		printlog_lvs2("IVSChange failed. Invalid IpInfo", LOG_SCOPE_IVS);
		return;
	}

	if (base_app_ptr->isDeviceOffline())
	{
		state = kIVSStateNone;
		printlog_lvs2("The Device is offline", LOG_SCOPE_IVS);
	}
	else if (base_app_ptr->checkAppDesc(kPODescHighLevel))
	{
		state = kIVSStateBusy;
		send_data = ip_info_ptr->getHighID();
		printlog_lvs2(QString("Device is connected with highlevel, hl ipaddr is %1").arg(send_data), LOG_SCOPE_IVS);
	}
	else if (base_app_ptr->checkAppDesc(kPODescNetInited))
	{
		//send cmdserver port to IVS client, equal call OnUpdatedNetworkAdapters() function
		//because cmd server port is changed when low-level reboot or offline
		state = kIVSStateNetInited;
		send_data = ip_info_ptr->getCmdPort();
		printlog_lvs2(QString("Device can be connect with IVS, comport is %1").arg(send_data), LOG_SCOPE_IVS);
	}

	if (mode == kIVSDisconnected)
	{
		m_ivs_server.setLogoutConn(conn);
		printlog_lvs2("IVS is disconnected.", LOG_SCOPE_IVS);

		//if IVS was disconnect and app is updating now, cancel update
		if (base_app_ptr->isUpdateNow(conn))
		{
			printlog_lvs2(QString("Update canceled, becasue IVS is disconnected. conn[%1]").arg(conn), LOG_SCOPE_IVS);
			base_app_ptr->deviceUpdateCancel();
		}
	}
	m_ivs_server.updateState(state, send_data);
}

void CBaseAppIVS::onRecivedIVSPacket(i32 conn, Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return;
	}
	
	i32 cmd = packet_ptr->getCmd();
	u8* buffer_ptr = packet_ptr->getData();
	i32 buffer_size = packet_ptr->getDataLen();

	switch (cmd)
	{
		case kIVSCmdOnline:
		{
			printlog_lv1("Read IVS Command: kIVSCmdOnline");
			onRequestIVSOnline(conn);
			break;
		}
		case kIVSCmdOffline:
		{
			printlog_lv1("Read IVS Command: kIVSCmdOffline");
			onRequestIVSOffline(conn);
			break;
		}
		case kIVSCmdLogin:
		{
			printlog_lv1("Read IVS Command: kIVSCmdLogin");
			onRequestIVSLogIn(conn, buffer_ptr, buffer_size);
			break;
		}
		case kIVSCmdLogout:
		{
			printlog_lv1("Read IVS Command: kIVSCmdLogout");
			onRequestIVSLogOut(conn);
			break;
		}
		case kIVSCmdChangePwd:
		{
			printlog_lv1("Read IVS Command: kIVSCmdChangePwd");
			onRequestIVSChangePassword(conn, buffer_ptr, buffer_size);
			break;
		}
		case kIVSCmdImport:
		{
			printlog_lv1("Read IVS Command: kIVSCmdImport");
			onRequestIVSImport(conn, packet_ptr);
			break;
		}
		case kIVSCmdExport:
		{
			printlog_lv1("Read IVS Command: kIVSCmdExport");
			onRequestIVSExport(conn);
			break;
		}
		case kIVSCmdConnect:
		{
			printlog_lv1("Read IVS Command: kIVSCmdConnect");
			onRequestIVSConnect(conn);
			break;
		}
		case kIVSCmdDisconnect:
		{
			printlog_lv1("Read IVS Command: kIVSCmdDisconnect");
			onRequestIVSDisconnect(conn);
			break;
		}
		case kIVSCmdUpdate:
		{
			printlog_lv1("Read IVS Command: kIVSCmdUpdate");
			onRequestIVSUpdate(conn, packet_ptr);
			break;
		}
		case kIVSCmdUpdateConfirm:
		{
			printlog_lv1("Read IVS Command: kIVSCmdUpdateFinish");
			onRequestIVSUpdateFinish(conn);
			break;
		}
		case kIVSCmdUpdateCancel:
		{
			printlog_lv1("Read IVS Command: kIVSCmdUpdateCancel");
			onRequestIVSUpdateCancel(conn);
			break;
		}
	}

	getBaseApp()->checkPacket(packet_ptr, buffer_ptr);
	POSAFE_DELETE(packet_ptr);
}

void CBaseAppIVS::onRequestIVSOnline(i32 conn)
{
	i32 retcode = kIVSReturnOK;
	getBaseApp()->onRequestOnline(NULL);

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdOnline, retcode);
}

void CBaseAppIVS::onRequestIVSOffline(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			retcode = kIVSReturnOK;
			m_ivs_server.setLogoutConn(conn);
			base_app_ptr->onRequestOffline(NULL);
		}
		else
		{
			printlog_lv1(QString("IVS can't be login with Device for offline, conn is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available for offline");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdOffline, retcode);
}

void CBaseAppIVS::onRequestIVSLogIn(i32 conn, u8*& buffer_ptr, i32& buffer_size)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	QThread::msleep(100);

	postring chk_password;
	CPOBase::memRead(buffer_ptr, buffer_size, chk_password);

	if (base_app_ptr->isDeviceAvailable())
	{
		i32 code = m_ivs_server.isLoginConn(conn);
		if (code == kIVSLoginSelf)
		{
			retcode = kIVSReturnOK;
			printlog_lv1("current IVS is already login Device");
		}
		else if (code == kIVSLoginOther)
		{
			retcode = kIVSReturnInvalidLogin;
			printlog_lv1("Device is already other IVS");
		}
		else
		{
			retcode = kIVSReturnAuthErr;
			if (base_app_ptr->checkPassword(chk_password))
			{
				m_ivs_server.setLoginConn(conn);
				retcode = kIVSReturnOK;
			}
			else
			{
				printlog_lv1(QString("IVS can't connect Device with password[%1]").arg(chk_password.c_str()));
			}
		}
	}
	else
	{
		printlog_lv1("Device is not available for login");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdLogin, retcode);
}

void CBaseAppIVS::onRequestIVSLogOut(i32 conn)
{
	m_ivs_server.setLogoutConn(conn);
	m_ivs_server.uploadReturnPacket(conn, kIVSCmdLogout, kIVSReturnOK);
}

void CBaseAppIVS::onRequestIVSChangePassword(i32 conn, u8*& buffer_ptr, i32& buffer_size)
{
	QThread::msleep(100);

	postring old_password;
	postring new_password;
	CPOBase::memRead(buffer_ptr, buffer_size, old_password);
	CPOBase::memRead(buffer_ptr, buffer_size, new_password);

	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnShortPassword;
		if (new_password.size() >= PO_PASSWORD_MINLEN)
		{
			if (base_app_ptr->setPassword(old_password, new_password))
			{
				retcode = kIVSReturnOK;
			}
			else
			{
				retcode = kIVSReturnAuthErr;
			}
		}
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdChangePwd, retcode);
}

void CBaseAppIVS::onRequestIVSImport(i32 conn, Packet* packet_ptr)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			retcode = kIVSReturnInvalidData;
			if (base_app_ptr->deviceImport(packet_ptr))
			{
				retcode = kIVSReturnOK;
			}
		}
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdImport, retcode);
}

void CBaseAppIVS::onRequestIVSExport(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			PacketPVector pak_vec;
			if (base_app_ptr->deviceExport(pak_vec))
			{
				i32 i, count = (i32)pak_vec.size();
				if (CPOBase::isPositive(count))
				{
					for (i = 0; i < count; i++)
					{
						pak_vec[i]->setCmd(kIVSCmdExport);
						pak_vec[i]->setSubCmd(kIVSReturnOK);
						m_ivs_server.sendPacket(conn, pak_vec[i]);
					}
					return;
				}
			}
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdExport, retcode);
}

void CBaseAppIVS::onRequestIVSUpdate(i32 conn, Packet* packet_ptr)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	i32 pid = packet_ptr->getSubCmd();
	i32 buffer_size = packet_ptr->getDataLen();
	u8* buffer_ptr = packet_ptr->getData();
	bool is_last = packet_ptr->isLast();
	
	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			//recive updating data stream
			retcode = base_app_ptr->deviceUpdateStream(conn, pid, buffer_ptr, buffer_size);

			Packet* pak = po_new Packet(kIVSCmdUpdate, kPOPacketRespOK);
			switch (retcode)
			{
				case kPOSuccess:
				{
					pak->setSubCmd(kIVSReturnOK);
					pak->setReservedi32(0, pid);
					break;
				}
				default:
				{
					pak->setSubCmd(kIVSReturnInvalidData);
					break;
				}
			}
			m_ivs_server.sendPacket(conn, pak);

			//automatic update from now...
			if (is_last && retcode == kPOSuccess && base_app_ptr->checkUpdateReady(conn))
			{
				m_ivs_server.updateState(kIVSStateUpdateConfirm, PO_UPDATE_DELAYTIME);
				m_ivs_server.setLogoutConn(conn);
				base_app_ptr->deviceUpdateConfirm(PO_UPDATE_DELAYTIME);
			}
			return;
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdUpdate, retcode);
}

void CBaseAppIVS::onRequestIVSUpdateFinish(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			//check updating is available
			if (!base_app_ptr->isUpdateReady(conn))
			{
				m_ivs_server.uploadReturnPacket(conn, kIVSCmdUpdateConfirm, kIVSReturnFail);
				printlog_lv1("Update Finish is failed");
				return;
			}

			//update device
			if (!base_app_ptr->deviceUpdateInternal())
			{
				base_app_ptr->deviceUpdateCancel();
				m_ivs_server.updateState(kIVSStateUpdateFailed, conn);
				printlog_lv1("DeviceUpdate was canceled, because file operation was failed");
			}
			else
			{
				m_ivs_server.updateState(kIVSStateUpdateCompleted, conn);
			}
			retcode = kIVSReturnOK;
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdUpdateConfirm, retcode);
}

void CBaseAppIVS::onRequestIVSUpdateCancel(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable())
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			if (!base_app_ptr->isUpdateNow(conn))
			{
				m_ivs_server.uploadReturnPacket(conn, kIVSCmdUpdateCancel, kIVSReturnFail);
				printlog_lv1("Update cancel is failed");
				return;
			}

			m_ivs_server.updateState(kIVSStateUpdateCanceled, conn);
			base_app_ptr->deviceUpdateCancel();
			retcode = kIVSReturnOK;
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdUpdateCancel, retcode);
}

void CBaseAppIVS::onRequestIVSConnect(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable() && !base_app_ptr->checkAppDesc(kPODescNetConnected))
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			retcode = kIVSReturnDupConnect;
			if (!base_app_ptr->checkAppDesc(kPODescHighLevel))
			{
				retcode = kIVSReturnOK;
			}
			else
			{
				printlog_lv1("Device is already connected with other HighLevel");
			}
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdConnect, retcode);
}

void CBaseAppIVS::onRequestIVSDisconnect(i32 conn)
{
	i32 retcode = kIVSReturnDevStatusErr;
	CPOBaseApp* base_app_ptr = getBaseApp();

	if (base_app_ptr->isDeviceAvailable() && base_app_ptr->checkAppDesc(kPODescHighLevel))
	{
		retcode = kIVSReturnAuthErr;
		if (m_ivs_server.isLoginConn(conn) == kIVSLoginSelf)
		{
			retcode = kIVSReturnOK;
		}
		else
		{
			printlog_lv1(QString("Device is already login with other IVS, current connection is %1").arg(conn));
		}
	}
	else
	{
		printlog_lv1("Device is not available or is not connect with HighLevel");
	}

	m_ivs_server.uploadReturnPacket(conn, kIVSCmdDisconnect, retcode);
}

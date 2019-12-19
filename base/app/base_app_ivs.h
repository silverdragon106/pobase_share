#pragma once

#include "define.h"
#include "struct.h"
#include "network/ivs_server.h"

class CPOBaseApp;
class CBaseAppIVS
{
public:
	CBaseAppIVS();
	virtual ~CBaseAppIVS();

	CPOBaseApp*					getBaseApp();

	void						onRecivedIVSChange(i32 mode, i32 conn);
	void						onRecivedIVSPacket(i32 conn, Packet* packet_ptr);

	void						onRequestIVSOnline(i32 conn);
	void						onRequestIVSOffline(i32 conn);
	void						onRequestIVSLogIn(i32 conn, u8*& buffer_ptr, i32& buffer_size);
	void						onRequestIVSLogOut(i32 conn);
	void						onRequestIVSChangePassword(i32 conn, u8*& buffer_ptr, i32& buffer_size);
	void						onRequestIVSImport(i32 conn, Packet* packet_ptr);
	void						onRequestIVSExport(i32 conn);
	void						onRequestIVSUpdate(i32 conn, Packet* packet_ptr);
	void						onRequestIVSUpdateFinish(i32 conn);
	void						onRequestIVSUpdateCancel(i32 conn);
	void						onRequestIVSConnect(i32 conn);
	void						onRequestIVSDisconnect(i32 conn);

public:
	CIVSServer					m_ivs_server;
};

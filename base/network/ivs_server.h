#pragma once

#include "define.h"
#include "server.h"
#include <QObject>

enum IVSState
{
	kIVSStateNone = 0,
	kIVSStateNetInited,
	kIVSStateNetFree,
	kIVSStateBusy,
	kIVSStateLogin,
	kIVSStateLogOut,
	kIVSStateUpdateConfirm,		//POCodeFix
	kIVSStateReconnect,
	kIVSStateUpdateCompleted,	//POCodeFix
	kIVSStateUpdateFailed,		//POCodeFix
	kIVSStateUpdateCanceled		//POCodeFix
};

enum IVSReturnType
{
	kIVSReturnUnknown = 0,

	kIVSReturnOK = 10,
	kIVSReturnFail,

	kIVSReturnDevStatusErr = 20,
	kIVSReturnAuthErr,
	kIVSReturnRootErr,
	kIVSReturnInvalidLogin,
	kIVSReturnInvalidUser,
	kIVSReturnInvalidData,
	kIVSReturnInvalidConnect,
	kIVSReturnInvalidUpdate,
	kIVSReturnShortPassword,
	kIVSReturnDupConnect
};

enum IVSConnType
{
	kIVSConnected = 0,
	kIVSDisconnected
};

enum IVSLoginType
{
	kIVSLoginNone = 0,
	kIVSLoginSelf,
	kIVSLoginOther
};

class CIVSServer : public CServer
{
	Q_OBJECT

public:
	CIVSServer();
	virtual ~CIVSServer();

	void					onReadPacket(Packet* packet_ptr, CConnection* connection_ptr);	// ready to read packet_ptr.
	void					onNewConnection(CConnection* connection_ptr);					// new connection was established.
	void					onLostConnection(CConnection* connection_ptr);					// old connection was closed.
	void					onCreatedServer();

	i32						isLoginConn(i32 conn);
	bool					setLoginConn(i32 conn);
	bool					setLogoutConn(i32 conn);
	void					setLogoutAll();
	void					setHLEmbedded(bool bHLEmbedded, bool bCanUpdate);

	void					uploadReturnPacket(i32 conn, i32 cmd, i32 code);
	void					updateState(i32 state, i64 data = 0);

signals:
	void					serverInited(i32 mode, i32 port);
	void					signalIVSChange(i32 mode, i32 conn);
	void					signalIVSReadPacket(i32 mode, Packet* packet_ptr);

public:
	bool					m_can_update;
	bool					m_is_hl_embedded;
	i32						m_login_conn;
	QMutex					m_login_mutex;
};

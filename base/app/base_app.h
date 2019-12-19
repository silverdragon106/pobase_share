#pragma once

#include "define.h"
#include "io/usb_dongle.h"
#include "base_app_ivs.h"
#include "base_app_update.h"
#include "network/cmd_server.h"
#include "network/ivs_server.h"
#include "network/udp_broadcast.h"
#include "streaming/ipc_manager.h"

#if defined(POR_WINDOWS)
  #include <QApplication>
  #define QApp  QApplication
#elif defined(POR_LINUX)
  #include <QCoreApplication>
  #define QApp	QCoreApplication
#endif

#include <QMutex>
#include <QTimer>

struct Packet;
class CIPCManager;
class CEmuSamples;

Q_DECLARE_METATYPE(u8); //u8 can be set into QVariant
Q_DECLARE_METATYPE(u16);
Q_DECLARE_METATYPE(u32);
Q_DECLARE_METATYPE(u64);
Q_DECLARE_METATYPE(i8);
Q_DECLARE_METATYPE(i16);
Q_DECLARE_METATYPE(i32);
Q_DECLARE_METATYPE(i64);
Q_DECLARE_METATYPE(f32);
Q_DECLARE_METATYPE(f64);
Q_DECLARE_METATYPE(Packet*);
Q_DECLARE_METATYPE(postring);
Q_DECLARE_METATYPE(powstring);

class CPOBaseApp : public QApp, public CBaseAppIVS, public CBaseAppUpdate
{
	Q_OBJECT

public:
	CPOBaseApp(i32 argc, char *argv[]);
	virtual ~CPOBaseApp();
	
	virtual	bool				initApplication(i32 desc, i32 argc, char *argv[]);
	virtual bool				exitApplication();
	virtual void				initEvent();
	virtual bool				initInstance();
	virtual bool				exitInstance();

    virtual void                registerDataTypes();

	void						preInitEvent();
	void						addAppDesc(i32 desc);
	void						removeAppDesc(i32 desc);
	bool						checkAppDesc(i32 desc);
	bool						checkNeedDesc(i32 desc);
	void						updateTimerStart(i32 delay_confirm_ms);
	void						updateTimerStop();
	void						updateNetworkAdapters();
	void						setAuthIdPassword(postring& str_auth_id, postring& str_auth_password);
	bool						getDataPath(const QString& strshmem, i64 hl_uid);
	QString						getDevicePath();
	QString						getDeviceHLPath();
	CEmuSamples*				getEmuSamples();
	void						uploadEmulatorThumb();

	void						sendPacketToNet(Packet* packet_ptr);
	void						sendPacketToNet(Packet* packet_ptr, u8* buffer_ptr);
	void						sendPacketToNet(PacketQueueVector& packet_vec);
	void						sendPacketToIPC(Packet* packet_ptr, u8* buffer_ptr);
	bool						checkPacket(Packet* packet_ptr, u8* buffer_ptr);

	void						uploadEndPacket(Packet* packet_ptr);
	void						uploadFailPacket(Packet* packet_ptr, i32 code = kPOErrCmdFail);
	void						uploadEndPacket(i32 cmd, i32 code = kPOSuccess, i32 subcmd = kPOSubTypeNone);
	void						uploadFailPacket(i32 cmd, i32 code = kPOErrCmdFail, i32 subcmd = kPOSubTypeNone);
	void						uploadEndPacket2(i32 conn, i32 cmd, i32 code = kPOSuccess, i32 sub_cmd = kPOSubTypeNone);
	void						uploadFailPacket2(i32 conn, i32 cmd, i32 code = kPOErrCmdFail, i32 sub_cmd = kPOSubTypeNone);

	virtual i32					onRequestSync(Packet* packet_ptr);
	virtual i32					onRequestUnSync(Packet* packet_ptr);
	virtual i32					onRequestOnline(Packet* packet_ptr);
	virtual i32					onRequestOffline(Packet* packet_ptr);
	virtual	i32					onRequestStop(Packet* packet_ptr);
	virtual i32					onRequestDevicePowerOff(Packet* packet_ptr, i32 mode);
	virtual i32					onRequestEmulator(Packet* packet_ptr);

	virtual bool				isSingleInstance(const char* uuid_str);
	virtual bool				getHeartBeat(HeartBeat& hb);

	virtual QString				getLowLevelName() = 0;
	virtual QString				getHighLevelName() = 0;
    virtual DeviceInfo*			getDeviceInfo() = 0;
    virtual CIPInfo*			getIPInfo() = 0;

	virtual bool				checkPassword(postring& chk_password) = 0;
	virtual bool				setPassword(postring& cur_password, postring& new_password) = 0;
	virtual bool				deviceImport(Packet* packet_ptr) = 0;
	virtual bool				deviceExport(PacketPVector& pak_vec) = 0;
	virtual bool				appendToFile(const char* filename, u8* buffer_ptr, i32 len) = 0;
	virtual bool				updateDeviceINISettings(QString& str_ll_path) = 0;
	virtual bool				updateDeviceOffline() = 0;
	virtual bool				executeSQLQuery(QString& strquery) = 0;
	
	virtual i32					onRequestConnect(Packet* packet_ptr, i32 conn) = 0;
	virtual i32					onRequestPreConnect(Packet* packet_ptr, i32 conn) = 0;
	virtual void				onReadCmdPacket(Packet* packet_ptr, i32 conn_mode) = 0;
	
public:
	inline i32					getAppDesc() { return m_app_desc; };
	inline i32					getInstallDesc() { return m_install_desc; };
	inline bool					isDeviceOffline() { return !m_is_inited; };
	inline bool					isDeviceAvailable() { return m_is_inited; };
	
public slots:
	virtual void				onOnlineToggleControl();
	virtual void				onPlugOutDongle();

	void						onTimerUpdate();

	void						_onNewConnection(i32 conn, i32 conn_mode);
	void						_onLostConnection();
	void						_onInitedNetworkAdapters(i32 mode, i32 port);

 	void						_onRecivedIVSChange(i32 mode, i32 conn);
 	void						_onRecivedIVSPacket(i32 conn, Packet* packet_ptr);

	void						_onRequestPreConnect(Packet* packet_ptr, i32 conn);
	void						_onRequestConnect(Packet* packet_ptr, i32 conn);
	void						_onReadCmdPacket(Packet* packet_ptr, i32 conn_mode);

public:
	CUSBDongle					m_usb_dongle;
	CUdpBroadCast				m_udp_broadcast;
	CCmdServer					m_cmd_server;

	CEmuSamples*				m_emu_sample_ptr;
	CIPCManager*				m_ipc_manager_ptr;

	std::atomic<i32>			m_app_desc;
	std::atomic<i32>			m_install_desc;

	bool						m_is_inited;
	bool						m_use_ext_path;
	QString						m_data_path;
	QString						m_data_path_hl;
	i32							m_action_auth_fail;

	QTimer						m_update_timer;
};

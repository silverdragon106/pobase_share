#include "base_app.h"
#include "base.h"
#include "base_disk.h"
#include "proc/emulator/emu_samples.h"
#include "network/packet.h"
#include "streaming/ipc_manager.h"
#include "os/os_support.h"
#include "log_config.h"

QSharedMemory	g_shmem;

//////////////////////////////////////////////////////////////////////////
CPOBaseApp::CPOBaseApp(i32 argc, char *argv[]) 
    : QApp(argc, argv)
{
	m_ipc_manager_ptr = NULL;
	m_emu_sample_ptr = NULL;
	
	m_app_desc = 0;
	m_install_desc = 0;

	m_is_inited = false;
	m_use_ext_path = false;
	m_data_path = "";
	m_data_path_hl = "";
	m_action_auth_fail = kPOTerminateApp;
}

CPOBaseApp::~CPOBaseApp()
{
}

bool CPOBaseApp::initApplication(i32 desc, i32 argc, char *argv[])
{
	m_install_desc = desc;
	CPOBaseApp::preInitEvent();

	//update device information
	updateNetworkAdapters();

	switch (argc)
	{
		case 4:
		{
			if (checkNeedDesc(kPODescIPCStream))
			{
				//provide test data_path and device ID for emulator, when started by IVS
				m_use_ext_path = true;
				getDataPath(QString(argv[1]), QString(argv[2]).toLongLong());
				getDeviceInfo()->device_id = QString(argv[3]).toInt();
			}
			break;
		}
		default:
		{
			//else case, provide none
			m_data_path = qApp->applicationDirPath();
			break;
		}
	}
	
	if (checkNeedDesc(kPODescIVSServer))
	{
		addAppDesc(kPODescIVSServer);
		m_ivs_server.netStart();
	}

	if (checkNeedDesc(kPODescEmulator))
	{
#if defined(POR_EMULATOR)
		addAppDesc(kPODescEmulator);
#if defined(POR_EMULATOR_LOCAL)
		m_emu_sample_ptr = po_new CLocalEmuSamples();
		QString sample_path = m_data_path + PO_SAMPLE_SUBPATH;
		((CLocalEmuSamples*)m_emu_sample_ptr)->loadSamples(sample_path);
#else
		m_emu_sample_ptr = po_new COneEmuSamples(PO_CAM_WIDTH, PO_CAM_HEIGHT);
#endif
#endif
	}

	deviceUpdateCancel();

	//GUID : Generated once for your application
	//you could get one GUID here: http://www.guidgenerator.com/online-guid-generator.aspx
	if (checkNeedDesc(kPODescSInstance))
	{
		if (!isSingleInstance(PO_PROJECT_GUID))
		{
			printlog_lv0("Device single instance check is failed...");
			alarmlog1(kPOErrSingleInstance);
			return false;
		}
		addAppDesc(kPODescSInstance);
	}
	return true;
}

bool CPOBaseApp::exitApplication()
{
	singlelog_lv0("POBaseApp Quitting Now...");

	//exit IVSServer
	if (checkAppDesc(kPODescIVSServer))
	{
		m_ivs_server.netClose();
		removeAppDesc(kPODescIVSServer);
	}
	
	//exit USBDongle processor
	if (checkAppDesc(kPODescUsbDongle))
	{
		m_usb_dongle.exitInstance();
		removeAppDesc(kPODescUsbDongle);
	}

	if (checkAppDesc(kPODescIPCStream))
	{
		POSAFE_DELETE(m_ipc_manager_ptr);
		removeAppDesc(kPODescIPCStream);
	}

	if (checkAppDesc(kPODescEmulator))
	{
		POSAFE_DELETE(m_emu_sample_ptr);
	}
    QApp::quit();
	return true;
}

void CPOBaseApp::registerDataTypes()
{
    qRegisterMetaType<u8>("u8");
    qRegisterMetaType<u16>("u16");
    qRegisterMetaType<u32>("u32");
    qRegisterMetaType<i8>("i8");
    qRegisterMetaType<i16>("i16");
    qRegisterMetaType<i32>("i32");
    qRegisterMetaType<f32>("f32");
    qRegisterMetaType<f64>("f64");
    qRegisterMetaType<postring>("postring");
    qRegisterMetaType<powstring>("powstring");
    qRegisterMetaType<Packet*>("ptrPacket");
}

void CPOBaseApp::initEvent()
{
}

void CPOBaseApp::preInitEvent()
{
	CIVSServer* ivs_server_ptr = &m_ivs_server;
	CCmdServer* cmd_server_ptr = &m_cmd_server;

	if (checkNeedDesc(kPODescCmdServer))
	{
		//connect signal CmdServer -> BasedApp
		connect(cmd_server_ptr, SIGNAL(networkPreConnect(Packet*, i32)), this, SLOT(_onRequestPreConnect(Packet*, i32)));
		connect(cmd_server_ptr, SIGNAL(networkConnect(Packet*, i32)), this, SLOT(_onRequestConnect(Packet*, i32)));
		connect(cmd_server_ptr, SIGNAL(networkReadPacket(Packet*, i32)), this, SLOT(_onReadCmdPacket(Packet*, i32)));
		connect(cmd_server_ptr, SIGNAL(networkLostConnection()), this, SLOT(_onLostConnection()));
		connect(cmd_server_ptr, SIGNAL(serverInited(i32, i32)), this, SLOT(_onInitedNetworkAdapters(i32, i32)));
	}
	
	if (checkNeedDesc(kPODescIVSServer))
	{
		//connect signal IVSServer -> BasedApp
		connect(ivs_server_ptr, SIGNAL(signalIVSChange(i32, i32)), this, SLOT(_onRecivedIVSChange(i32, i32)), Qt::QueuedConnection);
		connect(ivs_server_ptr, SIGNAL(signalIVSReadPacket(i32, Packet*)), this, SLOT(_onRecivedIVSPacket(i32, Packet*)), Qt::QueuedConnection);
		connect(ivs_server_ptr, SIGNAL(serverInited(i32, i32)), this, SLOT(_onInitedNetworkAdapters(i32, i32)));
	}

	if (checkNeedDesc(kPODescUsbDongle))
	{
		//connect plug out signal from USB dongle to main app
		connect(&m_usb_dongle, SIGNAL(licenseError()), this, SLOT(onPlugOutDongle()), Qt::QueuedConnection);
	}

	connect(&m_update_timer, SIGNAL(timeout()), this, SLOT(onTimerUpdate()));
	m_update_timer.setSingleShot(true);
	m_update_timer.stop();
}

bool CPOBaseApp::initInstance()
{
	//check usb dongle
	if (checkNeedDesc(kPODescUsbDongle))
	{
		singlelog_lv0("The USBDongle InitInstance");
		if (!m_usb_dongle.initInstance())
		{
			onPlugOutDongle();
			return false;
		}
		addAppDesc(kPODescUsbDongle);
	}
	
	//init CmdServer
	if (checkNeedDesc(kPODescCmdServer))
	{
		singlelog_lv0("The Network Module InitInstance");
		
		m_cmd_server.netStart();
		m_ivs_server.setHLEmbedded(getDeviceInfo()->is_hl_embedded, !(getDeviceInfo()->is_hl_embedded));
		addAppDesc(kPODescCmdServer);
	}

	m_is_inited = true;
	return true;
}

bool CPOBaseApp::exitInstance()
{
	//exit UDPBroadCast
	if (checkAppDesc(kPODescHeartBeat))
	{
		m_udp_broadcast.exitInstance();
		removeAppDesc(kPODescHeartBeat);
	}

	//exit CmdServer
	if (checkAppDesc(kPODescCmdServer))
	{
		singlelog_lv0("The Network Module ExitInstance");
		m_cmd_server.netClose();
		removeAppDesc(kPODescCmdServer);
	}

	m_is_inited = false;
	return true;
}

void CPOBaseApp::onOnlineToggleControl()
{
	if (isDeviceOffline())
	{
		//switch to online mode
		onRequestOnline(NULL);
	}
	else
	{
		//switch to offline mode
		onRequestOffline(NULL);
	}
}

void CPOBaseApp::onPlugOutDongle()
{
	printlog_lv0("The USBDongle is not connected...");
	alarmlog1(kPOErrDongleCheck);
	onRequestDevicePowerOff(NULL, m_action_auth_fail);
}

bool CPOBaseApp::checkNeedDesc(i32 desc)
{
	return CPOBase::bitCheck(m_install_desc, desc);
}

bool CPOBaseApp::checkAppDesc(i32 desc)
{
	return CPOBase::bitCheck(m_app_desc, desc);
}

void CPOBaseApp::addAppDesc(i32 desc)
{
	m_app_desc |= desc;
}

void CPOBaseApp::removeAppDesc(i32 desc)
{
	m_app_desc &= ~desc;
}

i32 CPOBaseApp::onRequestSync(Packet* packet_ptr)
{
	return kPOSuccess;
}

i32 CPOBaseApp::onRequestUnSync(Packet* packet_ptr)
{
	return kPOSuccess;
}

i32 CPOBaseApp::onRequestOnline(Packet* packet_ptr)
{
	return kPOSuccess;
}

i32 CPOBaseApp::onRequestOffline(Packet* packet_ptr)
{
	return kPOSuccess;
}

i32 CPOBaseApp::onRequestStop(Packet* packet_ptr)
{
	return kPOSuccess;
}

bool CPOBaseApp::isSingleInstance(const char* uuid_str)
{
#ifdef POR_LINUX
	//on linux/unix shared memory is not freed upon crash
	//so if there is any trash from previous instance, clean it
	QSharedMemory shmem_fix_crash(uuid_str);
	if (shmem_fix_crash.attach())
	{
		shmem_fix_crash.detach();
	}
#endif

	g_shmem.setKey(uuid_str);
	return (!g_shmem.attach() && g_shmem.create(64, QSharedMemory::ReadWrite));
}

i32 CPOBaseApp::onRequestDevicePowerOff(Packet* packet_ptr, i32 mode)
{
	switch (mode)
	{
		case kPOPowerSleep:
		{
			printlog_lv0("The device will be sleep now");
			actionlog(POAction(kPOActionDevSleep));
			break;
		}
		case kPOPowerReboot:
		{
			printlog_lv0("The device will be reboot now");
			actionlog(POAction(kPOActionDevReboot));
			break;
		}
		case kPOPowerShutdown:
		{
			printlog_lv0("The device will be shutdown now");
			actionlog(POAction(kPOActionDevPowerOff));
			break;
		}
		case kPOTerminateApp:
		{
			printlog_lv0("The app will be terminate now");
			actionlog(POAction(kPOActionDevUnplugDongle));
			break;
		}
		default:
		{
			return kPOErrInvalidOper;
		}
	}

	onRequestOffline(NULL);
	exitApplication();
    QApp::quit();

	//poweroff, reboot
	bool is_success = kPOSuccess;
	switch (mode)
	{
		case kPOPowerSleep:
		{
			is_success = COSBase::sleepSystem();
			break;
		}
		case kPOPowerReboot:
		{
			is_success = COSBase::rebootSystem();
			break;
		}
		case kPOPowerShutdown:
		{
			is_success = COSBase::poweroffSystem();
			break;
		}
		case kPOTerminateApp:
		{
			break;
		}
	}

	if (!is_success)
	{
		return kPOErrInvalidOper;
	}
	return kPOSuccess;
}

i32 CPOBaseApp::onRequestEmulator(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return kPOErrInvalidPacket;
	}
	if (!checkAppDesc(kPODescEmulator))
	{
		return kPOErrInvalidOper;
	}

#if defined(POR_EMULATOR)
	i32 sub_type = packet_ptr->getSubCmd();
	if (!m_emu_sample_ptr)
	{
		return kPOErrInvalidOper;
	}

	switch (sub_type)
	{
		// 모의기에 화상자료들을 설정하고 보관한다.(LocalEmu방식에서 망통신으로 자료전송할때)
		case kPOSubTypeEmuSetImage:
		{
			bool is_thumb_generation = packet_ptr->getReservedb8(0);
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr || !emu_sample_ptr->setSamples(packet_ptr, is_thumb_generation))
			{
				return kPOErrInvalidData;
			}

			QString sample_path = m_data_path + PO_SAMPLE_SUBPATH;
			if (!emu_sample_ptr->writeSamples(sample_path))
			{
				return kPOErrDiskCantWrite;
			}

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}

			if (is_thumb_generation)
			{
				uploadEmulatorThumb();
			}
			break;
		}

		// 모의기에 화상자료가 존재하는 등록부를 설정한다.(LocalEmu방식에서 로컬등록부를 지정할때)
		case kPOSubTypeEmuSetPath:
		{
			potstring data_path;
			u8* buffer_ptr = packet_ptr->getData();
			bool is_thumb_generation = packet_ptr->getReservedb8(0);
			if (!buffer_ptr)
			{
				return kPOErrInvalidPacket;
			}

			CPOBase::memRead(buffer_ptr, data_path);
			QString data_path_qstr = QString::fromTCharArray(data_path.c_str());
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr || !emu_sample_ptr->loadSamples(data_path_qstr, is_thumb_generation))
			{
				return kPOErrInvalidData;
			}

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}

			if (is_thumb_generation)
			{
				uploadEmulatorThumb();
			}
			break;
		}

		// 모의기의 화상자료순환을 시작한다. (LocalEmu방식)
		case kPOSubTypeEmuPlay:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}
			emu_sample_ptr->play();

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		// 모의기에서 화상자료순환을 중지한다. (LocalEmu방식)
		case kPOSubTypeEmuStop:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}
			emu_sample_ptr->stop();

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		// 모의기에서 화상자료순환주기를 설정한다. (LocalEmu방식)
		case kPOSubTypeEmuInterval:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}

			i32 interval = packet_ptr->getReservedi32(0);
			emu_sample_ptr->setEmuTriggerInterval(interval);

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		// 모의기에서 지정한 화상자료를 선택한다. (LocalEmu방식)
		case kPOSubTypeEmuSelected:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}

			i32 cam_index = packet_ptr->getReservedi32(0);
			i32 selected_index = packet_ptr->getReservedi32(1);
			emu_sample_ptr->setEmuSelected(cam_index, selected_index, false);

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		// 모의기에서 지정한 화상자료를 선택하고 순환명령을 중지시킨다. (LocalEmu방식)
		case kPOSubTypeEmuSelectStop:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}

			i32 cam_index = packet_ptr->getReservedi32(0);
			i32 selected_index = packet_ptr->getReservedi32(1);
			emu_sample_ptr->setEmuSelected(cam_index, selected_index, true);

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		// 모의기에 현재 적재된 모의기화상을 얻어낸다. (LocalEmu방식)
		case kPOSubTypeEmuThumb:
		{
			CLocalEmuSamples* emu_sample_ptr = dynamic_cast<CLocalEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr)
			{
				return kPOErrInvalidOper;
			}

			uploadEmulatorThumb();
			break;
		}

		// 모의기에 한개의 화상을 전송한다. (OneEmu방식)
		case kPOSubTypeEmuDownSample:
		{
			ImageData img_data;
			i32 w = packet_ptr->getReservedi32(0);
			i32 h = packet_ptr->getReservedi32(1);
			i32 channel = packet_ptr->getReservedi32(2);
			img_data.setImageData(packet_ptr->getData(), w, h, channel);
			if (!img_data.isValid())
			{
				return kPOErrInvalidPacket;
			}
						
			COneEmuSamples* emu_sample_ptr = dynamic_cast<COneEmuSamples*>(m_emu_sample_ptr);
			if (!emu_sample_ptr || !emu_sample_ptr->setEmuSample(img_data))
			{
				return kPOErrInvalidOper;
			}

			Packet* pak = Packet::makeRespPacket(packet_ptr);
			if (pak)
			{
				sendPacketToNet(pak);
			}
			break;
		}

		default:
		{
			return kPOErrInvalidOper;
			break;
		}
	}
#endif
	return kPOSuccess;
}

void CPOBaseApp::uploadEmulatorThumb()
{
#if defined(POR_EMULATOR)
	if (!m_emu_sample_ptr)
	{
		return;
	}

	i32 len = m_emu_sample_ptr->memSizeThumb();
	Packet* pak = po_new Packet(kPOCmdEmulator, kPOPacketRespOK);
	u8* buffer_ptr = NULL;
	i32 buffer_size = len;

	pak->setSubCmd(kPOSubTypeEmuThumb);
	pak->allocateBuffer(len, buffer_ptr);
	m_emu_sample_ptr->memWriteThumb(buffer_ptr, buffer_size);

	sendPacketToNet(pak, buffer_ptr);
#endif
}

void CPOBaseApp::_onNewConnection(i32 conn, i32 conn_mode)
{
	if (checkAppDesc(kPODescCmdServer))
	{
		onRequestUnSync(NULL);
		m_cmd_server.onNewHLConnection(conn, conn_mode);
		addAppDesc(kPODescNetConnected);
		printlog_lv0("Network Connection was established.");
	}
}

void CPOBaseApp::_onLostConnection()
{
	if (checkAppDesc(kPODescCmdServer))
	{
		onRequestUnSync(NULL);
		removeAppDesc(kPODescNetConnected);
		printlog_lv0("Network Connection was disconnected.");
	}
}

void CPOBaseApp::_onInitedNetworkAdapters(i32 net_mode, i32 port)
{
	CIPInfo* ip_info_ptr = getIPInfo();
	if (!ip_info_ptr)
	{
		return;
	}

	anlock_guard_ptr(ip_info_ptr);
	switch (net_mode)
	{
		case kPOServerCmd:
		{
			ip_info_ptr->m_cmd_port = port;
			ip_info_ptr->m_server |= net_mode;
			break;
		}
		case kPOServerIVS:
		{
			ip_info_ptr->m_ivs_port = port;
			ip_info_ptr->m_server |= net_mode;
			break;
		}
	}

	if (ip_info_ptr->m_server != kPOServerAll)
	{
		return;
	}

	//store network adapter list of low-level side network interface, it will be used to match establish connect IP
	i32 cmd_port = ip_info_ptr->m_cmd_port;
	i32 ivs_port = ip_info_ptr->m_ivs_port;
	printlog_lv1(QString("OnInitNetworkAdapter is accepted. cmd:%1, ivs:%2").arg(cmd_port).arg(ivs_port));

	m_ivs_server.updateState(kIVSStateNetInited, cmd_port);
	addAppDesc(kPODescNetInited);

	if (checkNeedDesc(kPODescHeartBeat))
	{
		DeviceInfo* dev_info_ptr = getDeviceInfo();
		m_udp_broadcast.initInstance(this, dev_info_ptr->getCommPort());
		addAppDesc(kPODescHeartBeat);

		if (checkAppDesc(kPODescIPCStream))
		{
			HeartBeat hb;
			getHeartBeat(hb);

			u8* buffer_ptr;
			i32 buffer_size = hb.memSize();
			Packet* packet_ptr = po_new Packet(kPOCmdHeartBeat, kPOPacketRespOK);
			packet_ptr->allocateBuffer(buffer_size, buffer_ptr);
			hb.memWrite(buffer_ptr, buffer_size);
			m_ipc_manager_ptr->send(PacketRef(packet_ptr));
			printlog_lv1("The first IPC Heartbeat packet is sent.");
		}
	}
}

void CPOBaseApp::_onRecivedIVSChange(i32 mode, i32 conn)
{
	CBaseAppIVS::onRecivedIVSChange(mode, conn);
}

void CPOBaseApp::_onRecivedIVSPacket(i32 conn, Packet* packet_ptr)
{
	CBaseAppIVS::onRecivedIVSPacket(conn, packet_ptr);
}

void CPOBaseApp::_onRequestPreConnect(Packet* packet_ptr, i32 conn)
{
	i32 code = onRequestPreConnect(packet_ptr, conn);
	if (code != kPOSuccess)
	{
		uploadFailPacket2(conn, kPOCmdPreConnect, code);
	}
	POSAFE_DELETE(packet_ptr); //must be delete
}

void CPOBaseApp::_onRequestConnect(Packet* packet_ptr, i32 conn)
{
	i32 code = onRequestConnect(packet_ptr, conn);
	if (code == kPOSuccess)
	{
		i32 conn_mode = packet_ptr->getReservedu8(0);
		if (CPOBase::checkIndex(conn_mode, kPOConnNone, kPOConnCount))
		{
			_onNewConnection(conn, conn_mode);
			uploadEndPacket2(conn, kPOCmdConnect);
		}
		else
		{
			uploadFailPacket2(conn, kPOCmdConnect, kPOErrInvalidConnect);
		}
	}
	else
	{
		uploadFailPacket2(conn, kPOCmdConnect, code);
	}
	POSAFE_DELETE(packet_ptr); //must be delete
}

void CPOBaseApp::_onReadCmdPacket(Packet* packet_ptr, i32 conn_mode)
{
	onReadCmdPacket(packet_ptr, conn_mode);
}

CEmuSamples* CPOBaseApp::getEmuSamples()
{
	return m_emu_sample_ptr;
}

QString CPOBaseApp::getDevicePath()
{
	return m_data_path;
}

QString CPOBaseApp::getDeviceHLPath()
{
	return m_data_path_hl;
}

bool CPOBaseApp::getDataPath(const QString& strshmem, i64 hl_uid)
{
#ifdef POR_LINUX
	//on linux/unix shared memory is not freed upon crash
	//so if there is any trash from previous instance, clean it
	QSharedMemory shmem_fix_crash(strshmem);
	if (shmem_fix_crash.attach())
	{
		shmem_fix_crash.detach();
	}
#endif

	QSharedMemory shared_mem;
	potstring str_data_path;

	shared_mem.setKey(strshmem);
	if (shared_mem.attach(QSharedMemory::ReadOnly))
	{
		shared_mem.lock();
		Packet* packet_ptr;
		IPCHeader* header_ptr = (IPCHeader*)shared_mem.data();
		
		i32 i, index;
		i32 buffer_size = 0;
		i32 count = header_ptr->packet_count;
		u8* buffer_ptr = (u8*)(header_ptr + 1);

		for (i = 0; i < count; i++)
		{
			//check hl_uid
			index = (header_ptr->read_index + i) % IPC_QUEUE_MAXSIZE;
			if (header_ptr->packet_uid[index] != hl_uid)
			{
				continue;
			}

			//check command type
			packet_ptr = (Packet*)(buffer_ptr + header_ptr->packet_pos[index]);
			if (packet_ptr->getCmd() != IPC_SendArugments)
			{
				continue;
			}

			//read datapath from buffer
			buffer_ptr = (u8*)packet_ptr + Packet::calcHeaderSize();
			buffer_size = packet_ptr->getDataLen();

			if (!CPOBase::memRead(buffer_ptr, buffer_size, str_data_path))
			{
				return false;
			}
			m_data_path = QString::fromTCharArray(str_data_path.c_str());

			printlog_lv1(QString("recived data path, hl_uid is %1 ").arg(hl_uid) + m_data_path);
			shared_mem.unlock();
			return true;
		}

		//error case
		shared_mem.unlock();
		printlog_lv1(QString("external arugment data is corrupted, hl_uid is %1 ").arg(hl_uid));
	}
	return false;
}

bool CPOBaseApp::getHeartBeat(HeartBeat& hb)
{	
	CIPInfo* ip_info_ptr = getIPInfo();
	DeviceInfo* dev_info_ptr = getDeviceInfo();
	if (!ip_info_ptr || !dev_info_ptr)
	{
		return false;
	}

	{
		anlock_guard_ptr(ip_info_ptr);
		hb.cmd_port = ip_info_ptr->m_cmd_port;
		hb.ivs_port = ip_info_ptr->m_ivs_port;
		hb.netadapter_vec = ip_info_ptr->m_netadapter_vec;
	}
	hb.state = kHBStateIdle;

	if (checkAppDesc(kPODescNetConnected))
	{
		hb.state = kHBStateWork;
	}
	else if (isDeviceOffline())
	{
		hb.state = kHBStateOFF;
	}

	hb.dev_id = dev_info_ptr->device_id;
	hb.device_name = dev_info_ptr->device_name;
	hb.device_version = dev_info_ptr->device_version;
	hb.device_model = dev_info_ptr->model_name;
	hb.is_emulator = dev_info_ptr->is_emulator;
	hb.connections = m_cmd_server.getConnection();
	return true;
}

void CPOBaseApp::uploadEndPacket(i32 cmd, i32 code, i32 sub_cmd)
{
	Packet* pak = po_new Packet(cmd, kPOPacketRespOK);
	pak->setSubCmd(sub_cmd);
	pak->setReturnCode(code);
	sendPacketToNet(pak);
}

void CPOBaseApp::uploadEndPacket2(i32 conn, i32 cmd, i32 code, i32 sub_cmd)
{
	Packet* pak = po_new Packet(cmd, kPOPacketRespOK);
	pak->setSubCmd(sub_cmd);
	pak->setReturnCode(code);
	m_cmd_server.sendPacket(conn, pak);
}

void CPOBaseApp::uploadEndPacket(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return;
	}

	Packet* pak = Packet::makeRespPacket(packet_ptr);
	if (pak)
	{
		pak->setReturnCode(kPOSuccess);
		sendPacketToNet(pak);
	}
}

void CPOBaseApp::uploadFailPacket(i32 cmd, i32 code, i32 sub_cmd)
{
	Packet* pak = po_new Packet(cmd, kPOPacketRespFail);
	pak->setSubCmd(sub_cmd);
	pak->setReturnCode(code);
	sendPacketToNet(pak);
}

void CPOBaseApp::uploadFailPacket2(i32 conn, i32 cmd, i32 code, i32 sub_cmd)
{
	Packet* pak = po_new Packet(cmd, kPOPacketRespFail);
	pak->setSubCmd(sub_cmd);
	pak->setReturnCode(code);
	m_cmd_server.sendPacket(conn, pak);
}

void CPOBaseApp::uploadFailPacket(Packet* packet_ptr, i32 code)
{
	if (!packet_ptr)
	{
		return;
	}

	Packet* pak = Packet::makeRespPacket(packet_ptr);
	if (pak)
	{
		pak->setHeaderType(kPOPacketRespFail);
		pak->setReturnCode(code);
		sendPacketToNet(pak);
	}
}

void CPOBaseApp::sendPacketToNet(Packet* pak)
{
	if (!pak)
	{
		return;
	}
	m_cmd_server.sendPacket(m_cmd_server.getConnection(), pak);
}

void CPOBaseApp::sendPacketToNet(Packet* pak, u8* buffer_ptr)
{
	checkPacket(pak, buffer_ptr);
	m_cmd_server.sendPacket(m_cmd_server.getConnection(), pak);
}

void CPOBaseApp::sendPacketToNet(PacketQueueVector& packet_vec)
{
	i32 i, count = (i32)packet_vec.size();
	i32 conn = m_cmd_server.getConnection();
	if (count <= 0)
	{
		return;
	}

	for (i = 0; i < count; i++)
	{
		packet_vec[i].conn = conn;
	}
	m_cmd_server.sendPacket(packet_vec);
}

bool CPOBaseApp::checkPacket(Packet* packet_ptr, u8* buffer_ptr)
{
	if (!packet_ptr)
	{
		return false;
	}

	u8* data_ptr = packet_ptr->getData();
	if (!data_ptr)
	{
		return true;
	}

	i32 data_diff = (i32)(buffer_ptr - data_ptr) - packet_ptr->getDataLen();
	if (data_diff != 0)
	{
		i32 cmd = packet_ptr->getCmd();
		printlog_lv0(QString("[UNCORRECT ACCESS] cmd:%1 diff:%2(-:remain, +:over)").arg(cmd).arg(data_diff));
		assert(false);
		return false;
	}
	return true;
}

void CPOBaseApp::sendPacketToIPC(Packet* pak, u8* buffer_ptr)
{
	if (!m_ipc_manager_ptr || !pak || !m_ipc_manager_ptr->isInited())
	{
		return;
	}

	checkPacket(pak, buffer_ptr);
	m_ipc_manager_ptr->send(PacketRef(pak));
}

void CPOBaseApp::updateTimerStart(i32 delay_confirm_ms)
{
	m_update_timer.setInterval(delay_confirm_ms);
	m_update_timer.setSingleShot(true);
	m_update_timer.start();
}

void CPOBaseApp::updateTimerStop()
{
	m_update_timer.stop();
}

void CPOBaseApp::onTimerUpdate()
{
	m_update_timer.stop();
	i32 conn = m_app_update.getFrom();
	
	if (deviceUpdate())
	{
		m_ivs_server.updateState(kIVSStateUpdateCompleted, conn);
	}
	else
	{
		m_ivs_server.updateState(kIVSStateUpdateFailed, conn);
	}
}

void CPOBaseApp::setAuthIdPassword(postring& str_auth_id, postring& str_auth_password)
{
	m_usb_dongle.m_auth_str_id = str_auth_id;
	m_usb_dongle.m_auth_str_password = str_auth_password;
}

void CPOBaseApp::updateNetworkAdapters()
{
	CIPInfo* ip_info_ptr = getIPInfo();
	if (!ip_info_ptr)
	{
		return;
	}

	anlock_guard_ptr(ip_info_ptr);
	{
		NetAdapterArray& adapter_vec = ip_info_ptr->m_netadapter_vec;
		COSBase::getNetworkAdapters(adapter_vec);

#if defined(POR_EMULATOR) && (0)
		//remove all non-loop-back network interface, if app is simulator
		for (i32 i = (i32)adapter_vec.size() - 1; i >= 0; i--)
		{
			if (adapter_vec[i].is_loopback)
			{
				continue;
			}
			CPOBase::eraseInVector(adapter_vec, i);
		}

		//remove all net adapter from 2nd to end, if system has multiple loop-back network interface
		if (adapter_vec.size() > 1)
		{
			adapter_vec.erase(adapter_vec.begin() + 1, adapter_vec.end());
		}
#endif
	}
}

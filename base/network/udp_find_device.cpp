#include "base.h"
#include "udp_find_device.h"
#include "connection.h"
#include "struct.h"
#include "netlink/core.h"

CUDPDeviceFinder::CUDPDeviceFinder()
{
	m_is_inited = false;
	m_is_thread_canceled = false;
	m_udp_port = PO_NETWORK_PORT;

	CConnection::initNL();
}

CUDPDeviceFinder::~CUDPDeviceFinder()
{
	exitInstance();
}

void CUDPDeviceFinder::initInstance(int port, postring strModelName, postring strVersion)
{
	exitInstance();
	if (!m_is_inited)
	{
		m_is_inited = true;
		m_is_thread_canceled = false;

		m_udp_port = port;
		m_heart_beat_version = strVersion;
		m_device_model_name = strModelName;

		QThreadStart();
	}
}

void CUDPDeviceFinder::exitInstance()
{
	if (m_is_inited)
	{
		m_is_inited = false;
		m_is_thread_canceled = true;
		QThreadStop();
	}
}


bool CUDPDeviceFinder::getUDPPacket(HeartBeat &hb, FDPacket::_FDData* pak)
{
	if (!pak)
	{
		return false;
	}

	if ((pak->pak.token != POTOKEN_START) || (pak->pak.type = kPOHeartBeatUdp))
	{
		return false;
	}

	u8* pbuffer = (u8*)&pak->pak.buf;
	i32 buffer_size = pak->pak.len;

	CPOBase::memRead(hb.hb_ver, pbuffer, buffer_size);
	if (hb.hb_ver != PO_HBVERSION)
	{
		return false;
	}

	CPOBase::memRead(hb.dev_id, pbuffer, buffer_size);
	CPOBase::memRead(hb.is_emulator, pbuffer, buffer_size);
	CPOBase::memRead(pbuffer, buffer_size, hb.device_name);
	CPOBase::memRead(pbuffer, buffer_size, hb.device_model);
	if (hb.device_model != m_device_model_name)
	{
		return false;
	}

	CPOBase::memRead(pbuffer, hb.device_version, buffer_size);
	if (hb.device_version != m_heart_beat_version)
 	{
 		return false;
 	}

	CPOBase::memRead(hb.cmd_port, pbuffer, buffer_size);
	CPOBase::memRead(hb.ivs_port, pbuffer, buffer_size);
	CPOBase::memRead(hb.state, pbuffer, buffer_size);
	CPOBase::memRead(hb.connections, pbuffer, buffer_size);
	CPOBase::memRead(hb.duration, pbuffer, buffer_size);

	int num = 0;
	CPOBase::memRead(num, pbuffer, buffer_size);

	for (int i = 0; i < num; i++)
	{
        NetAdapter adapter;
		adapter.memRead(pbuffer, buffer_size);
		hb.netadapter_vec.push_back(adapter);
	}

	return true;
}


void CUDPDeviceFinder::run()
{
	CConnection::initNL();

#ifdef POR_WINDOWS
	SOCKET clientSocket = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (clientSocket == -1)
	{
		// Socket Create Error.
		return;
	}

#elif defined(POR_LINUX)
	int clientSocket = socket(PF_INET, SOCK_DGRAM, 0);
	if (clientSocket == -1)
	{
		// "Error in creating socket";
		return;
	}
#endif

	sockaddr_in hbServerAddr;

	memset(&hbServerAddr, 0, sizeof(hbServerAddr));
	hbServerAddr.sin_family = AF_INET;
	hbServerAddr.sin_port = htons(m_udp_port);
	hbServerAddr.sin_addr.s_addr = INADDR_ANY;

	//int len = sizeof(HBserveraddr);

	if (bind(clientSocket, (sockaddr*)&hbServerAddr, sizeof(sockaddr_in)) < 0)
	{
		// ERROR binding in the server socket;
		printlog_lv0("[Error] Binding Error. Port num is being used.");
		return;
	}
	
	FDPacket fdpacket;
	int nFDHeaderSize = sizeof(fdpacket.d.pak.len) + sizeof(fdpacket.d.pak.token) + sizeof(fdpacket.d.pak.type);;

	char buf[MAX_HBPACKET_SIZE];
	while (!m_is_thread_canceled)
	{
		fd_set fds;
		struct timeval timeout;
		timeout.tv_sec = 0;
		timeout.tv_usec = 500;

		FD_ZERO(&fds);
		FD_SET(clientSocket, &fds);

		int rc = select(sizeof(fds) * 8, &fds, NULL, NULL, &timeout);
		if (rc > 0)
		{
			sockaddr_in clientaddr;
			int len = sizeof(clientaddr);
			int readenBytes = 0;
			if ((readenBytes = recvfrom(clientSocket, buf, MAX_HBPACKET_SIZE, 0, (sockaddr*)&clientaddr, (socklen_t*)&len)) > 0)
			{
				FDPacket::_FDData* pak = (FDPacket::_FDData*)buf;
				if (pak->pak.len + nFDHeaderSize == readenBytes)
				{
					onReceiveHB(pak, ntohl(clientaddr.sin_addr.s_addr));
				}
				else
				{
					printlog_lv0("HeartBeat packet corrupted.");
				}
			}
		}

		SLEEP(1);
	}

#ifdef OS_WIN32
	closesocket(clientSocket);
#else
	close(clientSocket);
#endif

	printlog_lv1("UDP FindDevice Thread is exited.");
}

bool CUDPDeviceFinder::fromHBPacket(ServerInfo& si, HeartBeat& hb, i32 ip)
{
	si.dev_type = PO_CUR_DEVICE;
	si.dev_id = hb.dev_id;
	si.dev_state = hb.state;
	si.cmd_port = hb.cmd_port;
	si.device_name = hb.device_name;
	si.device_desc = hb.device_version;

	si.adapter_name = "";
	si.dev_ip = po::IpA2N("127.0.0.1");
	si.subnet_mask = po::IpA2N("255.255.255.0");
	si.is_conf_dhcp = false;
	si.duration = hb.duration;

	bool bFoundAdapter = false;
	if (ip == 0)
	{
		bFoundAdapter = true;
	}
	else
	{
		for (int i = 0; i < hb.netadapter_vec.size(); i++)
		{
			if (hb.netadapter_vec[i].ip_address == ip)
			{
				si.dev_ip = hb.netadapter_vec[i].ip_address;
				si.subnet_mask = hb.netadapter_vec[i].ip_subnet;
				si.is_conf_dhcp = hb.netadapter_vec[i].is_conf_dhcp;
				si.adapter_name = hb.netadapter_vec[i].adapter_name;
				si.dev_adapter = i;

				bFoundAdapter = true;
				break;
			}
		}
	}
	
	return bFoundAdapter;
}

void CUDPDeviceFinder::onReceiveHB(FDPacket::_FDData* pak, int ip)
{
	HeartBeat hb;
	if (!getUDPPacket(hb, pak))
	{
		return;
	}

	ServerInfo si;
	bool bFoundAdapter = fromHBPacket(si, hb, ip);

	if (bFoundAdapter)
	{
		processHB(si);
		onParseHB(si, hb.netadapter_vec[si.dev_adapter]);
	}
	else
	{
		simplelog("No adapter Info.");
	}
}


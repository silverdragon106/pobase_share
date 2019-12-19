#include "udp_broadcast.h"
#include "connection.h"
#include "app/base_app.h"

CUdpBroadCast::CUdpBroadCast()
{
	m_is_inited = false;
	m_is_tread_cancel = false;
	m_port = PO_NETWORK_PORT;
	m_base_app = NULL;

	CConnection::initNL();
}

CUdpBroadCast::~CUdpBroadCast()
{
	exitInstance();
}

void CUdpBroadCast::initInstance(CPOBaseApp* main_app_ptr, i32 port)
{
	exitInstance();
	if (!m_is_inited)
	{
		m_is_inited = true;
		m_is_tread_cancel = false;
		m_port = port;
		m_base_app = main_app_ptr;

		QThreadStart();
	}
}

void CUdpBroadCast::exitInstance()
{
	if (m_is_inited)
	{
		m_is_inited = false;
		m_is_tread_cancel = true;
		QThreadStop();
	}
}

void CUdpBroadCast::run()
{
	singlelog_lv0("The UDPBroadCaster thread is");

	bool is_first;
	i32 i, count, rcode;
	FDPacket pak;
	HeartBeat hb;

	while (!m_is_tread_cancel)
	{
#if defined(POR_WINDOWS)
		SOCKET s = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
		if (s == -1)
		{
			// "Error in creating socket";
			return;
		}

		char opt = 1;
		setsockopt(s, SOL_SOCKET, SO_BROADCAST, (char*)&opt, sizeof(char));
		SOCKADDR_IN brdcastaddr;
#elif defined(POR_LINUX)
        i32 s = socket(PF_INET, SOCK_DGRAM, 0);
		if (s == -1)
		{
			// "Error in creating socket";
			return;
		}

		struct sockaddr_in brdcastaddr;
		i32 is_broadcast_enabled = 1;
        setsockopt(s, SOL_SOCKET, SO_BROADCAST, &is_broadcast_enabled, sizeof(is_broadcast_enabled));
#endif

		i64 prev_time = 0, cur_time = 0;
		is_first = true;

		memset(&brdcastaddr, 0, sizeof(brdcastaddr));
		brdcastaddr.sin_family = AF_INET;
		brdcastaddr.sin_port = htons((short)m_port);

		while (!m_is_tread_cancel)
        {
			cur_time = sys_cur_time;
			if (cur_time - prev_time > UDP_HBTIME_INTERVAL)
			{
				if (m_base_app && m_base_app->getHeartBeat(hb))
				{
					rcode = 0;
					NetAdapterArray& netlist = hb.netadapter_vec;
					hb.setUDPPacket(&pak);

					count = (i32)netlist.size();
					for (i = 0; i < count; i++)
					{
						NetAdapter& adpt = netlist[i];
						if (adpt.is_loopback)
						{
							brdcastaddr.sin_addr.s_addr = htonl(adpt.ip_address | 0xFFFFFF);
						}
						else
						{
							brdcastaddr.sin_addr.s_addr = htonl(adpt.ip_address | ~adpt.ip_subnet);
						}
						rcode = sendto(s, pak.d.data, pak.len, 0, (sockaddr*)&brdcastaddr, sizeof(brdcastaddr));
						if (rcode < 0)
						{
							break;
						}
					}
					if (rcode < 0)
					{
						break;
					}

					if (rcode > 0) //used for only logging mode
					{
						if (is_first)
						{
							is_first = false;
							printlog_lv1(QString("The first UDP Heartbeat packet is broadcasted, count:%1, size: %2.").arg(count).arg(rcode));
						}
						printlog_lvs3("The UDP Heartbeat packet is broadcasted.", LOG_SCOPE_NET);
					}
				}
				prev_time = cur_time;
			}
			SLEEP(UDP_HBDELAY_INTERVAL);
        }
#if defined(POR_WINDOWS)
        ::closesocket(s);
#elif defined(POR_LINUX)
        close(s);
#endif
		SLEEP(UDP_HBDELAY_INTERVAL);
	}
}

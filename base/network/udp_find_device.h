#pragma once

#include "netlink/socket.h"
#include "packet.h"
#include <QThread>
#include <QMutex>

using namespace NL;

#define UDP_HBINTERVAL			100
#define UDP_HBTIME_INTERVAL		30
#define UDP_HBRETRY_NUM			4

#define MAX_HBPACKET_SIZE		1024
#define MAX_HBPACKET_DATA_SIZE	1024

class CUDPDeviceFinder : public QThread
{
public:
	CUDPDeviceFinder();
	~CUDPDeviceFinder();

	void					initInstance(int port, postring strModelName, postring strVersion);
	void					exitInstance();
	bool					getUDPPacket(HeartBeat &hb, FDPacket::_FDData* upak);

	void					onReceiveHB(FDPacket::_FDData* pak, int ip);
	virtual void			processHB(ServerInfo si) = 0;
	virtual void			onParseHB(ServerInfo si, NetAdapter net_info){}

	static bool				fromHBPacket(ServerInfo& si, HeartBeat& hb, i32 ip = 0);

private:
	void					run() Q_DECL_OVERRIDE;

protected:
	bool					m_is_inited;

	postring				m_heart_beat_version;
	postring				m_device_model_name;

	std::atomic<int>		m_udp_port;
	std::atomic<bool>		m_is_thread_canceled;
};

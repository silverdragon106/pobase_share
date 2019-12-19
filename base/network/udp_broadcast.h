#pragma once

#include "netlink/socket.h"
#include "packet.h"
#include <QThread>
#include <QMutex>

using namespace NL;

#define UDP_HBTIME_INTERVAL		3000
#define UDP_HBDELAY_INTERVAL	100
#define UDP_HBRETRY_NUM			4

class CPOBaseApp;
class CUdpBroadCast : public QThread
{
	Q_OBJECT

public:
	CUdpBroadCast();
	virtual ~CUdpBroadCast();

	void					initInstance(CPOBaseApp* main_app_ptr, i32 port);
	void					exitInstance();

private:
	void					run() Q_DECL_OVERRIDE;

protected:
	bool					m_is_inited;
	std::atomic<bool>		m_is_tread_cancel;

	std::atomic<i32>		m_port;
	CPOBaseApp*				m_base_app;
};


#pragma once

#define POTOKEN_START		0xb9896949
#define POTOKEN_END			0xa1b1c1d1
#define POPING_TIMEOUT		10000
#define POPING_DELAY		10000

#include "packet.h"
#include "netlink/socket.h"
#include "netlink/socket_group.h"
#include <string.h>
#include <list>
#include <QMutex>
#include "logger/logger.h"

using namespace NL;

enum ConnectionErrorType
{
	kConnectionErrPacket = 0,
	kConnectionErrUnknown,
	kConnectionTimeout,
};

enum WriteModeType
{
	kPacketWritePrepare = 0,
	kPacketWriteHeader,
	kPacketWriteBody
};

class CConnection;
class CConnectionListener
{
public:
	virtual void			onRead(Packet* /*packet_ptr*/, CConnection* /*connection_ptr*/){}
	virtual void			onConnected(CConnection* /*connection_ptr*/){}
	virtual void			onDisconnected(CConnection* /*connection_ptr*/){}
	virtual void			onError(ConnectionErrorType /*err*/, CConnection* /*connection_ptr*/){}
};

class CConnection
{
	ERR_DEFINE(0)

	friend class CServer;
	friend class CClient;

public:
	CConnection(Socket* socket_ptr = NULL, CConnectionListener* listener_ptr = NULL);
	virtual ~CConnection();

	void					clearPacketQueue();

	bool					onRead();				// called when a piece of packet_ptr is arrived through socket.
	bool					onWrite();

	void					sendPacket(Packet* packet_ptr, bool is_omitable, bool need_lock = true);
	bool					writePacket(Packet* packet_ptr);

	void					writeLock();
	void					writeUnlock();

	Socket*					getSocket();
	void					setSocket(Socket* sock);
	void					setListener(CConnectionListener* listener_ptr);

	inline i32				getID() { return m_address; };
	inline i64				getData64() { return m_reserved; };
	inline void*			getData() { return m_reserved_ptr; };
	inline void				setID(i32 id) { m_address = id; };
	inline void				setData(void* data_ptr) { m_reserved_ptr = data_ptr; };
	inline void				setData64(i64 data) { m_reserved = data; };
		
	static void				initNL();

protected:
	bool					checkStartToken(i32& available_bytes);
	bool					checkEndToken(Packet* packet_ptr);
	void					prepareNewRead();
	bool					prepareNewWrite(Packet* packet_ptr);
	void					freeWriteBuffer(bool is_free_packet);

	void					onCompletedPacket();
	void					onError(ConnectionErrorType err);	// socket error occurred.

private:
	i32						m_read_token_start;
	i32						m_read_token_end;
	i32						m_read_size;
	i32						m_read_pos;
	Packet*					m_read_packet_ptr;

	i32						m_write_mode;
	i32						m_write_size;
	i32						m_write_pos;
	u8*						m_write_buffer_ptr;
	bool					m_write_buffer_allocated;
	Packet*					m_write_packet_ptr;

	Socket*					m_socket_ptr;
	CConnectionListener*	m_conn_listener_ptr;
	
	i32						m_queue_size;
	PacketIDMap				m_queue_map;
	PacketPList				m_packet_queue;
	PacketPList				m_write_queue;
	POMutex					m_packet_queue_mutex;
	POMutex					m_write_queue_mutex;

	i32						m_address;
	i32						m_port;
	void*					m_reserved_ptr;
	i64						m_reserved;
	i64						m_last_read_time;

	static bool				s_is_3rdlib_inited;
};



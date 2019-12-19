#pragma once

#include "struct.h"
#include "network/packet.h"
#include <QThread>
#include <QSystemSemaphore>
#include <QSharedMemory>
#include <QSharedPointer>
#include <QMutex>
#include <QByteArray>
#include <QDataStream>
#include <QWaitCondition>

#pragma pack(push, 4)

#define IPC_QUEUE_MAXSIZE		100

enum IPCInternalCommands
{
	IPC_SendArugments = 0xF001
};

struct IPCHeader
{
	i32				packet_count;
	i32				packet_pos[IPC_QUEUE_MAXSIZE];
	i32				packet_len[IPC_QUEUE_MAXSIZE];
	i64				packet_uid[IPC_QUEUE_MAXSIZE];	// device identify id array.
	i32				write_index;					// last written index in nPacketPos
	i32				read_index;						// last readen index in nPacketPos.
	u8				is_connected;
};
#pragma pack(pop)

typedef QSharedPointer<Packet> PacketRef;
Q_DECLARE_METATYPE(PacketRef);

class CIPCManager : public QThread, public CVirtualEncoder
{
	Q_OBJECT
public:
	enum SideType
	{
		kSideA,		// Write Side.
		kSideB,		// Read Side.
		kSideCount
	};

public:
	CIPCManager();
	virtual ~CIPCManager();

	void					initInstance();
	void					initInstance(SideType side, const postring& share_mem_key, i32 mem_size = 0);
	void					exitInstance();
	void					resetIPC(bool is_connect);

	bool					isCanceled() const;
	bool					isPaired() const;
	bool					isInited() const;

	void					send(PacketRef pak);

	bool					setIPCInfo(SideType side, const postring& strkey, i32 memsize = 0);
	void					setRing(bool b);
	void					setUid(u64 uuid);

	void					setEnableSignalWaiting(bool is_enable_waiting);
	void					waitForReadyToEmit();
	void					wakeReadyToEmit();

signals:
	void					receivedIpcPacket(PacketRef);
	// SideA
	void					started();

	// SideA and SideB
	void					paired();
	void					pairedTimeout();

protected:
	void					run() Q_DECL_OVERRIDE;

	bool					create();
	void					release();

	void					read();
	bool					write(PacketRef pak);
	bool					isWritable();

	void					arrangeHeaderInfo(IPCHeader* ipc_header_ptr);

	virtual bool			acquireEncoder(i32 encoder, i32 w, i32 h, i32 channel, i32 frate, i32 brate, i32 vid);
	virtual void			releaseEncoder();
	virtual void			setImageToEncoder(ImageData* img_data_ptr);
	virtual void			setImageToEncoder(ImageData* img_data_ptr, i32 cam_id);

	virtual void*			onEncodedFrame(u8* buffer_ptr, i32 size, i64 pts, ImageData* img_data_ptr);
	virtual void			onSendFrame(void* send_void_ptr);

private:
	QSharedMemory*			m_shared_mem_ptr;
	QMutex					m_queue_mutex;
	QList<PacketRef>		m_queue;

	postring				m_shared_key;
	SideType				m_side_type;
	i32						m_written;
	u64						m_uuid;		// used in case of SideB.

	bool					m_is_inited;
	std::atomic<bool>		m_is_threadcancel;

	i32						m_mem_size;			// mem size in bytes. used in SideA
	i32						m_time_out;			// connection timeout used in SideB.
	bool					m_is_paired;		// accepted flag used in SideA | SideB.
	bool					m_is_ring_queue;	// packet queue is ring ? if ring, write packet without considering the packet is read or not.

	bool					m_is_enable_signal_waiting;
	i32						m_emitted_signals;			// notified receivedIpcPacket signal count.
	QMutex					m_signal_mutex;
	QWaitCondition			m_wait_condition;	//
};

#include "ipc_manager.h"
#include "base.h"
#include "network/packet.h"

#define IsNullRef(ref)		(ref).isNull()
#define IsNotNullRef(ref)	!(ref).isNull()

//////////////////////////////////////////////////////////////////////////
CIPCManager::CIPCManager()
{
	m_is_inited = false;
	m_is_threadcancel = false;

	m_shared_mem_ptr = NULL;

	m_side_type = kSideA;
	m_written = 0;
	m_mem_size = 0;
	m_uuid = 0;
	m_time_out = 3000;
	m_is_ring_queue = true;

	QMutexLocker l(&m_queue_mutex);
	m_queue.clear();
}

CIPCManager::~CIPCManager()
{
	exitInstance();
}

void CIPCManager::initInstance(SideType side, const postring& share_mem_key, i32 mem_size)
{
	if (!m_is_inited)
	{
		singlelog_lv0("IPCManager InitInstance");

		m_side_type = side;
		m_shared_key = share_mem_key;
		m_mem_size = mem_size;
		if (m_mem_size <= 0 && m_side_type == kSideA)
		{
			printlog_lv1("Must be set IPC Memory size for sideA.");
			return;
		}
		
		m_is_inited = false;
		m_is_paired = false;
		m_is_threadcancel = false;
		
		if (create())
		{
			m_is_inited = true;
			QThreadStart();
		}
	}
}

void CIPCManager::initInstance()
{
	initInstance(m_side_type, m_shared_key, m_mem_size);
}

void CIPCManager::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0("IPCManager ExitInstance");

		m_is_inited = false;
		m_is_threadcancel = true;
		QThreadStop1(1000);
		release();

		{
			QMutexLocker l(&m_queue_mutex);
			m_queue.clear();
		}
	}
}

void CIPCManager::resetIPC(bool is_connected)
{
	if (is_connected)
	{
		initInstance();
	}
	else
	{
		exitInstance();
	}
}

void CIPCManager::send(PacketRef pak)
{
	if (m_side_type != kSideA)
	{
		printlog_lvs2("Must be SideA to send via IPC.", LOG_SCOPE_IPC);
		return;
	}
	
	m_queue_mutex.lock();
	if (m_queue.size() < IPC_QUEUE_MAXSIZE)
	{
		m_queue.append(pak);
		m_queue_mutex.unlock();
	}
	else
	{
		i32 queue_size = m_queue.size();
		m_queue_mutex.unlock();
		printlog_lvs2(QString("IPC queue overflow. queue size is %1.").arg(queue_size), LOG_SCOPE_IPC);
	}
}

void CIPCManager::run()
{
	singlelog_lv0("The IPCManager thread is");
	bool is_writable = false;

	while (!isCanceled())
	{
		// Create
		if (m_side_type == kSideA)
		{
			if (isWritable())
			{
				PacketRef pak;
				is_writable = false;

				m_queue_mutex.lock();
				if (m_queue.size() > 0)
				{
					is_writable = true;
					pak = m_queue.first();
				}
				m_queue_mutex.unlock();

				if (is_writable && write(pak))
				{
					QMutexLocker l(&m_queue_mutex);
					m_queue.takeFirst();
				}
			}
			QThread::msleep(1);
		}
		else if (m_side_type == kSideB)
		{
			read();
			QThread::msleep(10);
		}
	}
}

bool CIPCManager::create()
{
	release();

	m_shared_mem_ptr = po_new QSharedMemory(m_shared_key.c_str());
	if (!m_shared_mem_ptr)
	{
		return false;
	}

	if (m_side_type == kSideA)
	{
		if (m_shared_mem_ptr)
		{
			if (!m_shared_mem_ptr->create(m_mem_size))
			{
				printlog_lv1("Failed to create shared memory.");
				return false;
			}

			m_shared_mem_ptr->lock();
			{
				// init shared memory header.
				char* buffer_ptr = (char*)m_shared_mem_ptr->data();
				memset(buffer_ptr, 0, sizeof(IPCHeader));
			}
			m_shared_mem_ptr->unlock();

			// emited sideA started.
			emit started();
			printlog_lv1("IPC Started.");
		}
	}
	else if (m_side_type == kSideB)
	{
		i32 time_out = m_time_out;

		// try to connect during timeout.
		while (true)
		{
			if (m_shared_mem_ptr->attach())
			{
				break;
			}

			if (!isCanceled() && time_out > 0)
			{
				printlog_lv1("Retry connect via IPC.");
			}
			else
			{
				printlog_lv1("Failed to attach shared memory.");
				if (time_out <= 0)
				{
					emit pairedTimeout();
				}
				return false;
			}

			QThread::msleep(100);
			time_out -= 100;
		}

		m_is_paired = true;
		printlog_lv1("IPC Paired.");
		emit paired();
	}

	return true;
}

void CIPCManager::release()
{
	POSAFE_DELETE(m_shared_mem_ptr);

#ifdef POR_LINUX
	//on linux/unix shared memory is not freed upon crash
	//so if there is any trash from previous instance, clean it
	QSharedMemory shmem_fix_crash(m_shared_key.c_str());
	if (shmem_fix_crash.attach())
	{
		shmem_fix_crash.detach();
	}
#endif
}

bool CIPCManager::write(PacketRef pak)
{
	if (!m_shared_mem_ptr || IsNullRef(pak))
	{
		return true;
	}

	m_shared_mem_ptr->lock();
	u8* data_ptr = (u8*)m_shared_mem_ptr->data();
	i32 mem_size = m_shared_mem_ptr->size() - sizeof(IPCHeader);
	i32 written_bytes = 0;

	IPCHeader* header_ptr = (IPCHeader*)data_ptr;
	i32 cur_write_index = header_ptr->write_index;
	i32 prev_write_index = cur_write_index - 1;
	if (prev_write_index < 0)
	{
		prev_write_index += IPC_QUEUE_MAXSIZE;
	}

	// previous packet address pos.
	written_bytes = header_ptr->packet_pos[prev_write_index] + header_ptr->packet_len[prev_write_index];
	if (header_ptr->packet_count >= IPC_QUEUE_MAXSIZE)
	{
		m_shared_mem_ptr->unlock();
		printlog_lvs2(QString("IPC PacketQueue is full %1").arg(header_ptr->packet_count), LOG_SCOPE_IPC);
		return false;
	}

	i32 pak_size = pak->memSize();
	i32 readen_index = (header_ptr->read_index + IPC_QUEUE_MAXSIZE - 1) % IPC_QUEUE_MAXSIZE;
	i32 read_pos = header_ptr->packet_pos[readen_index] + header_ptr->packet_len[readen_index];
	i32 rw_offset = read_pos - written_bytes;
	if (rw_offset >= 0 && rw_offset < pak_size && header_ptr->packet_count)
	{
		m_shared_mem_ptr->unlock();
		printlog_lvs2(QString("IPC can't write(1) packet, rwoffset %1, len %2, pak count %3")
						.arg(rw_offset).arg(pak_size).arg(header_ptr->packet_count), LOG_SCOPE_IPC);
		return false;
	}
	
	if (pak_size + written_bytes <= mem_size)
	{
		data_ptr += (written_bytes + sizeof(IPCHeader));
	}
	else if (pak_size <= mem_size)
	{
		written_bytes = 0;
		data_ptr += sizeof(IPCHeader);

		// 쓰기위치가 바뀌므로 읽기위치와의 관계를 다시 검사해야 한다.
		i32 rw_offset = read_pos - written_bytes;
		if (rw_offset >= 0 && rw_offset < pak_size && header_ptr->packet_count)
		{
			m_shared_mem_ptr->unlock();
			printlog_lvs2(QString("IPC can't write(2) packet, rwoffset %1, len %2, pak count %3")
							.arg(rw_offset).arg(pak_size).arg(header_ptr->packet_count), LOG_SCOPE_IPC);
			return false;
		}
	}
	else
	{
		data_ptr = NULL;
		m_shared_mem_ptr->unlock();
		printlog_lvs2(QString("Insufficient shared memory to write. len is %1, size is %2")
						.arg(pak_size).arg(mem_size), LOG_SCOPE_IPC);
		return true;
	}

	if (data_ptr)
	{
		pak->memWrite(data_ptr);

		header_ptr->packet_pos[cur_write_index] = written_bytes;
		header_ptr->packet_len[cur_write_index] = pak_size;
		header_ptr->packet_uid[cur_write_index] = m_uuid;
		header_ptr->write_index = (cur_write_index + 1) % IPC_QUEUE_MAXSIZE;
		header_ptr->packet_count++;

		printlog_lvs4(QString("IPC send packet%1, cmd %2, sub_cmd %3, len %4, write pos %5, read pos %6")
						.arg(cur_write_index).arg(pak->getCmd()).arg(pak->getSubCmd()).arg(pak_size)
						.arg(written_bytes).arg(read_pos), LOG_SCOPE_IPC);
	}
	m_shared_mem_ptr->unlock();
	return true;
}

void CIPCManager::read()
{
	if (!m_shared_mem_ptr)
	{
		return;
	}

	PacketRef pak;
	i32 packet_count = 0;

	m_shared_mem_ptr->lock();
	i32 data_size = m_shared_mem_ptr->size();
	u8* data_ptr = (u8*)m_shared_mem_ptr->data();

	IPCHeader* header = (IPCHeader*)data_ptr;
	if (!header->is_connected)
	{
		header->is_connected = true;
	}
	if (header->packet_count > 0)
	{
		if (header->packet_uid[header->read_index] == m_uuid)
		{
			data_ptr += (header->packet_pos[header->read_index] + sizeof(IPCHeader));

			pak = PacketRef(po_new Packet());
			pak->memRead(data_ptr);

			header->packet_count--;
			packet_count = header->packet_count;
			header->read_index = (header->read_index + 1) % IPC_QUEUE_MAXSIZE;
		}
	}
	m_shared_mem_ptr->unlock();

	if (IsNotNullRef(pak))
	{
        if (!signalsBlocked())
        {
            emit receivedIpcPacket(pak);
        }
	}
}

bool CIPCManager::isCanceled() const
{
	return m_is_threadcancel;
}

bool CIPCManager::isPaired() const
{
	return m_is_paired && m_is_inited;
}

bool CIPCManager::isInited() const
{
	return m_is_inited;
}

bool CIPCManager::isWritable()
{
	bool is_success = false;
	bool is_connected = false;
	i32 packet_count = 0;

	m_shared_mem_ptr->lock();
	{
		IPCHeader* header_ptr = (IPCHeader*)m_shared_mem_ptr->data();
		is_connected = header_ptr->is_connected;
		packet_count = header_ptr->packet_count;
	}
	m_shared_mem_ptr->unlock();
	
	if (m_is_ring_queue || packet_count < IPC_QUEUE_MAXSIZE)
	{
		is_success = true;
	}

	// connected flag is changed to true at first time, then emit accepted signal.
	if (!m_is_paired && is_connected && m_side_type == kSideA)
	{
		m_is_paired = true;
		printlog_lv1("IPC Paired.");
		emit paired();
	}
	return is_success;
}

void CIPCManager::arrangeHeaderInfo(IPCHeader* header_ptr)
{
	i32 wx = header_ptr->write_index - 1;
	if (wx < 0)
	{
		wx += IPC_QUEUE_MAXSIZE;
	}

	i32 wp = header_ptr->packet_pos[wx];
	i32 wl = header_ptr->packet_len[wx];
	i32 ix = 0;

	// check overhead.
	for (i32 i = 0; i < header_ptr->packet_count; i++)
	{
		ix = (header_ptr->read_index + i) % IPC_QUEUE_MAXSIZE;
		i32 ip = header_ptr->packet_pos[ix];
		i32 il = header_ptr->packet_len[ix];

		// check intersects read address and write address.
		// 1. -wp-ip-- || -ip-wp--(perhaps not such case) || 
		if ((wp <= ip && wp + wl > ip) || (ip <= wp && ip + il > wp))
		{
			continue;
		}
		break;
	}
	header_ptr->read_index = ix;

	if (m_is_ring_queue)
	{
		// modify Read pos refer Write pos. prevent to write pos pass over read pos.
		// At this point, nIndexToWrite > nIndexToRead in sequential. 
		// when nIndexToWrite == nIndexToRead is the time that the packet queue is full.
		if (header_ptr->write_index == header_ptr->read_index)
		{
			// if packet queue is full and overwrite packet, then increase nIndexToRead.
			header_ptr->read_index = (header_ptr->read_index + 1) % IPC_QUEUE_MAXSIZE;
		}
	}
}

void CIPCManager::setUid(u64 nUuid)
{
	m_uuid = nUuid;
}

void CIPCManager::setRing(bool b)
{
	m_is_ring_queue = b;
}

bool CIPCManager::setIPCInfo(SideType side, const postring& strkey, i32 memsize)
{
	m_side_type = side;
	m_shared_key = strkey;
	m_mem_size = memsize;

	return !(m_mem_size <= 0 && m_side_type == kSideA);
}

void CIPCManager::setEnableSignalWaiting(bool b)
{
	m_is_enable_signal_waiting = b;
}

void CIPCManager::waitForReadyToEmit()
{
	if (!m_is_enable_signal_waiting)
	{
		return;
	}

	m_signal_mutex.lock();
	m_emitted_signals++;
	if (m_emitted_signals >= 10)
	{
		m_wait_condition.wait(&m_signal_mutex);
	}
	m_signal_mutex.unlock();
}

void CIPCManager::wakeReadyToEmit()
{
	if (!m_is_enable_signal_waiting)
	{
		return;
	}

	m_signal_mutex.lock();
	m_emitted_signals--;
	if (m_emitted_signals < 10)
	{
		m_wait_condition.wakeAll();
	}
	m_signal_mutex.unlock();
}

bool CIPCManager::acquireEncoder(i32 encoder, i32 w, i32 h, i32 channel, i32 frate, i32 brate, i32 vid)
{
	return isPaired();
}

void CIPCManager::releaseEncoder()
{
}

void CIPCManager::setImageToEncoder(ImageData* img_data_ptr)
{
	onSendFrame(onEncodedFrame(NULL, -1, 0, img_data_ptr));
}

void CIPCManager::setImageToEncoder(ImageData* img_data_ptr, i32 cam_id)
{
	onSendFrame(onEncodedFrame(NULL, cam_id, 0, img_data_ptr));
}

void* CIPCManager::onEncodedFrame(u8* buffer_ptr, i32 size, i64 pts, ImageData* img_data_ptr)
{
	assert(false);
	return NULL;
}

void CIPCManager::onSendFrame(void* send_void_ptr)
{
	assert(false);
}

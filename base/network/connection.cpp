#include "connection.h"
#include "server.h"
#include "packet.h"
#include "base.h"
#include "logger/logger.h"
#include <algorithm>

#define RETURN_READ(a, b)		if ((n=(m_socket_ptr->read((char*)(a),(b)))) != (b)) return false;
#define RETURN_CHECK(a, b)		if ((n=(m_socket_ptr->read((char*)(a),(b)))) < 0) return false;

const i32 kPOServerMaxQueue = 50;
const i32 kPOServerMaxWrite = 10;

bool CConnection::s_is_3rdlib_inited = false;

//////////////////////////////////////////////////////////////////////////
// CConnection class
//////////////////////////////////////////////////////////////////////////

CConnection::CConnection(Socket* socket_ptr, CConnectionListener* listener_ptr)
{
	m_read_token_start = -1;
	m_read_token_end = -1;
	
	m_address = 0;
	m_port = 0;
	m_reserved = 0;
	m_last_read_time = -1;
	m_reserved_ptr = NULL;

	m_read_packet_ptr = NULL;
	m_conn_listener_ptr = listener_ptr;

	ERR_PREPARE(0);
	m_queue_size = 0;
	m_packet_queue.clear();
	m_write_queue.clear();
	m_queue_map.clear();

	m_write_packet_ptr = NULL;
	m_write_buffer_ptr = NULL;
	m_write_buffer_allocated = false;
	m_write_pos = 0;
	m_write_size = 0;
	
	if (!socket_ptr)
	{
		printlog_lv1("Error new connection, socket is NULL");
		return;
	}

	initNL();
	prepareNewRead();
	freeWriteBuffer(false);
	setSocket(socket_ptr);
	setID(socket_ptr->getIPAddress());

	if (m_socket_ptr && m_conn_listener_ptr)
	{
		m_conn_listener_ptr->onConnected(this);
	}
}

void CConnection::initNL()
{
	if (!s_is_3rdlib_inited)
	{
		NL::init();
		s_is_3rdlib_inited = true;
	}
}

CConnection::~CConnection()
{
	if (m_conn_listener_ptr)
	{
		m_conn_listener_ptr->onDisconnected(this);
	}

	clearPacketQueue();
	freeWriteBuffer(false);
	POSAFE_DELETE(m_read_packet_ptr);
}

void CConnection::clearPacketQueue()
{
	{
		POMutexLocker l(m_packet_queue_mutex);
		for (PacketPList::iterator iter = m_packet_queue.begin(); iter != m_packet_queue.end(); iter++)
		{
			POSAFE_DELETE(*iter);
		}
		m_queue_size = 0;
		m_packet_queue.clear();
		m_queue_map.clear();
	}
	{
		POMutexLocker l(m_write_queue_mutex);
		for (PacketPList::iterator iter = m_write_queue.begin(); iter != m_write_queue.end(); iter++)
		{
			POSAFE_DELETE(*iter);
		}
		m_write_queue.clear();
	}
}

void CConnection::onError(ConnectionErrorType err)
{
	if (m_conn_listener_ptr)
	{
		m_conn_listener_ptr->onError(err, this);
	}
}

bool CConnection::onRead()
{
	if (!m_socket_ptr)
	{
		return false;
	}

	i32 n = 0;
	i32 available_bytes = 0;
	i32 check_header_bytes = Packet::calcHeaderSize();
	//////////////////////////////////////////////////////////////////////////
	// check to read.
	try
	{
		available_bytes = m_socket_ptr->nextReadSize();

		while (available_bytes > 0)
		{
			if (m_read_size == -1)
			{
				if (checkStartToken(available_bytes))
				{
					//////////////////////////////////////////////////////////////////////////
					// read packet_ptr header
					if (available_bytes >= check_header_bytes)
					{
						if (!m_read_packet_ptr)
						{
							m_read_packet_ptr = po_new Packet();
						}

						RETURN_READ(&m_read_packet_ptr->header, PACKET_HEADER_BYTES);
						RETURN_READ(&m_read_packet_ptr->data_size, sizeof(u32));
						RETURN_READ(&m_read_packet_ptr->crc_code, sizeof(u16));

						m_read_size = m_read_packet_ptr->data_size;
						available_bytes -= check_header_bytes;
						
						//testpoint
						//if (m_read_packet_ptr->header.cmd != kPOCmdPing)
						//{
						//	printlog_lvs4(QString("Network command[%1] is accepted")
						//						.arg(m_read_packet_ptr->header.cmd), LOG_SCOPE_NET);
						//}

						if (m_read_size >= MAX_SOCK_BUFSIZE || m_read_size < 0) //exception case
						{
							prepareNewRead();
						}
						else if (m_read_size > 0) //packet has data
						{
							//new packet buffer
							m_read_packet_ptr->data_ptr = po_new u8[m_read_size];
						}
						else if (m_read_size == 0) //packet completed
						{
							m_read_packet_ptr->data_ptr = NULL;
							if (checkEndToken(m_read_packet_ptr))
							{
								onCompletedPacket();
							}
							else
							{
								onError(kConnectionErrPacket);
								prepareNewRead();
							}
						}
					}
					else
					{
						break; //out of while
					}
				}
				else
				{
					printlog_lvs2(QString("Network Token%1 incomplete.").arg(m_read_token_start), LOG_SCOPE_NET);
					break; //out of while
				}
			}
			else
			{
				//////////////////////////////////////////////////////////////////////////
				// read packet_ptr body data.
				i32 size_to_read = m_read_size - m_read_pos;
				if (size_to_read > 0)
				{
					RETURN_CHECK((u8*)(m_read_packet_ptr->data_ptr + m_read_pos), size_to_read);
					m_read_pos += n;
					available_bytes -= n;
				}

				if (m_read_size == m_read_pos)
				{
					// process buffer.
					if (checkEndToken(m_read_packet_ptr))
					{
						onCompletedPacket();
					}
					else
					{
						onError(kConnectionErrPacket);
						prepareNewRead();
					}
				}
			}
		}
	}
	catch (NL::Exception e)
	{
		printlog_lvs2("Network communication expection error", LOG_SCOPE_NET);
		prepareNewRead();
		return false;
	}

	m_last_read_time = sys_cur_time;
	return true;
}

void CConnection::onCompletedPacket()
{
	Packet* packet_ptr = m_read_packet_ptr;
	m_read_packet_ptr = NULL;

	prepareNewRead();

	if (m_conn_listener_ptr)
	{
		m_conn_listener_ptr->onRead(packet_ptr, this);
	}
}

void CConnection::sendPacket(Packet* packet_ptr, bool is_omitable, bool need_lock)
{
	if (!packet_ptr)
	{
		POSAFE_DELETE(packet_ptr);
		return;
	}

	//check response command can be ignore previous response
	singlelog_lvs4("Send command packet to queue", LOG_SCOPE_NET);

	if (need_lock)
	{
		m_packet_queue_mutex.lock();
	}

	//update packet queue map
	i32 cmd = packet_ptr->getCmd();
	i32 sub_cmd = packet_ptr->getSubCmd();
	PacketID pid(cmd, sub_cmd);

	if (is_omitable)
	{
		PacketIDMap::iterator iter;
		iter = m_queue_map.find(pid);
		if (iter != m_queue_map.end())
		{
			iter->second->setDirty();
			m_queue_map.erase(iter);
			m_queue_size--;
//			debug_log(QString("Remove Packet From Queue: %1").arg(cmd));
		}
	}

	//add packet_ptr to queue
	if (m_queue_size <= kPOServerMaxQueue)
	{
		//보내려는 명령대기렬이 kPOServerMaxQueue크기의 1.7배가 되면 명령대기렬에서 무의미한 파케트들을 삭제하고 정리한다.
		if (m_packet_queue.size() > kPOServerMaxQueue * 1.7)
		{
			PacketPList::iterator pak_iter;
			for (pak_iter = m_packet_queue.begin(); pak_iter != m_packet_queue.end();)
			{
				Packet* p = *pak_iter;
				if (p->isDirty())
				{
					pak_iter = m_packet_queue.erase(pak_iter);
					POSAFE_DELETE(p);
				}
				else
				{
					pak_iter++;
				}
			}
		}

		if (is_omitable)
		{
			//add packet to map
			m_queue_map[pid] = packet_ptr;
		}
		m_queue_size++;
		m_packet_queue.push_back(packet_ptr);

		if (need_lock)
		{
			m_packet_queue_mutex.unlock();
		}
		if (m_queue_size < kPOServerMaxQueue)
		{
			ERR_UNOCCUR(0);
		}
	}
	else
	{
		if (need_lock)
		{
			m_packet_queue_mutex.unlock();
		}
		POSAFE_DELETE(packet_ptr);
		ERR_OCCUR(0, debug_log(QString("TCP connection problem[%1]").arg(_err_rep0)));
	}
}

bool CConnection::onWrite()
{
	bool is_found = false;
	i32 skip_packet_count = 0;
	i32 write_packet_count = 0;
	Packet* packet_ptr = NULL;

	{
		m_write_queue_mutex.lock();
		if (m_write_queue.size() == 0) //if current writting finished...
		{
			POMutexLocker p(m_packet_queue_mutex);
			if (m_packet_queue.size() > 0) //if packet queue is not empty...
			{
				m_write_queue = m_packet_queue;
				m_packet_queue.clear();
				m_queue_map.clear();
				m_queue_size = 0;
			}
		}

		while (m_write_queue.size() > 0 && write_packet_count < kPOServerMaxWrite)
		{
			packet_ptr = m_write_queue.front();
			m_write_queue.pop_front();

			//check null-unused-command packet
			if (packet_ptr && !packet_ptr->isDirty())
			{
				bool is_success = false;
				write_packet_count++;

				//unlock
				i32 cmd = packet_ptr->getCmd();
				i32 sub_cmd = packet_ptr->getSubCmd();
				i32 data_len = packet_ptr->getDataLen();
				i64 reserved = 0; //1- show all packet
				m_write_queue_mutex.unlock();
				{
					//////////////////////////////////////////////////////////////////////////
					//testpoint
					//if (cmd == 567 || sub_cmd == 226)
					//{
					//	reserved = packet_ptr->getReservedi64(0);
					//}
					//////////////////////////////////////////////////////////////////////////

					//write connection with searched connection
					try
					{
						is_success = writePacket(packet_ptr);//must delete packet in write function
					}
					catch (NL::Exception e)
					{
						printlog_lv0(QString("Network Server Send Exception: code%1, %2").arg(e.code()).arg(e.what()));
						return false;
					}
				}
				m_write_queue_mutex.lock(); //relock

				if (is_success)
				{
					if (reserved != 0)
					{
						debug_log(QString("cmd send cmd:%1, sum_cmd:%2, len:%3, reserved:%4")
									.arg(cmd).arg(sub_cmd).arg(data_len).arg(reserved)); //log only
					}
				}
				else
				{
					//printlog_lvs4(QString("ReAdded to queue, cmd:%1").arg(packet_ptr->getCmd()), LOG_SCOPE_NET);
					m_write_queue.push_front(packet_ptr);
					break;
				}
				continue;
			}

			skip_packet_count++;
			POSAFE_DELETE(packet_ptr);
		}
		m_write_queue_mutex.unlock();
	}

	//output skipped packet count
	if (skip_packet_count)
	{
		printlog_lvs2(QString("cmd sent skip:%1").arg(skip_packet_count), LOG_SCOPE_NET);
	}
	return true;
}

void CConnection::writeLock()
{
	m_packet_queue_mutex.lock();
}

void CConnection::writeUnlock()
{
	m_packet_queue_mutex.unlock();
}

bool CConnection::writePacket(Packet* packet_ptr)
{
	if (!m_socket_ptr || !packet_ptr)
	{
		POSAFE_DELETE(packet_ptr);
		return true;
	}

	while (true)
	{
		while (m_write_size > 0)
		{
			i32 send_byte = m_socket_ptr->sendAsync(m_write_buffer_ptr + m_write_pos, m_write_size);
			if (send_byte == 0)
			{
				return false;
			}

			m_write_pos += send_byte;
			m_write_size -= send_byte;
		}

		if (!prepareNewWrite(packet_ptr)) 
		{
			//has nothing sending data, packet will be delete
			return true;
		}
	}
	printlog_lv0("Unexpected Network send error!!!");
	return false;
}

void CConnection::freeWriteBuffer(bool is_free_packet)
{
	//free write buffer
	m_write_mode = kPacketWritePrepare;
	m_write_size = 0;
	m_write_pos = 0;

	//free write packet, if it need
	if (is_free_packet)
	{
		POSAFE_DELETE(m_write_packet_ptr);
	}

	//free write buffer, if it need
	if (m_write_buffer_allocated)
	{
		POSAFE_DELETE_ARRAY(m_write_buffer_ptr);
	}
	m_write_buffer_ptr = NULL;
	m_write_buffer_allocated = true;
}

bool CConnection::checkStartToken(i32& available_bytes)
{
	i32 n;

	if (m_read_token_start == -1)
	{
		if (available_bytes < 4)
		{
			return false;
		}

		RETURN_CHECK(&m_read_token_start, 4);
		available_bytes -= n;
		if (n != 4)
		{
			return false;
		}
	}
	else
	{
		char* p = (char*)&m_read_token_start;
		while (available_bytes > 0 && m_read_token_start != POTOKEN_START)
		{
			printlog_lvs2("Network command Token search", LOG_SCOPE_NET);

			m_read_token_start = m_read_token_start >> 8;
			RETURN_CHECK(p + 3, 1);
			available_bytes -= n;
			if (n != 1)
			{
				return false;
			}
		}
	}
	return m_read_token_start == POTOKEN_START;
}

bool CConnection::checkEndToken(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return false;
	}

	u16 crc_code = packet_ptr->crc_code;
	return (crc_code == packet_ptr->makeCRCCode());
}

void CConnection::prepareNewRead()
{
	m_read_token_start = -1;
	m_read_token_end = -1;
	m_read_pos = 0;
	m_read_size = -1;

	POSAFE_DELETE(m_read_packet_ptr);
}

bool CConnection::prepareNewWrite(Packet* packet_ptr)
{
	if (m_write_mode == kPacketWritePrepare)
	{
		freeWriteBuffer(true);

		m_write_mode = kPacketWriteHeader;
		m_write_packet_ptr = packet_ptr;

		i32 token = POTOKEN_START;
		packet_ptr->makeCRCCode();

		m_write_size = sizeof(token) + Packet::calcHeaderSize();
		m_write_buffer_ptr = po_new u8[m_write_size];
		m_write_buffer_allocated = true;
		
		u8* buffer_ptr = m_write_buffer_ptr;
		i32 buffer_size = m_write_size;
		CPOBase::memWrite(token, buffer_ptr, buffer_size);
		CPOBase::memWrite(packet_ptr->header, buffer_ptr, buffer_size);
		CPOBase::memWrite(packet_ptr->data_size, buffer_ptr, buffer_size);
		CPOBase::memWrite(packet_ptr->crc_code, buffer_ptr, buffer_size);
	}
	else if (m_write_mode == kPacketWriteHeader)
	{
		freeWriteBuffer(false);

		m_write_mode = kPacketWriteBody;
		m_write_buffer_allocated = false;
		m_write_size = m_write_packet_ptr->getDataLen();
		m_write_buffer_ptr = m_write_packet_ptr->getData();
	}
	else if (m_write_mode == kPacketWriteBody)
	{
		bool is_same_packet = (m_write_packet_ptr == packet_ptr);
		freeWriteBuffer(true);
		if (is_same_packet)
		{
			return false;
		}
	}
	else
	{
		freeWriteBuffer(true);
		return false;
	}
	return true;
}

Socket* CConnection::getSocket()
{
	return m_socket_ptr;
}

void CConnection::setSocket(Socket* socket_ptr)
{
	m_socket_ptr = socket_ptr;
	if (socket_ptr)
	{
		socket_ptr->setData((void*)this);
		socket_ptr->blocking(false);
		socket_ptr->tcpNoDelay(true);
		socket_ptr->setOption(64000, -1); //send buffer: 64k
	}
}

void CConnection::setListener(CConnectionListener* pListener)
{
	m_conn_listener_ptr = pListener;
}

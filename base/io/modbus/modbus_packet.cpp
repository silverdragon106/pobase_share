#include "modbus_packet.h"

CModBusPacket::CModBusPacket()
{
	m_dev_addr = 0;
	m_cmd = MODBUS_NONE;
	m_data_u = NULL;
	m_crc_code = NULL;

	m_len = 0;
	m_buffer_ptr = NULL;
	m_is_valid = false;
}

CModBusPacket::CModBusPacket(bool is_master, u8 mbdevaddr, u8 mbcmd, u8 data_len)
{
	m_dev_addr = 0;
	m_cmd = MODBUS_NONE;
	m_data_u = NULL;
	m_crc_code = NULL;

	m_len = 0;
	m_buffer_ptr = NULL;
	m_is_valid = false;

	if (is_master)
	{
		makeRequestPacket(mbdevaddr, mbcmd, data_len);
	}
	else
	{
		makeResponsePacket(mbdevaddr, mbcmd, data_len);
	}
}

CModBusPacket::~CModBusPacket()
{
	POSAFE_DELETE_ARRAY(m_buffer_ptr);
}

void CModBusPacket::makeResponsePacket(u8 mbdevaddr, u8 mbcmd, u8 data_len)
{
	switch (mbcmd)
	{
		case MODBUS_READCOIL:
		{
			m_len = sizeof(DataPacket::ReadCoilResp) + data_len + 3;
			break;
		}
		case MODBUS_READHOLD:
		{
			m_len = sizeof(DataPacket::ReadHoldResp) + data_len + 3;
			break;
		}
		case MODBUS_READREG:
		{
			m_len = sizeof(DataPacket::ReadRegResp) + data_len + 3;
			break;
		}
		case MODBUS_WRITEREG:
		{
			m_len = sizeof(DataPacket::WriteRegResp) + 4;
			break;
		}
		case MODBUS_MULWRITEREG:
		{
			m_len = sizeof(DataPacket::MulWriteCoilResp) + 4;
			break;
		}
		default:
		{
			if (mbcmd > MODBUS_ERROR)
			{
				m_len = sizeof(DataPacket::ErrorResp) + 4;
				break;
			}
			return;
		}
	}

	POSAFE_DELETE_ARRAY(m_buffer_ptr);
	m_buffer_ptr = po_new u8[m_len];
	memset(m_buffer_ptr, 0, m_len);

	m_buffer_ptr[0] = mbdevaddr; m_dev_addr = mbdevaddr;
	m_buffer_ptr[1] = mbcmd; m_cmd = mbcmd;
	m_data_u = (DataPacket*)(m_buffer_ptr + 2);
	m_crc_code = (u16*)(m_buffer_ptr + m_len - 2);
	m_is_valid = true;
}

void CModBusPacket::makeRequestPacket(u8 mbdevaddr, u8 mbcmd, u8 data_len)
{
	switch (mbcmd)
	{
		case MODBUS_READCOIL:
		{
			m_len = sizeof(DataPacket::ReadCoilReq) + 4;
			break;
		}
		case MODBUS_WRITECOIL:
		{
			m_len = sizeof(DataPacket::WriteCoilReq) + 4;
			break;
		}
		case MODBUS_MULWRITECOIL:
		{
			m_len = sizeof(DataPacket::MulWriteCoilReq) + data_len + 3;
			break;
		}
		default:
		{
			return;
		}
	}

	POSAFE_DELETE_ARRAY(m_buffer_ptr);
	m_buffer_ptr = po_new u8[m_len];
	memset(m_buffer_ptr, 0, m_len);

	m_buffer_ptr[0] = mbdevaddr; m_dev_addr = mbdevaddr;
	m_buffer_ptr[1] = mbcmd; m_cmd = mbcmd;
	m_data_u = (DataPacket*)(m_buffer_ptr + 2);
	m_crc_code = (u16*)(m_buffer_ptr + m_len - 2);
	m_is_valid = true;
}

void CModBusPacket::makeCRCCode()
{
	*m_crc_code = makeCRCCode(m_buffer_ptr, m_len-2);
}

u16 CModBusPacket::makeCRCCode(u8* buffer_ptr, u16 len)
{
	u16 crc_code = 0xFFFF;
	u16 crc_const_param = 0xA001;
	u16 data_word;
	u16 byte_index;
	i32 loop = 8;

	if (!buffer_ptr)
	{
		return crc_code;
	}

	for (byte_index = 0; byte_index < len; byte_index++)
	{
		data_word = buffer_ptr[byte_index];
		crc_code ^= data_word;
		loop = 8;
		while (loop-- != 0)
		{
			if ((crc_code & 0x01) == 0)
			{
				crc_code >>= 1;
			}
			else
			{
				crc_code >>= 1;
				crc_code ^= crc_const_param;
			}
		}
	}
	return crc_code;
}

bool CModBusPacket::isModBusPacket(bool is_master, i32 devaddr, u8* buffer_ptr, i32 memlen)
{
	if (!buffer_ptr || memlen < 2)
	{
		return false;
	}

	if (buffer_ptr[0] != devaddr)
	{
		m_is_valid = false;
		return true;
	}

	packetFromMem(is_master, buffer_ptr, memlen);
	return (m_len > 0);
}

void CModBusPacket::packetFromMem(bool is_master, u8* buffer_ptr, i32 memlen)
{
	m_dev_addr = buffer_ptr[0];
	m_cmd = buffer_ptr[1];

	//build packet_ptr body
	if (is_master)
	{
		responsePacketFromMem(buffer_ptr);
	}
	else
	{
		requestPacketFromMem(buffer_ptr);
	}

	//check packet_ptr size and crc code
	if (m_len > memlen) { m_len = 0; return; }
	if (m_len < 4) { m_is_valid = false; return; }
	m_is_valid = (makeCRCCode(buffer_ptr, m_len-2) == *m_crc_code);
}

void CModBusPacket::requestPacketFromMem(u8* buffer_ptr)
{
	m_data_u = (DataPacket*)(buffer_ptr + 2);
	switch (m_cmd)
	{
		case MODBUS_READCOIL:
		case MODBUS_READHOLD:
		case MODBUS_READREG:
		case MODBUS_WRITEREG:
		{
			m_crc_code = (u16*)(buffer_ptr + 6);
			m_len = 8;
			break;
		}
		case MODBUS_MULWRITEREG:
		{
			m_crc_code = (u16*)(buffer_ptr + 7 + m_data_u->mulwregreq.bytes);
			m_len = 9 + m_data_u->mulwregreq.bytes;
			break;
		}
		default:
		{
			m_len = 1;
			break;
		}
	}
}

void CModBusPacket::responsePacketFromMem(u8* buffer_ptr)
{
	m_data_u = (DataPacket*)(buffer_ptr + 2);
	switch (m_cmd)
	{
		case MODBUS_READCOIL:
		{
			m_crc_code = (u16*)(buffer_ptr + 3 + m_data_u->rcoilresp.bytes);
			m_len = m_data_u->rcoilresp.bytes + 5;
			break;
		}
		case MODBUS_WRITECOIL:
		case MODBUS_MULWRITECOIL:
		{
			m_crc_code = (u16*)(buffer_ptr + 6);
			m_len = 8;
			break;
		}
		default:
		{
			if (m_cmd > MODBUS_ERROR)
			{
				m_crc_code = (u16*)(buffer_ptr + 3);
				m_len = 5;
			}
			else
			{
				m_len = 1;
			}
			break;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
CModNetPacket::CModNetPacket()
{
	m_header_ptr = NULL;
	m_cmd = MODBUS_NONE;
	m_data_u = NULL;

	m_len = 0;
	m_buffer_ptr = NULL;
	m_is_valid = false;
}

CModNetPacket::CModNetPacket(bool bclient, MBAPHeader* pheader, u8 mbcmd, u8 data_len)
{
	m_header_ptr = NULL;
	m_cmd = MODBUS_NONE;
	m_data_u = NULL;

	m_len = 0;
	m_buffer_ptr = NULL;
	m_is_valid = false;

	if (bclient)
	{
		makeRequestPacket(pheader, mbcmd, data_len);
	}
	else
	{
		makeResponsePacket(pheader, mbcmd, data_len);
	}
}

CModNetPacket::~CModNetPacket()
{
	POSAFE_DELETE_ARRAY(m_buffer_ptr);
}

void CModNetPacket::makeRequestPacket(MBAPHeader* pheader, u8 mbcmd, u8 data_len)
{
	i32 mbap_len = sizeof(MBAPHeader);
	switch (mbcmd)
	{
		case MODBUS_READCOIL:
		{
			m_len = (u16)sizeof(DataPacket::ReadCoilReq) + mbap_len + 1;
			break;
		}
		case MODBUS_READHOLD:
		{
			m_len = (u16)sizeof(DataPacket::ReadHoldReq) + mbap_len + 1;
			break;
		}
		case MODBUS_WRITECOIL:
		{
			m_len = (u16)sizeof(DataPacket::WriteCoilReq) + mbap_len + 1;
			break;
		}
		case MODBUS_WRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::WriteRegReq) + mbap_len + 1;
			break;
		}
		case MODBUS_MULWRITECOIL:
		{
			m_len = (u16)sizeof(DataPacket::MulWriteCoilReq) + mbap_len + data_len;
			break;
		}
		default:
		{
			return;
		}
	}

	POSAFE_DELETE_ARRAY(m_buffer_ptr);
	m_buffer_ptr = po_new u8[m_len];
	memset(m_buffer_ptr, 0, m_len);

	m_header_ptr = (MBAPHeader*)m_buffer_ptr;
	m_header_ptr->pid = MODNET_PROTOCAL;
	m_header_ptr->tid = pheader->tid;
	m_header_ptr->uid = pheader->uid;
	m_header_ptr->len = po::swap_endian16(m_len - mbap_len + 1);

	m_buffer_ptr[mbap_len] = mbcmd;
	m_cmd = mbcmd;
	m_data_u = (DataPacket*)(m_buffer_ptr + mbap_len + 1);
	m_is_valid = true;
}

void CModNetPacket::makeResponsePacket(MBAPHeader* pheader, u8 mbcmd, u8 data_len)
{
	i32 mbap_len = sizeof(MBAPHeader);
	switch (mbcmd)
	{
		case MODBUS_READCOIL:
		{
			m_len = (u16)sizeof(DataPacket::ReadCoilResp) + mbap_len + data_len;
			break;
		}
		case MODBUS_READHOLD:
		{
			m_len = (u16)sizeof(DataPacket::ReadHoldResp) + mbap_len + data_len;
			break;
		}
		case MODBUS_READREG:
		{
			m_len = (u16)sizeof(DataPacket::ReadRegResp) + mbap_len + data_len;
			break;
		}
		case MODBUS_WRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::WriteRegResp) + mbap_len + 1;
			break;
		}
		case MODBUS_MULWRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::MulWriteCoilResp) + mbap_len + 1;
			break;
		}
		default:
		{
			if (mbcmd > MODBUS_ERROR)
			{
				m_len = (u16)sizeof(DataPacket::ErrorResp) + mbap_len + 1;
				break;
			}
			return;
		}
	}

	POSAFE_DELETE_ARRAY(m_buffer_ptr);
	m_buffer_ptr = po_new u8[m_len];
	memset(m_buffer_ptr, 0, m_len);

	m_header_ptr = (MBAPHeader*)m_buffer_ptr;
	m_header_ptr->pid = MODNET_PROTOCAL;
	m_header_ptr->tid = pheader->tid;
	m_header_ptr->uid = pheader->uid;
	m_header_ptr->len = po::swap_endian16(m_len - mbap_len + 1);

	m_buffer_ptr[mbap_len] = mbcmd;
	m_cmd = mbcmd;
	m_data_u = (DataPacket*)(m_buffer_ptr + mbap_len + 1);
	m_is_valid = true;
}

bool CModNetPacket::isModNetPacket(bool is_client, u8* buffer_ptr, i32 memlen)
{
	i32 mbap_len = sizeof(MBAPHeader);
	if (!buffer_ptr || memlen <= mbap_len)
	{
		return false;
	}

	m_header_ptr = (MBAPHeader*)buffer_ptr;
	if (po::swap_endian16(m_header_ptr->pid) != MODNET_PROTOCAL)
	{
		m_is_valid = false;
		return true; 
	}
	if (memlen < mbap_len + po::swap_endian16(m_header_ptr->len) - 1)
	{
		m_len = 0;
		return false; 
	}

	packetFromMem(is_client, buffer_ptr, memlen);
	return (m_len > 0);
}

void CModNetPacket::packetFromMem(bool is_client, u8* buffer_ptr, i32 memlen)
{
	//build packet_ptr body
	if (is_client)
	{
		responsePacketFromMem(buffer_ptr);
	}
	else
	{
		requestPacketFromMem(buffer_ptr);
	}

	//check packet_ptr size and crc code
	if (m_len > memlen) { m_len = 0; return; }
	if (m_len < 4) { m_is_valid = false; return; }
	m_is_valid = true;
}

void CModNetPacket::requestPacketFromMem(u8* buffer_ptr)
{
	i32 mbap_len = sizeof(MBAPHeader);
	m_cmd = buffer_ptr[mbap_len];
	m_data_u = (DataPacket*)(buffer_ptr + mbap_len + 1);

	switch (m_cmd)
	{
		case MODBUS_READCOIL:
		case MODBUS_READHOLD:
		case MODBUS_READREG:
		case MODBUS_WRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::ReadCoilReq) + mbap_len + 1;
			break;
		}
		case MODBUS_MULWRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::MulWriteRegReq) + mbap_len + m_data_u->mulwregreq.bytes;
			break;
		}
		default:
		{
			m_len = 0;
			break;
		}
	}
}

void CModNetPacket::responsePacketFromMem(u8* buffer_ptr)
{
	i32 mbap_len = sizeof(MBAPHeader);
	m_cmd = buffer_ptr[mbap_len];
	m_data_u = (DataPacket*)(buffer_ptr + mbap_len + 1);

	switch (m_cmd)
	{
		case MODBUS_READCOIL:
		{
			m_len = (u16)sizeof(DataPacket::ReadCoilResp) + mbap_len + m_data_u->rcoilresp.bytes;
			break;
		}
		case MODBUS_READHOLD:
		{
			m_len = (u16)sizeof(DataPacket::ReadHoldResp) + mbap_len + m_data_u->rholdresp.bytes;
			break;
		}
		case MODBUS_WRITEREG:
		{
			m_len = (u16)sizeof(DataPacket::WriteRegResp) + mbap_len + 1;
			break;
		}
		case MODBUS_WRITECOIL:
		{
			m_len = (u16)sizeof(DataPacket::WriteCoilResp) + mbap_len + 1;
			break;
		}
		case MODBUS_MULWRITECOIL:
		{
			m_len = (u16)sizeof(DataPacket::MulWriteCoilResp) + mbap_len + 1;
			break;
		}
		default:
		{
			if (m_cmd > MODBUS_ERROR)
			{
				m_len = (u16)sizeof(DataPacket::ErrorResp) + mbap_len + 1;
			}
			else
			{
				m_len = 0;
			}
			break;
		}
	}
}

CModNetPacket* CModNetPacket::clone()
{
	if (m_len <= 0 || !m_is_valid)
	{
		return NULL;
	}

	CModNetPacket* pak = po_new CModNetPacket();
	if (m_buffer_ptr)
	{
		pak->m_buffer_ptr = po_new u8[m_len];
		memcpy(pak->m_buffer_ptr, m_buffer_ptr, m_len);
	}
	else
	{
		pak->m_buffer_ptr = po_new u8[m_len];
		memcpy(pak->m_buffer_ptr, (const char*)m_header_ptr, m_len);
	}

	pak->m_cmd = m_cmd;
	pak->m_len = m_len;
	pak->m_is_valid = m_is_valid;
	pak->m_header_ptr = (MBAPHeader*)pak->m_buffer_ptr;
	pak->m_data_u = (DataPacket*)(pak->m_buffer_ptr + sizeof(MBAPHeader) + 1);
	return pak;
}
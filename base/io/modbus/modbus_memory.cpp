#include "modbus_memory.h"
#include "base.h"

//////////////////////////////////////////////////////////////////////////
CMBMemory::CMBMemory()
{
	m_is_inited = false;

	m_reg_start = 0;
	m_reg_end = 0;
	m_reg_cmd_count = 0;
	m_coil_start = 0;
	m_coil_end = 0;
	m_hold_start = 0;
	m_hold_end = 0;

	m_reg_addr_ptr = NULL;
	m_reg_section_ptr = NULL;

	m_buffer_ptr = NULL;
	m_register_ptr = NULL;
	m_max_address = 0xFFFF;
}

CMBMemory::~CMBMemory()
{
	freeBuffer();
}

bool CMBMemory::initBuffer(i32 size, u16* section_addr_ptr, u16 section_count)
{
	if (size <= 0)
	{
		printlog_lv1(QString("Can't initialize modbus memory with size[%1].").arg(size));
		return false;
	}

	if (m_is_inited)
	{
		return true;
	}

	freeBuffer();
	m_is_inited = true;
	m_max_address = size;
	
	if (m_reg_start > m_reg_end || m_coil_start > m_coil_end || m_hold_start > m_hold_end || 
		section_count > m_reg_end - m_reg_start + 1)
	{
		printlog_lvs2("MBMemory initBuffer failed.(invalid section)", LOG_SCOPE_COMM);
		return false;
	}
	if (m_reg_end > m_max_address || m_coil_end > m_max_address || m_hold_end > m_max_address)
	{
		printlog_lvs2("MBMemory initBuffer failed.(oversection)", LOG_SCOPE_COMM);
		return false;
	}
	if (!CPOBase::checkRangeOverlap(m_reg_start, m_reg_end, m_coil_start, m_coil_end) ||
		!CPOBase::checkRangeOverlap(m_reg_start, m_reg_end, m_hold_start, m_hold_end) ||
		!CPOBase::checkRangeOverlap(m_coil_start, m_coil_end, m_hold_start, m_hold_end))
	{
		printlog_lvs2("MBMemory initBuffer failed.(overlaped section)", LOG_SCOPE_COMM);
		return false;
	}

	//update cmd register section
	if (section_addr_ptr && section_count > 0)
	{
		m_reg_cmd_count = section_count;
		m_reg_addr_ptr = section_addr_ptr;
		m_reg_section_ptr = po_new bool[section_count + 1];
		memset(m_reg_section_ptr, 0, section_count);

		i32 i, ni;
		for (i = 0; i < section_count; i++)
		{
			ni = i + 1;
			if (ni < section_count && section_addr_ptr[i] == section_addr_ptr[ni])
			{
				continue;
			}
			m_reg_section_ptr[i] = true;
		}
	}
	
	m_register_ptr = po_new u16[size];
	m_buffer_ptr = m_register_ptr;
	memset(m_register_ptr, 0, size*sizeof(u16));
	return true;
}

void CMBMemory::freeBuffer()
{
	if (m_is_inited)
	{
		m_reg_addr_ptr = NULL;
		m_buffer_ptr = NULL;
		POSAFE_DELETE_ARRAY(m_reg_section_ptr);
		POSAFE_DELETE_ARRAY(m_register_ptr);
		
		m_is_inited = false;
		m_max_address = 0;
		m_reg_start = 0; m_reg_end = 0;
		m_coil_start = 0; m_coil_end = 0;
		m_hold_start = 0; m_hold_end = 0;
	}
}

void CMBMemory::setRegRange(i32 st, i32 ed)
{
	m_reg_start = st;
	m_reg_end = ed;
}

void CMBMemory::setCoilRange(i32 st, i32 ed)
{
	m_coil_start = st;
	m_coil_end = ed;
}

void CMBMemory::setHoldRange(i32 st, i32 ed)
{
	m_hold_start = st;
	m_hold_end = ed;
}

bool CMBMemory::checkRegCmdAddress(i32 pos)
{
	return CPOBase::checkIndex(pos, m_reg_cmd_count);
}

bool CMBMemory::checkRegCmdSection(i32 pos)
{
	if (CPOBase::checkIndex(pos, m_reg_cmd_count))
	{
		return m_reg_section_ptr ? m_reg_section_ptr[pos] : false;
	}
	return false;
}

bool CMBMemory::checkRegAddress(i32 addr, i32 count)
{
	if (count < 0)
	{
		return false;
	}
	if (!CPOBase::checkRange(addr, m_reg_start, m_reg_end))
	{
		return false;
	}
	return CPOBase::checkRange(addr + count - 1, m_reg_start, m_reg_end);
}

bool CMBMemory::checkCoilAddress(i32 addr, i32 count)
{
	if (count < 0)
	{
		return false;
	}
	if (!CPOBase::checkRange(addr, m_coil_start, m_coil_end))
	{
		return false;
	}
	return CPOBase::checkRange(addr + count - 1, m_coil_start, m_coil_end);
}

bool CMBMemory::checkHoldAddress(i32 addr, i32 count)
{
	if (count < 0)
	{
		return false;
	}
	if (!CPOBase::checkRange(addr, m_hold_start, m_hold_end))
	{
		return false;
	}
	return CPOBase::checkRange(addr + count - 1, m_hold_start, m_hold_end);
}

bool CMBMemory::seekAddress(i32 addr)
{
	u8* tmp_buffer_ptr;
	return seekAddress(addr, 0, tmp_buffer_ptr);
}

bool CMBMemory::seekAddress(i32 addr, i32 count, u8*& buffer_ptr)
{
	u16* register_ptr = NULL;
	buffer_ptr = NULL;

	if (!m_is_inited)
	{
		return false;
	}
		
	if (addr < 0)
	{
		if (m_buffer_ptr + count - m_register_ptr > m_max_address)
		{
			return false;
		}
		register_ptr = m_buffer_ptr;
	}
	else
	{
		if (addr + count > m_max_address)
		{
			return false;
		}
		register_ptr = m_register_ptr + addr;
	}

	buffer_ptr = (u8*)register_ptr;
	m_buffer_ptr = register_ptr + count;
	return true;
}

bool CMBMemory::getValueu8(u8& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}
	
	u16 tmp = 0;
	CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	value = (u8)tmp;
	return true;
}

bool CMBMemory::getValueu16(u16& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}
	
	u16 tmp = 0;
	CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		value = po::swap_endian16(tmp);
	}
	value = tmp;
	return true;
}

bool CMBMemory::getValueu32(u32& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp = 0;
	CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		value = po::swap_endian32(tmp);
	}
	value = tmp;
	return true;
}

bool CMBMemory::getValueu64(u64& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u64 tmp = 0;
	CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		value = po::swap_endian64(tmp);
	}
	value = tmp;
	return true;
}

bool CMBMemory::getValuei8(i8& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	value = (i8)*((i16*)&tmp);
	return true;
}

bool CMBMemory::getValuei16(i16& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	value = *((i16*)&tmp);
	return true;
}

bool CMBMemory::getValuei32(i32& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian32(tmp);
	}
	value = *((i32*)&tmp);
	return true;
}

bool CMBMemory::getValuei64(i64& value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u64 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian64(tmp);
	}
	value = *((i64*)&tmp);
	return true;
}

bool CMBMemory::getValuef32(f32& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian64(tmp);
	}

	if (fmt_number < 0)
	{
		value = *((f32*)&tmp);
		return true;
	}
	else if (fmt_number > 10)
	{
		return false;
	}
	value = *((i32*)&tmp)/ CPOBase::fastPow10(fmt_number);
	return true;
}

bool CMBMemory::getValuef64(f64& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u64 tmp = 0;
    CPOBase::memRead(tmp, buffer_ptr);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian64(tmp);
	}
	if (fmt_number < 0)
	{
		value = *((f64*)&tmp);
		return true;
	}
	else if (fmt_number > 10)
	{
		return false;
	}

	value = *((i64*)&tmp) / CPOBase::fastPow10(fmt_number);
	return true;
}

bool CMBMemory::getValueVector2df(vector2df& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	if (!getValuef32(value.x, endian_mode, fmt_number, addr))
	{
		return false;
	}
	return getValuef32(value.y, endian_mode, fmt_number);
}

bool CMBMemory::getValueRectf(Rectf& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	if (!getValuef32(value.x1, endian_mode, fmt_number, addr))
	{
		return false;
	}
	if (!getValuef32(value.y1, endian_mode, fmt_number))
	{
		return false;
	}
	if (!getValuef32(value.x2, endian_mode, fmt_number))
	{
		return false;
	}
	return getValuef32(value.y2, endian_mode, fmt_number);
}

bool CMBMemory::setValueu8(u8 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = value;
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValueu16(u16 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = value;
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValueu32(u32 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp = value;
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian32(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValueu64(u64 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u64 tmp = value;
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian32(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuei8(i8 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = *((u8*)&value);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuei16(i16 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 1, buffer_ptr))
	{
		return false;
	}

	u16 tmp = *((u16*)&value);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian16(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuei32(i32 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp = *((u32*)&value);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian32(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuei64(i64 value, i32 endian_mode, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u64 tmp = *((u64*)&value);
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian64(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuef32(f32 value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 2, buffer_ptr))
	{
		return false;
	}

	u32 tmp;
	if (fmt_number < 0)
	{
		tmp = *((u32*)&value);
	}
	else if (fmt_number > 10)
	{
		return false;
	}
	else
	{
		i32 itmp = value*CPOBase::fastPow10(fmt_number);
		tmp = *((u32*)&itmp);
	}
	
	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian32(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValuef64(f64 value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	u8* buffer_ptr;
	if (!seekAddress(addr, 4, buffer_ptr))
	{
		return false;
	}

	u32 tmp;
	if (fmt_number < 0)
	{
		tmp = *((u64*)&value);
	}
	else if (fmt_number > 10)
	{
		return false;
	}
	else
	{
		i64 itmp = value*CPOBase::fastPow10(fmt_number);
		tmp = *((u64*)&itmp);
	}

	if (endian_mode == kPOModbusBigEndian)
	{
		tmp = po::swap_endian64(tmp);
	}
	CPOBase::memWrite(tmp, buffer_ptr);
	return true;
}

bool CMBMemory::setValueVector2df(vector2df& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	if (!setValuef32(value.x, endian_mode, fmt_number, addr))
	{
		return false;
	}
	return setValuef32(value.y, endian_mode, fmt_number);
}

bool CMBMemory::setValueRectf(Rectf& value, i32 endian_mode, i32 fmt_number, i32 addr)
{
	if (!setValuef32(value.x1, endian_mode, fmt_number, addr))
	{
		return false;
	}
	if (!setValuef32(value.y1, endian_mode, fmt_number))
	{
		return false;
	}
	if (!setValuef32(value.x2, endian_mode, fmt_number))
	{
		return false;
	}
	return setValuef32(value.y2, endian_mode, fmt_number);
}

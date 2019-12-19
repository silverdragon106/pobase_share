#include "modbus_serial.h"
#include "base.h"

CModBusSerial::CModBusSerial()
{
	m_is_inited = false;
}

CModBusSerial::~CModBusSerial()
{
}

bool CModBusSerial::initInstance(CMBMemory* modbus_mem_ptr, CModbusDevParam* modbus_param_ptr)
{
	if (!modbus_mem_ptr || !modbus_param_ptr)
	{
		return false;
	}

	if (!m_is_inited)
	{
		singlelog_lv0("RS232/485 Modbus InitInstance");

		m_modbus_mem_ptr = modbus_mem_ptr;
		CModBusCom::initInstance(modbus_param_ptr, false);
		m_is_inited = true;
	}
	return true;
}

void CModBusSerial::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0("RS232/485 Modbus ExitInstance");

		m_is_inited = false;
		CModBusCom::exitInstance();
		m_modbus_mem_ptr = NULL;
	}
}

void CModBusSerial::onReadPacket(CModBusPacket* pak)
{
	printlog_lvs4(QString("RS485 Modbus packet_ptr is accepted, %1").arg(pak->m_cmd), LOG_SCOPE_COMM);

#if defined(POR_WITH_IOMODULE)
	i32 quan, addr, tmp;
	i32 errtype = kMBSuccess;

	switch (pak->m_cmd)
	{
		case MODBUS_READCOIL://read built-in output and DO result [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rcoilreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rcoilreq.quan);
			if (quan < 1 || quan > MODBUS_MAXRCOILQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getCoilStart() || addr + quan > m_modbus_mem_ptr->getCoilEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestReadCoil(addr, quan);
			}
			break;
		}
		case MODBUS_READHOLD://read output register value [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rholdreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rholdreq.quan);
			if (quan < 1 || quan > MODBUS_MAXRREGQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getHoldStart() || addr + quan > m_modbus_mem_ptr->getHoldEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestReadHoldRegister(addr, quan);
			}
			break;
		}
		case MODBUS_READREG://read input register value writed already[multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rregreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rregreq.quan);
			if (quan < 1 || quan > MODBUS_MAXRREGQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getRegStart() || addr + quan > m_modbus_mem_ptr->getRegEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestReadRegister(addr, quan);
			}
			break;
		}
		case MODBUS_WRITEREG://write input register [solo]
		{
			addr = po::swap_endian16(pak->m_data_u->wregreq.addr);
			tmp = po::swap_endian16(pak->m_data_u->wregreq.val);
			if (addr < m_modbus_mem_ptr->getRegStart() || addr > m_modbus_mem_ptr->getRegEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestWriteRegister(addr, tmp);
			}
			break;
		}
		case MODBUS_MULWRITEREG://write input register [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->mulwregreq.addr);
			quan = po::swap_endian16(pak->m_data_u->mulwregreq.quan);
			tmp = pak->m_data_u->mulwregreq.bytes;
			if (quan < 1 || quan > MODBUS_MAXWREGQUAN || tmp != 2 * quan)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getRegStart() || addr + quan > m_modbus_mem_ptr->getRegEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				//copy data to register buffer
				{
					anlock_guard_ptr(m_modbus_mem_ptr);
					u16* register_ptr = m_modbus_mem_ptr->getRegister();
					if (register_ptr)
					{
						memcpy(register_ptr + addr, &(pak->m_data_u->mulwregreq.buffer), tmp);
					}
				}
				errtype = onRequestMulWriteRegister(addr, quan);
			}
			break;
		}
		default:
		{
			errtype = kMBErrFuncNoSupport;
			break;
		}
	}

	if (errtype != kMBSuccess)
	{
		//write error response
		writeErrResponse(pak, errtype);
	}
#endif
}

i32 CModBusSerial::onRequestReadCoil(i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr || !m_modbus_param_ptr)
	{
		return kMBErrProcess;
	}

#if defined(POR_WITH_IOMODULE)
	i32 i, bytes = (quan + 7) / 8;
	CModBusPacket pak(m_is_master, m_modbus_param_ptr->getDevAddress(), MODBUS_READCOIL, bytes);

	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			register_ptr += addr;
			u8* buffer_ptr = &(pak.m_data_u->rcoilresp.buffer);
			for (i = 0; i < quan; i++)
			{
				buffer_ptr[i / 8] += (register_ptr[i] ? 1 : 0) << (i % 8);
			}
		}
		m_modbus_mem_ptr->unlock();
	}

	pak.m_data_u->rcoilresp.bytes = bytes;
	writeCmdToDevice(&pak);
#endif
	return kMBSuccess;
}

i32 CModBusSerial::onRequestReadRegister(i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr || !m_modbus_param_ptr)
	{
		return kMBErrProcess;
	}

#if defined(POR_WITH_IOMODULE)
	i32 bytes = quan * 2;
	CModBusPacket pak(m_is_master, m_modbus_param_ptr->getDevAddress(), MODBUS_READREG, bytes);
	
	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			memcpy(&(pak.m_data_u->rregresp.buffer), (u8*)(register_ptr + addr), bytes);
		}
		m_modbus_mem_ptr->unlock();
	}
	pak.m_data_u->rregresp.bytes = bytes;
	writeCmdToDevice(&pak);
#endif
	return kMBSuccess;
}

i32 CModBusSerial::onRequestReadHoldRegister(i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr || !m_modbus_param_ptr)
	{
		return kMBErrProcess;
	}

#if defined(POR_WITH_IOMODULE)
	i32 bytes = quan * 2;
	CModBusPacket pak(m_is_master, m_modbus_param_ptr->getDevAddress(), MODBUS_READHOLD, bytes);

	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			memcpy(&(pak.m_data_u->rholdresp.buffer), (u8*)(register_ptr + addr), bytes);
		}
		m_modbus_mem_ptr->unlock();
	}
	pak.m_data_u->rholdresp.bytes = bytes;
	writeCmdToDevice(&pak);
#endif
	return kMBSuccess;
}

i32 CModBusSerial::onRequestWriteRegister(i32 addr, i32 value)
{
	if (!m_modbus_mem_ptr || !m_modbus_param_ptr)
	{
		return kMBErrProcess;
	}

#if defined(POR_WITH_IOMODULE)
	i32 tmp = po::swap_endian16(value); //lendian -> bendian
	CModBusPacket pak(m_is_master, m_modbus_param_ptr->getDevAddress(), MODBUS_WRITEREG);
	pak.m_data_u->wregresp.addr = po::swap_endian16(addr);
	pak.m_data_u->wregresp.val = tmp;

	{
		anlock_guard_ptr(m_modbus_mem_ptr);
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			register_ptr[addr] = tmp;
		}
	}
	writeCmdToDevice(&pak);
#endif

	i32 pos = addr - m_modbus_mem_ptr->getRegStart();
	if (m_modbus_mem_ptr->checkRegCmdAddress(pos))
	{
		onRequestCommand(pos);
	}
	else if (m_modbus_mem_ptr->checkRegAddress(addr))
	{
		onRequestRegChange(addr, 1);
	}
	return kMBSuccess;
}

i32 CModBusSerial::onRequestMulWriteRegister(i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr || !m_modbus_param_ptr)
	{
		return kMBErrProcess;
	}

#if defined(POR_WITH_IOMODULE)
	CModBusPacket pak(m_is_master, m_modbus_param_ptr->getDevAddress(), MODBUS_MULWRITEREG);
	pak.m_data_u->mulwregresp.addr = po::swap_endian16(addr);
	pak.m_data_u->mulwregresp.quan = po::swap_endian16(quan);
	writeCmdToDevice(&pak);
#endif

	i32 i, pos = 0;
	i32 reg_start = m_modbus_mem_ptr->getRegStart();
	bool is_processed = false;

	for (i = 0; i < quan; i++)
	{
		pos = addr + i - reg_start;
		if (!m_modbus_mem_ptr->checkRegCmdSection(pos))
		{
			is_processed = false;
			continue;
		}

		if (m_modbus_mem_ptr->checkRegCmdAddress(pos))
		{
			is_processed = true;
			onRequestCommand(pos);
		}
	}

	if (!is_processed)
	{
		if (m_modbus_mem_ptr->checkRegCmdAddress(pos))
		{
			onRequestCommand(pos);
		}
		else if (m_modbus_mem_ptr->checkRegAddress(addr, quan))
		{
			onRequestRegChange(addr, quan);
		}
	}
	return kMBSuccess;
}

#include "modbus_network.h"
#include "base.h"

CModBusNetwork::CModBusNetwork()
{
	m_is_inited = false;
	m_net_device = kPOCommNone;
}

CModBusNetwork::~CModBusNetwork()
{
}

bool CModBusNetwork::initInstance(i32 mode, CMBMemory* modbus_mem_ptr, CModbusDevParam* modbus_param_ptr)
{
	if (!modbus_param_ptr || !modbus_param_ptr)
	{
		return false;
	}

	if (!m_is_inited)
	{
		singlelog_lv0(QString("ModBusNetwork InitInstance, %1").arg(mode));

		m_is_inited = true;
		m_net_device = mode;
		m_modbus_mem_ptr = modbus_mem_ptr;
		CModBusServer::initInstance(mode, modbus_param_ptr);
	}
	return true;
}

void CModBusNetwork::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0(QString("ModBusNetwork ExitInstance, %1").arg(m_net_device));

		CModBusServer::exitInstance();
		m_is_inited = false;
		m_modbus_mem_ptr = NULL;
		m_modbus_param_ptr = NULL;
		m_net_device = kPOCommNone;
	}
}

void CModBusNetwork::restartNetwork()
{
	if (m_is_inited)
	{
		i32 mode = m_net_device;
		CMBMemory* modbus_mem_ptr = m_modbus_mem_ptr;
		CModbusDevParam* modbus_param_ptr = m_modbus_param_ptr;

		exitInstance();
		initInstance(m_net_device, modbus_mem_ptr, m_modbus_param_ptr);
	}
}

CModNetPacket* CModBusNetwork::onReadPacket(CModNetPacket* pak)
{
	if (!m_modbus_mem_ptr)
	{
		return NULL;
	}

	i32 quan, addr, tmp;
	i32 errtype = kMBSuccess;
	CModNetPacket* rpak = NULL;

	switch (pak->m_cmd)
	{
		case MODBUS_READCOIL://read built-in output and DO result [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rcoilreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rcoilreq.quan);
			if (quan < 1 || quan > MODNET_MAXRCOILQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getCoilStart() || addr + quan > m_modbus_mem_ptr->getCoilEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestReadCoil(pak, rpak, addr, quan);
			}
			break;
		}
		case MODBUS_READHOLD://read output register value [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rholdreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rholdreq.quan);
			if (quan < 1 || quan > MODNET_MAXRREGQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getHoldStart() || addr + quan > m_modbus_mem_ptr->getHoldEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = onRequestReadHoldRegister(pak, rpak, addr, quan);
			}
			break;
		}
		case MODBUS_READREG://read input register value writed already[multi]
		{
			addr = po::swap_endian16(pak->m_data_u->rregreq.addr);
			quan = po::swap_endian16(pak->m_data_u->rregreq.quan);
			if (quan < 1 || quan > MODNET_MAXRREGQUAN)
			{
				errtype = kMBErrQuantityOver;
			}
			else if (addr < m_modbus_mem_ptr->getRegStart() || addr + quan > m_modbus_mem_ptr->getRegEnd())
			{
				errtype = kMBErrAddressOver;
			}
			else
			{
				errtype = oRequestReadRegister(pak, rpak, addr, quan);
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
				errtype = onRequestWriteRegister(pak, rpak, addr, tmp);
			}
			break;
		}
		case MODBUS_MULWRITEREG://write input register [multi]
		{
			addr = po::swap_endian16(pak->m_data_u->mulwregreq.addr);
			quan = po::swap_endian16(pak->m_data_u->mulwregreq.quan);
			tmp = pak->m_data_u->mulwregreq.bytes;
			if (quan < 1 || quan > MODNET_MAXWREGQUAN || tmp != 2 * quan)
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
					m_modbus_mem_ptr->lock();
					u16* register_ptr = m_modbus_mem_ptr->getRegister();
					if (register_ptr)
					{
						memcpy((u8*)(register_ptr + addr), &(pak->m_data_u->mulwregreq.buffer), tmp);
					}
					m_modbus_mem_ptr->unlock();
				}
				errtype = onRequestMulWriteRegister(pak, rpak, addr, quan);
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
		POSAFE_DELETE(rpak);
		rpak = po_new CModNetPacket(false, pak->m_header_ptr, pak->m_cmd + MODBUS_ERROR);
		rpak->m_data_u->err.code = errtype;
	}
	return rpak;
}

i32 CModBusNetwork::onRequestReadCoil(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr)
	{
		return kMBErrProcess;
	}

	i32 i, bytes = (quan + 7) / 8;
	rpak = po_new CModNetPacket(false, pak->m_header_ptr, MODBUS_READCOIL, bytes);

	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister() + addr;
		if (register_ptr)
		{
			u8* buffer_ptr = &(rpak->m_data_u->rcoilresp.buffer);
			for (i = 0; i < quan; i++)
			{
				buffer_ptr[i / 8] += (register_ptr[i] ? 1 : 0) << (i % 8);
			}
		}
		m_modbus_mem_ptr->unlock();
	}
	rpak->m_data_u->rcoilresp.bytes = bytes;
	return kMBSuccess;
}

i32 CModBusNetwork::oRequestReadRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan)
{
	if (!m_modbus_mem_ptr)
	{
		return kMBErrProcess;
	}

	i32 bytes = quan * 2;
	rpak = po_new CModNetPacket(false, pak->m_header_ptr, MODBUS_READREG, bytes);
	
	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			memcpy(&(rpak->m_data_u->rregresp.buffer), (u8*)(register_ptr + addr), bytes);
		}
		m_modbus_mem_ptr->unlock();
	}
	rpak->m_data_u->rregresp.bytes = bytes;
	return kMBSuccess;
}

i32 CModBusNetwork::onRequestReadHoldRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan)
{
	if (!pak || !m_modbus_mem_ptr)
	{
		return kMBErrProcess;
	}

	i32 bytes = quan * 2;
	rpak = po_new CModNetPacket(false, pak->m_header_ptr, MODBUS_READHOLD, bytes);

	{
		m_modbus_mem_ptr->lock();
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			memcpy(&(rpak->m_data_u->rholdresp.buffer), (u8*)(register_ptr + addr), bytes);
		}
		m_modbus_mem_ptr->unlock();
	}
	rpak->m_data_u->rholdresp.bytes = bytes;
	return kMBSuccess;
}

i32 CModBusNetwork::onRequestWriteRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 value)
{
	if (!pak || !m_modbus_mem_ptr)
	{
		return kMBErrProcess;
	}

	i32 tmp = po::swap_endian16(value); //little endian -> big endian
	rpak = po_new CModNetPacket(false, pak->m_header_ptr, MODBUS_WRITEREG);
	rpak->m_data_u->wregresp.addr = po::swap_endian16(addr);
	rpak->m_data_u->wregresp.val = tmp;

	{
		anlock_guard_ptr(m_modbus_mem_ptr);
		u16* register_ptr = m_modbus_mem_ptr->getRegister();
		if (register_ptr)
		{
			register_ptr[addr] = tmp;
		}
	}

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

i32 CModBusNetwork::onRequestMulWriteRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan)
{
	rpak = po_new CModNetPacket(false, pak->m_header_ptr, MODBUS_MULWRITEREG);
	rpak->m_data_u->mulwregresp.addr = po::swap_endian16(addr);
	rpak->m_data_u->mulwregresp.quan = po::swap_endian16(quan);

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

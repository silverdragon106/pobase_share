#pragma once

#include "define.h"
#include "modbus_server.h"
#include "modbus_memory.h"

class CMBMemory;
class CModBusNetwork : public CModBusServer
{
	Q_OBJECT

public:
	CModBusNetwork();
	virtual ~CModBusNetwork();

	bool						initInstance(i32 mode, CMBMemory* modbus_mem_ptr, CModbusDevParam* modbus_param_ptr);
	void						exitInstance();
	void						restartNetwork();

	virtual bool				onRequestCommand(i32 addr) = 0;
	virtual bool				onRequestRegChange(i32 addr, i32 size) = 0;

private:
	CModNetPacket*				onReadPacket(CModNetPacket* pak);
	i32							onRequestReadCoil(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan);
	i32							oRequestReadRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan);
	i32							onRequestReadHoldRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan);
	i32							onRequestWriteRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 value);
	i32							onRequestMulWriteRegister(CModNetPacket* pak, CModNetPacket*& rpak, i32 addr, i32 quan);

public:
	bool						m_is_inited;
	i32							m_net_device;
	CMBMemory*					m_modbus_mem_ptr;
};


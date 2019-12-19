#pragma once

#include "define.h"
#include "modbus_com.h"
#include "modbus_memory.h"

class CMBMemory;
class CModBusSerial : public CModBusCom
{
	Q_OBJECT

public:
	CModBusSerial();
	virtual ~CModBusSerial();

	bool						initInstance(CMBMemory* modbus_mem_ptr, CModbusDevParam* modbus_param_ptr);
	void						exitInstance();

	virtual bool				onRequestCommand(i32 pos) = 0;
	virtual bool				onRequestRegChange(i32 addr, i32 size) = 0;

private:
	void						onReadPacket(CModBusPacket* pak);
	i32							onRequestReadCoil(i32 addr, i32 quan);
	i32							onRequestReadRegister(i32 addr, i32 quan);
	i32							onRequestReadHoldRegister(i32 addr, i32 quan);
	i32							onRequestWriteRegister(i32 addr, i32 value);
	i32							onRequestMulWriteRegister(i32 addr, i32 quan);

public:
	bool						m_is_inited;
	CMBMemory*					m_modbus_mem_ptr;
};


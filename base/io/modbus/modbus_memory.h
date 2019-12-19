#pragma once

#include "define.h"
#include "struct.h"
#include "lock_guide.h"

#pragma pack(push, 4)

class CMBMemory : public CLockGuard
{
public:
	CMBMemory();
	~CMBMemory();

	bool					initBuffer(i32 size, u16* section_addr_ptr, u16 section_count);
	void					freeBuffer();

	void					setRegRange(i32 st, i32 ed);
	void					setCoilRange(i32 st, i32 ed);
	void					setHoldRange(i32 st, i32 ed);

	bool					seekAddress(i32 addr);
	bool					seekAddress(i32 addr, i32 count, u8*& buffer_ptr);
	
	bool					getValueu8(u8& value, i32 endian_mode, i32 addr = -1);
	bool					getValueu16(u16& value, i32 endian_mode, i32 addr = -1);
	bool					getValueu32(u32& value, i32 endian_mode, i32 addr = -1);
	bool					getValueu64(u64& value, i32 endian_mode, i32 addr = -1);
	bool					getValuei8(i8& value, i32 endian_mode, i32 addr = -1);
	bool					getValuei16(i16& value, i32 endian_mode, i32 addr = -1);
	bool					getValuei32(i32& value, i32 endian_mode, i32 addr = -1);
	bool					getValuei64(i64& value, i32 endian_mode, i32 addr = -1);
	bool					getValuef32(f32& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					getValuef64(f64& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					getValueVector2df(vector2df& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					getValueRectf(Rectf& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);

	bool					setValueu8(u8 value, i32 endian_mode, i32 addr = -1);
	bool					setValueu16(u16 value, i32 endian_mode, i32 addr = -1);
	bool					setValueu32(u32 value, i32 endian_mode, i32 addr = -1);
	bool					setValueu64(u64 value, i32 endian_mode, i32 addr = -1);
	bool					setValuei8(i8 value, i32 endian_mode, i32 addr = -1);
	bool					setValuei16(i16 value, i32 endian_mode, i32 addr = -1);
	bool					setValuei32(i32 value, i32 endian_mode, i32 addr = -1);
	bool					setValuei64(i64 value, i32 endian_mode, i32 addr = -1);
	bool					setValuef32(f32 value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					setValuef64(f64 value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					setValueVector2df(vector2df& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);
	bool					setValueRectf(Rectf& value, i32 endian_mode, i32 fmt_number, i32 addr = -1);

	bool					checkRegCmdAddress(i32 pos);
	bool					checkRegCmdSection(i32 pos);
	bool					checkRegAddress(i32 addr, i32 count = 1);
	bool					checkCoilAddress(i32 addr, i32 count = 1);
	bool					checkHoldAddress(i32 addr, i32 count = 1);

	inline u16				getRegStart() { return m_reg_start; };
	inline u16				getCoilStart() { return m_coil_start; };
	inline u16				getHoldStart() { return m_hold_start; };
	inline u16				getRegEnd() { return m_reg_end; };
	inline u16				getCoilEnd() { return m_coil_end; };
	inline u16				getHoldEnd() { return m_hold_end; };
	inline u16				getRegCmdAddress(i32 pos) { return m_reg_addr_ptr ? m_reg_addr_ptr[pos] : 0xFFFF; };

	inline u16*				getRegister() { return m_register_ptr; };
	inline u16				getRegisterCount() { return m_max_address; };

	inline bool				isInited() { return m_is_inited; };

public:
	bool					m_is_inited;

	//address allocation
	u16						m_reg_start;
	u16						m_reg_end;
	u16						m_coil_start;
	u16						m_coil_end;
	u16						m_hold_start;
	u16						m_hold_end;

	//input register section
	u16						m_reg_cmd_count;
	u16*					m_reg_addr_ptr;
	bool*					m_reg_section_ptr;

	//table content
	u32						m_max_address;
	u16*					m_register_ptr;
	u16*					m_buffer_ptr;
};

#pragma pack(pop)
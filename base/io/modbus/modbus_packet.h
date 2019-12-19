#pragma once
#include "define.h"

#define MODNET_PROTOCAL			0
#define MODBUS_INTERVAL			20

#define MODBUS_NONE				0x00
#define MODBUS_READCOIL			0x01
#define MODBUS_READHOLD			0x03
#define MODBUS_READREG			0x04
#define MODBUS_WRITECOIL		0x05
#define MODBUS_WRITEREG			0x06
#define MODBUS_MULWRITECOIL		0x0F
#define MODBUS_MULWRITEREG		0x10
#define MODBUS_ERROR			0x80

#define MODBUS_MAXRCOILQUAN		0x07D0
#define MODBUS_MAXWCOILQUAN		0x07B0
#define MODBUS_MAXRREGQUAN		0x007D
#define MODBUS_MAXWREGQUAN		0x007B
#define MODBUS_MAXADDRESS		0xFFFF

#define MODNET_MAXRCOILQUAN		0x07F8
#define MODNET_MAXWCOILQUAN		0x07F8
#define MODNET_MAXRREGQUAN		0x007F
#define MODNET_MAXWREGQUAN		0x007F
#define MODNET_MAXADDRESS		0xFFFF

enum ModBusErrorType
{
	kMBSuccess			= 0x00,
	kMBErrFuncNoSupport = 0x01,
	kMBErrAddressOver	= 0x02,
	kMBErrQuantityOver	= 0x03,
	kMBErrInvalidValue	= 0x03,
	kMBErrProcess		= 0x04
};

//////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
struct MBAPHeader
{
	u16					tid;
	u16					pid;
	u16					len;
	u8					uid;

public:
	MBAPHeader()
	{
		tid = 0;
		pid = MODNET_PROTOCAL; 
		len = 0;
		uid = 0;
	};
	inline void update()
	{
		tid = po::swap_endian16(po::swap_endian16(tid) + 1);
	};
};
#pragma pack(pop)

#pragma pack(push, 4)
union DataPacket
{
	//MODBUS_READCOIL = 0x01
	struct ReadCoilReq		{ u16 addr; u16 quan; } rcoilreq;
	struct ReadCoilResp		{ u8 bytes; u8 buffer; } rcoilresp;

	//MODBUS_READHOLD = 0x03
	struct ReadHoldReq		{ u16 addr; u16 quan; } rholdreq;
	struct ReadHoldResp		{ u8 bytes; u8 buffer; } rholdresp;

	//MODBUS_READREG = 0x04
	struct ReadRegReq		{ u16 addr; u16 quan; } rregreq;
	struct ReadRegResp		{ u8 bytes; u8 buffer; } rregresp;

	//MODBUS_WRITECOIL = 0x05
	struct WriteCoilReq		{ u16 addr; u16 val; } wcoilreq;
	struct WriteCoilResp	{ u16 addr; u16 val; } wcoilresp;

	//MODBUS_WRITEREG = 0x06
	struct WriteRegReq		{ u16 addr; u16 val; } wregreq;
	struct WriteRegResp		{ u16 addr; u16 val; } wregresp;

	//MODBUS_MULWRITECOIL = 0x0F
	struct MulWriteCoilReq	{ u16 addr; u16 quan; u8 bytes; u8 buffer; } mulwcoilreq;
	struct MulWriteCoilResp { u16 addr; u16 quan; } mulwcoilresp;

	//MODBUS_MULWRITEREG = 0x10
	struct MulWriteRegReq	{ u16 addr; u16 quan; u8 bytes; u8 buffer; } mulwregreq;
	struct MulWriteRegResp	{ u16 addr; u16 quan; } mulwregresp;

	//MODBUS_ERROR >= 0x80
	struct ErrorResp		{ u8 code; } err;
};

class CModBusPacket
{
public:
	CModBusPacket(bool is_master, u8 mbdevaddr, u8 mbcmd, u8 datalen = 0);
	CModBusPacket();
	~CModBusPacket();

	bool				isModBusPacket(bool is_master, i32 devaddr, u8* buffer_ptr, i32 len);
	inline bool			isValid() { return m_is_valid; };

	void				makeCRCCode();
	static u16			makeCRCCode(u8* buffer_ptr, u16 len);

private:
	void				makeRequestPacket(u8 mbdevaddr, u8 mbcmd, u8 datalen);
	void				makeResponsePacket(u8 mbdevaddr, u8 mbcmd, u8 datalen);

	void				packetFromMem(bool is_master, u8* buffer_ptr, i32 memlen);
	void				responsePacketFromMem(u8* buffer_ptr);
	void				requestPacketFromMem(u8* buffer_ptr);

public:
	u8					m_dev_addr;
	u8					m_cmd;
	DataPacket*			m_data_u;
	u16*				m_crc_code;

	u16					m_len;			//for only management
	u8*					m_buffer_ptr;	//for only management
	bool				m_is_valid;		//for only management
};

class CModNetPacket
{
public:
	CModNetPacket(bool is_client, MBAPHeader* header_ptr, u8 mbcmd, u8 datalen = 0);
	CModNetPacket();
	~CModNetPacket();

	CModNetPacket*		clone();

	bool				isModNetPacket(bool is_client, u8* buffer_ptr, i32 len);
	inline bool			isValid() { return m_is_valid; };

private:
	void				makeRequestPacket(MBAPHeader* header_ptr, u8 mbcmd, u8 datalen);
	void				makeResponsePacket(MBAPHeader* header_ptr, u8 mbcmd, u8 datalen);

	void				packetFromMem(bool is_client, u8* buffer_ptr, i32 memlen);
	void				responsePacketFromMem(u8* buffer_ptr);
	void				requestPacketFromMem(u8* buffer_ptr);

public:
	MBAPHeader*			m_header_ptr;
	u8					m_cmd;
	DataPacket*			m_data_u;

	u16					m_len;			//for only management
	u8*					m_buffer_ptr;	//for only management
	bool				m_is_valid;		//for only management
};
#pragma pack(pop)

typedef std::list<CModNetPacket*> ModNetPacketList;

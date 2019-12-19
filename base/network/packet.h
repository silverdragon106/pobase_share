#pragma once
#include "struct.h"

//////////////////////////////////////////////////////////////////////////
#define LOCALHOST					0x0100007F		//1.0.0.127
#define MAX_SOCK_BUFSIZE			52428800		//50MBytes

#define PACKET_SIGN_CODE			0xA1A2A3A4
#define PO_PAKVERSION				101	//2019.1.5

#define PO_HBVERSION				105	//2018.12.27
//										remove - is_remotable, 
//									104 //2018.9.15
//										add network information(such as gateway, dnsserver, is_dhcp)
//									103 //2018.3.6
//										optimize parameters
//									102 //2017.06.08
//										add - bremotable(remoteable flag)
//										add - bclient (current connected high-level)
//									101 //2017.04.08

//common structure for command packet_ptr with tcp-connection
#define PACKET_RESERVED_NUMS		4
#define PACKET_HEADER_BYTES			(sizeof(PacketHeader))
#define PACKET_SENDBUFFER_BYTES		20480 //20K
#define PACKET_CRC_LIMIT_SIZE		20480

enum PacketDataOperTypes
{
	kPacketDataNone = 0,
	kPacketDataCopy,
	kPacketDataMove,
	kPacketDataShare
};

enum PacketFlagTypes
{
	kPacketFlagNone = 0x00,
	kPacketFlagLast = 0x10,
	kPacketFlagImage = 0x20,
	kPacketFlagVideo = 0x40,

	kPacketFlagStream = (kPacketFlagImage | kPacketFlagVideo)
};

#pragma pack(push, 4)
struct PacketHeader
{
	u32					pak_sign;		//PACKET_SIGN_CODE
	u16					pak_version;	//PO_PAKVERSION
	u32					pak_id;			//0 ~ max
	
	u16					app_type;		//enum POApplication
	u8					pak_type;		//enum HeaderType
	u8					pak_flag;		//enum PacketFlagTypes

	u16					cmd;
	u32					sub_cmd;
	i32					code;			//retrun code;
	u32					reserved[PACKET_RESERVED_NUMS];
};

struct Packet
{
	PacketHeader		header;
	u32					data_size;
	u8*					data_ptr;

	bool				is_require_delete;
	u16					crc_code;

public:
	Packet();
	Packet(Packet* pak);
	Packet(u16 cmd, u8 type, bool is_last = true, i32 id = -1);
	~Packet();

	static Packet*		makeRespPacket(Packet* pak, bool is_last = true);
	static Packet*		makeRespPacket(Packet* pak, i32 result, i32 data_mode, bool is_last = true);
	static i32			calcHeaderSize();

	i32					memSize();
	bool				memWrite(u8*& buffer_ptr);
	bool				memRead(u8*& buffer_ptr);

	u16					makeCRCCode();
	bool				allocateBuffer(i32 len, u8*& buffer_ptr);

	void				clone(Packet* packet_ptr);
	void				copyData(Packet* packet_ptr);
	void				freeData();

	void				setReservedi32(i32 index, i32 val);
	void				setReservedi64(i32 index, i64 val);
	void				setReservedf32(i32 index, f32 val);
	void				setReservedf64(i32 index, f64 val);
	void				setReservedb8(i32 index, bool val);
	void				setReservedu8(i32 index, u8 val);

	i32					getReservedi32(i32 index);
	bool				getReservedb8(i32 index);
	f32					getReservedf32(i32 index);
	f64					getReservedf64(i32 index);
	i64					getReservedi64(i32 index);
	u16					getReservedu16(i32 index);
	u8					getReservedu8(i32 index);
	u8*					getReservedData();

	void				setHeaderType(u8 type);
	void				setCmd(u16 cmd);
	void				setSubCmd(u32 sub_cmd);
	void				setReturnCode(i32 code);
	void				setData(u8* data_ptr, i32 len, bool is_require_delete);
	void				setData(Packet* packet_ptr);
	void				setLastPacket(bool is_last);
	void				setPacketFlag(i32 flag);
	void				setDirty();

	inline bool			isDirty()	{ return header.cmd == kPOCmdNone; };
	inline bool			isLast()	{ return (header.pak_flag & kPacketFlagLast) != 0; };
	inline bool			isRTDPacket() { return (header.pak_flag & kPacketFlagStream) != 0; };

	inline u8			getHeaderType() { return header.pak_type; };
	inline u16			getDeviceType() { return header.app_type; };
	inline u16			getID()			{ return header.pak_id; };
	inline u16			getCmd()		{ return header.cmd; };
	inline u32			getSubCmd()		{ return header.sub_cmd; };
	inline i32			getReturnCode()	{ return header.code; };
	inline u8*			getData()		{ return data_ptr; };
	inline u32			getDataLen()	{ return data_size; };
	
	template<typename T>
	void				setReserved(i32 i, T v) { ((T*)header.reserved)[i] = v; };
};
#pragma pack(pop)
typedef std::vector<Packet*> PacketPVector;
typedef std::list<Packet*> PacketPList;

//find-device structure for find-device-service packet_ptr with Udp-broadcast
enum HBStateType
{
	kHBStateUnknown = 0,
	kHBStateIdle,
	kHBStateWork,
	kHBStateOFF
};

#pragma pack(push, 4)
struct FDPacket
{
	u16	len;
	union _FDData
	{
		struct _FDHeader
		{
			i32			token;
			u16			type;
			u16			len;
			char		buf;
		} pak;
		char data[1024];
	} d;

public:
	FDPacket() { memset(this, 0, sizeof(FDPacket)); };
	inline void clone(FDPacket& other) { memcpy(this, &other, sizeof(FDPacket)); };
};

struct HeartBeat
{
	u16					hb_ver;
	i32					dev_id;
	bool				is_emulator;
	postring			device_name;
	postring			device_model;
	postring			device_version;

	u16					cmd_port;
	u16					ivs_port;
	u8					state;		//HDState
	i32					connections;
	i32					duration;	//seconds from lowlevel boot-up

	NetAdapterArray		netadapter_vec;

public:
	HeartBeat();

	void				init();

	i32					memSize();
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);

	void				clone(HeartBeat& hb);
	void				setUDPPacket(FDPacket* upak);
};

struct ServerInfo
{
	i32					dev_type;
	i32					dev_id;
	i32					dev_state;		// server connection status
	i32					dev_adapter;	//
	i32					dev_ip;
	i32					subnet_mask;
	i32					duration;
	bool				is_conf_dhcp;

	i32					client_ip;		/* added by sdragon */
	bool				is_remotable;	/* added by sdragon */

	i32					cmd_port;		// tcp port
	postring			adapter_name;
	postring			device_name;
	postring			device_desc;

	bool				is_update;
	i32					net_adapter;
	u32					last_update_time;
	postring			password;

	ServerInfo();
	bool isValid() const;
	bool operator == (const ServerInfo& sinfo) const;
	bool operator != (const ServerInfo& sinfo) const;
};

#pragma pack(pop)

//////////////////////////////////////////////////////////////////////////
class PacketID
{
	friend class PacketLessThan;
public:
	u16					cmd;
	u16					sub_cmd;

public:
	PacketID(const u16 cmd, const u16 sub_cmd);
};

class PacketLessThan
{
public:
	bool operator()(const PacketID&, const PacketID&) const;
};
typedef std::map<PacketID, Packet*, PacketLessThan>	PacketIDMap;

struct PacketQueueItem
{
public:
	i32					conn;
	Packet*				pak;

public:
	PacketQueueItem() { conn = 0; pak = NULL; }
	PacketQueueItem(i32 conn_, Packet* pak_) { conn = conn_; pak = pak_; }
};
typedef std::list<PacketQueueItem>		PacketQueueList;
typedef std::vector<PacketQueueItem>	PacketQueueVector;

#include "packet.h"
#include "base.h"
#include "connection.h"

i32 g_pak_id = 0;
inline i32 getNextPakId()
{
	return g_pak_id++;
}

Packet::Packet()
{
	memset(this, 0, sizeof(Packet));

	header.pak_sign = PACKET_SIGN_CODE;
	header.pak_version = PO_PAKVERSION;
	header.app_type = PO_CUR_DEVICE;
	header.pak_id = getNextPakId();
	is_require_delete = true;
}

Packet::Packet(Packet* pak)
{
	memcpy(this, (u8*)pak, sizeof(Packet));

	header.pak_sign = PACKET_SIGN_CODE;
	header.pak_version = PO_PAKVERSION;
	header.app_type = PO_CUR_DEVICE;
	is_require_delete = true;

	if (data_size > 0)
	{
		data_ptr = po_new u8[data_size];
		CPOBase::memCopy(data_ptr, pak->data_ptr, data_size);
	}
	else
	{
		data_ptr = NULL;
	}
}

Packet::Packet(u16 cmd, u8 type, bool is_last, i32 id)
{
	memset(this, 0, sizeof(Packet));
	header.pak_sign = PACKET_SIGN_CODE;
	header.pak_version = PO_PAKVERSION;
	header.pak_id = (id < 0) ? getNextPakId() : id;

	header.app_type = PO_CUR_DEVICE;
	header.pak_type = type;
	header.pak_flag = (is_last ? kPacketFlagLast : kPacketFlagNone);
	header.cmd = cmd;
	is_require_delete = true;
}

Packet::~Packet()
{
	freeData();
}

bool Packet::allocateBuffer(i32 len, u8*& buffer_ptr)
{
	buffer_ptr = NULL;
	if (len <= 0)
	{
		return false;
	}

	POSAFE_DELETE_ARRAY(data_ptr);

	data_size = len;
	data_ptr = po_new u8[len];
	memset(data_ptr, 0, len);
	buffer_ptr = data_ptr;
	return true;
}

void Packet::setReservedi32(i32 index, i32 val)
{
	header.reserved[index] = val;
}

void Packet::setReservedi64(i32 index, i64 val)
{
	*(((i64*)header.reserved) + index) = val;
}

void Packet::setReservedf32(i32 index, f32 val)
{
	*(((f32*)header.reserved) + index) = val;
}

void Packet::setReservedf64(i32 index, f64 val)
{
	*(((f64*)header.reserved) + index) = val;
}

void Packet::setReservedb8(i32 index, bool val)
{
	*(((bool*)header.reserved) + index) = val;
}

void Packet::setReservedu8(i32 index, u8 val)
{
	*(((bool*)header.reserved) + index) = val;
}

i32	Packet::getReservedi32(i32 index)
{
	return *((i32*)header.reserved + index);
}

i64 Packet::getReservedi64(i32 index)
{
	return *((i64*)header.reserved + index);
}

f32 Packet::getReservedf32(i32 index)
{
	return *(((f32*)header.reserved) + index);
}

f64 Packet::getReservedf64(i32 index)
{
	return *(((f64*)header.reserved) + index);
}

bool Packet::getReservedb8(i32 index)
{
	return *(((bool*)header.reserved) + index);
}

u16 Packet::getReservedu16(i32 index)
{
	return *(((u16*)header.reserved) + index);
}

u8 Packet::getReservedu8(i32 index)
{
	return *(((u8*)header.reserved) + index);
}

u8* Packet::getReservedData()
{
	return (u8*)(header.reserved);
}

void Packet::setHeaderType(u8 type)
{
	header.pak_type = type;
}

void Packet::setCmd(u16 cmd)
{
	header.cmd = cmd;
}

void Packet::setSubCmd(u32 sub_cmd)
{
	header.sub_cmd = sub_cmd;
}

void Packet::setReturnCode(i32 code)
{
	header.code = code;
}

void Packet::setData(u8* data_ptr, i32 len, bool is_require_delete)
{
	this->data_ptr = data_ptr;
	this->data_size = len;
	this->is_require_delete = is_require_delete;
}

void Packet::setData(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return;
	}
	this->data_ptr = packet_ptr->data_ptr;
	this->data_size = packet_ptr->data_size;
	this->is_require_delete = packet_ptr->is_require_delete;
}

void Packet::setDirty()
{
	if (is_require_delete)
	{
		POSAFE_DELETE_ARRAY(data_ptr);
	}
	data_size = 0;
	is_require_delete = false;
	header.cmd = kPOCmdNone;
}

void Packet::setLastPacket(bool is_last)
{
	if (is_last)
	{
		header.pak_flag |= kPacketFlagLast;
	}
	else
	{
		header.pak_flag &= ~kPacketFlagLast;
	}
}

void Packet::setPacketFlag(i32 flag)
{
	header.pak_flag |= flag;
}

Packet* Packet::makeRespPacket(Packet* pak, bool is_last)
{
	if (!pak || pak->getHeaderType() != kPOPacketRequest)
	{
		return NULL;
	}
	
	Packet* new_pak_ptr = po_new Packet();
	{
		new_pak_ptr->header = pak->header;
		new_pak_ptr->header.pak_type = kPOPacketRespOK;
		new_pak_ptr->setLastPacket(is_last);
	}
	return new_pak_ptr;
}

Packet* Packet::makeRespPacket(Packet* pak, i32 result, i32 data_mode, bool is_last)
{
	if (!pak || pak->getHeaderType() != kPOPacketRequest)
	{
		return NULL;
	}

	Packet* new_pak_ptr = po_new Packet();

	new_pak_ptr->header = pak->header;
	new_pak_ptr->header.pak_type = (result == kPOSuccess) ? kPOPacketRespOK : kPOPacketRespFail;
	new_pak_ptr->header.code = result;
	new_pak_ptr->setLastPacket(new_pak_ptr);

	switch (data_mode)
	{
		case kPacketDataCopy:
		{
			new_pak_ptr->copyData(pak);
			break;
		}
		case kPacketDataMove:
		{
			new_pak_ptr->setData(pak);
			pak->setData(NULL, 0, false);
			break;
		}
		case kPacketDataShare:
		{
			new_pak_ptr->setData(pak);
			break;
		}
	}
	return new_pak_ptr;
}

i32 Packet::calcHeaderSize()
{
	return sizeof(PacketHeader) + sizeof(u32) + sizeof(u16); //data_size, crc_code
}

i32 Packet::memSize()
{
	i32 len = 0;
	len += sizeof(header);
	len += sizeof(data_size);
	len += sizeof(crc_code);
	len += data_size;
	return len;
}

bool Packet::memWrite(u8*& buffer_ptr)
{
	if (!buffer_ptr)
	{
		return false;
	}

	CPOBase::memWrite(header, buffer_ptr);
	CPOBase::memWrite(data_size, buffer_ptr);
	CPOBase::memWrite(crc_code, buffer_ptr);
	if (CPOBase::isPositive(data_size))
	{
		CPOBase::memWrite(data_ptr, data_size, buffer_ptr);
	}
	return true;
}

bool Packet::memRead(u8*& buffer_ptr)
{
	freeData();
	if (!buffer_ptr)
	{
		return false;
	}

	CPOBase::memRead(header, buffer_ptr);
	CPOBase::memRead(data_size, buffer_ptr);
	CPOBase::memRead(crc_code, buffer_ptr);
	if (CPOBase::isPositive(data_size))
	{
		data_ptr = po_new u8[data_size];
		CPOBase::memRead(data_ptr, data_size, buffer_ptr);
	}
	return true;
}

u16 Packet::makeCRCCode()
{
	u16 code = CPOBase::makeCRCCode((u8*)&header, sizeof(PacketHeader));
	if (!isRTDPacket() && data_ptr && CPOBase::checkCount(data_size, PACKET_CRC_LIMIT_SIZE))
	{
		code ^= CPOBase::makeCRCCode(data_ptr, data_size);
	}
	crc_code = code;
	return crc_code;
}

void Packet::clone(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return;
	}

	header = packet_ptr->header;
	data_size = packet_ptr->data_size;
	is_require_delete = packet_ptr->is_require_delete;

	if (data_size)
	{
		data_ptr = po_new u8[data_size];
		CPOBase::memCopy(data_ptr, packet_ptr->getData(), data_size);
	}
	else
	{
		data_ptr = NULL;
	}
}

void Packet::copyData(Packet* packet_ptr)
{
	if (!packet_ptr)
	{
		return;
	}

	freeData();

	i32 len = packet_ptr->getDataLen();
	if (len > 0)
	{
		data_size = len;
		data_ptr = po_new u8[len];
		CPOBase::memCopy(data_ptr, packet_ptr->getData(), len);
	}
}

void Packet::freeData()
{
	if (is_require_delete)
	{
		POSAFE_DELETE_ARRAY(data_ptr);
	}
	data_size = 0;
	data_ptr = NULL;
	is_require_delete = true;
}

//////////////////////////////////////////////////////////////////////////
HeartBeat::HeartBeat()
{
	init();
}

void HeartBeat::init()
{
	hb_ver = PO_HBVERSION;
	dev_id = -1;
	is_emulator = false;
	device_name = "";
	device_model = "";
	device_version = "";

	cmd_port = 0;
	ivs_port = 0;
	state = kHBStateUnknown;
	connections = 0;
	duration = 0;
	
	netadapter_vec.clear();
}

i32 HeartBeat::memSize()
{
	i32 len = 0;

	len += sizeof(hb_ver);
	len += sizeof(dev_id);
	len += sizeof(is_emulator);
	len += CPOBase::getStringMemSize(device_name);
	len += CPOBase::getStringMemSize(device_model);
	len += CPOBase::getStringMemSize(device_version);

	len += sizeof(cmd_port);
	len += sizeof(ivs_port);
	len += sizeof(state);
	len += sizeof(connections);
	len += sizeof(duration);

	len += sizeof(i32);
	for (i32 i = 0; i < netadapter_vec.size(); i++)
	{
		NetAdapter& adapter = netadapter_vec[i];
		len += adapter.memSize();
	}
	return len;
}

i32 HeartBeat::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(hb_ver, buffer_ptr, buffer_size);
	CPOBase::memWrite(dev_id, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_emulator, buffer_ptr, buffer_size);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_name);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_model);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_version);

	CPOBase::memWrite(cmd_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(ivs_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(state, buffer_ptr, buffer_size);
	CPOBase::memWrite(connections, buffer_ptr, buffer_size);
	CPOBase::memWrite(duration, buffer_ptr, buffer_size);

	i32 i, count = (i32)netadapter_vec.size();
	CPOBase::memWrite(count, buffer_ptr, buffer_size);

	for (i = 0; i < count; i++)
	{
		netadapter_vec[i].memWrite(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

i32 HeartBeat::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	init();
	CPOBase::memRead(hb_ver, buffer_ptr, buffer_size);
	CPOBase::memRead(dev_id, buffer_ptr, buffer_size);
	CPOBase::memRead(is_emulator, buffer_ptr, buffer_size);
	CPOBase::memRead(buffer_ptr, buffer_size, device_name);
	CPOBase::memRead(buffer_ptr, buffer_size, device_model);
	CPOBase::memRead(buffer_ptr, buffer_size, device_version);

	CPOBase::memRead(cmd_port, buffer_ptr, buffer_size);
	CPOBase::memRead(ivs_port, buffer_ptr, buffer_size);
	CPOBase::memRead(state, buffer_ptr, buffer_size);
	CPOBase::memRead(connections, buffer_ptr, buffer_size);
	CPOBase::memRead(duration, buffer_ptr, buffer_size);

	i32 i, count = (i32)netadapter_vec.size();
	CPOBase::memRead(count, buffer_ptr, buffer_size);
	if (!CPOBase::isPositive(count))
	{
		return buffer_ptr - buffer_pos;
	}

	netadapter_vec.resize(count);
	for (i = 0; i < count; i++)
	{
		netadapter_vec[i].memRead(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

void HeartBeat::clone(HeartBeat& hb)
{
	*this = hb;
}

void HeartBeat::setUDPPacket(FDPacket* upak)
{
	i32 len = memSize();
	if (len > sizeof(FDPacket) - sizeof(FDPacket::_FDData::_FDHeader))
	{
		return;
	}

	upak->d.pak.token = POTOKEN_START;
	upak->d.pak.type = kPOHeartBeatUdp;
	upak->d.pak.len = len;

	i32 buffer_size = len;
	u8* buffer_ptr = (u8*)&upak->d.pak.buf;
	
	CPOBase::memWrite(hb_ver, buffer_ptr, buffer_size);
	CPOBase::memWrite(dev_id, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_emulator, buffer_ptr, buffer_size);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_name);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_model);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_version);

	CPOBase::memWrite(cmd_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(ivs_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(state, buffer_ptr, buffer_size);
	CPOBase::memWrite(connections, buffer_ptr, buffer_size);
	CPOBase::memWrite(duration, buffer_ptr, buffer_size);
	
	i32 i, count = (i32)netadapter_vec.size();
	CPOBase::memWrite(count, buffer_ptr, buffer_size);

	for (i = 0; i < count; i++)
	{
		netadapter_vec[i].memWrite(buffer_ptr, buffer_size);
	}
	upak->len = buffer_ptr - (u8*)upak->d.data;
}

//////////////////////////////////////////////////////////////////////////
ServerInfo::ServerInfo()
{
	dev_id = -1;
	dev_state = kHBStateOFF;
	dev_ip = 0;
	subnet_mask = 0;
	duration = 0;
	net_adapter = 0;
	is_update = true;
	is_conf_dhcp = false;

	cmd_port = 0;
}

bool ServerInfo::isValid() const
{
	return dev_id >= 0;
}

bool ServerInfo::operator==(const ServerInfo& sinfo) const
{
	return dev_id == sinfo.dev_id &&
		is_conf_dhcp == sinfo.is_conf_dhcp && dev_ip == sinfo.dev_ip && subnet_mask == sinfo.subnet_mask &&
		dev_state == sinfo.dev_state && duration == sinfo.duration && device_name == sinfo.device_name && 
		device_desc == sinfo.device_desc && net_adapter == sinfo.net_adapter;
}

bool ServerInfo::operator!=(const ServerInfo& sinfo) const
{
	return !(*this == sinfo);
}

//////////////////////////////////////////////////////////////////////////////
PacketID::PacketID(const u16 cmd, const u16 sub_cmd)
{
	this->cmd = cmd;
	this->sub_cmd = sub_cmd;
}

bool PacketLessThan::operator()(const PacketID& p1, const PacketID& p2) const
{
	return ((p1.cmd << 16) | p1.sub_cmd) < ((p2.cmd << 16) | p2.sub_cmd);
}

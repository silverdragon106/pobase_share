#include "struct.h"
#include "base.h"

#if defined(POR_EXPLORER)
	#include "version.h"
#endif

//////////////////////////////////////////////////////////////////////////
CFileHeader::CFileHeader()
{
	memset(this, 0, sizeof(CFileHeader));
}

CFileHeader::CFileHeader(const char* sz_header)
{
	memset(this, 0, sizeof(CFileHeader));

#if defined(POR_WINDOWS)
    strcpy_s(header, 32, sz_header);
#elif defined(POR_LINUX)
    strncpy(header, sz_header, 32);
#endif
}

void CFileHeader::setSeek(i32 id, u64 pos)
{
	seek[id] = pos;
}

//////////////////////////////////////////////////////////////////////////
DateTime::DateTime()
{
	init();
}

DateTime::DateTime(u16 _yy, u8 _mm, u8 _dd, u8 _h, u8 _m, u8 _s, u16 _ms)
{
	setDateTime(_yy, _mm, _dd, _h, _m, _s, _ms);
}

void DateTime::init()
{
	yy = 0; mm = 0; dd = 0;
	h = 0; m = 0; s = 0; ms = 0;
}

void DateTime::setDateTime(u16 _yy, u8 _mm, u8 _dd, u8 _h, u8 _m, u8 _s, u16 _ms)
{
	yy = _yy; mm = _mm; dd = _dd;
	h = _h; m = _m; s = _s; ms = _ms;
}

bool DateTime::isEqual(const DateTime& r_dtm) const
{
	return (yy == r_dtm.yy && mm == r_dtm.mm && dd == r_dtm.dd &&
		h == r_dtm.h && m == r_dtm.m && s == r_dtm.s &&
		ms == r_dtm.ms);
}

postring DateTime::toString(i32 id)
{
	char filename[PO_MAXPATH];
	if (id >= 0)
	{
		po_sprintf(filename, PO_MAXPATH, "%04d%02d%02d_%02d%02d%02d_%05d", yy, mm, dd, h, m, s, id);
		return postring(filename);
	}

	po_sprintf(filename, PO_MAXPATH, "%04d%02d%02d_%02d%02d%02d", yy, mm, dd, h, m, s);
	return postring(filename);
}

//////////////////////////////////////////////////////////////////////////
BlobData::BlobData()
{
	m_blob_size = 0;
	m_blob_data_ptr = NULL;
	m_is_external = false;
}

BlobData::~BlobData()
{
	freeBuffer();
}

void BlobData::initBuffer(const u8* buffer_ptr, i32 size)
{
	freeBuffer();

	if (size > 0)
	{
		m_is_external = false;
		m_blob_size = size;
		m_blob_data_ptr = po_new u8[size];
		if (buffer_ptr)
		{
			CPOBase::memCopy(m_blob_data_ptr, buffer_ptr, size);
		}
	}
}

void BlobData::setBuffer(u8* buffer_ptr, i32 size)
{
	freeBuffer();

	if (buffer_ptr && size > 0)
	{
		m_is_external = true;
		m_blob_data_ptr = buffer_ptr;
		m_blob_size = size;
	}
}

void BlobData::freeBuffer()
{
	if (!m_is_external)
	{
		POSAFE_DELETE_ARRAY(m_blob_data_ptr);
	}

	m_is_external = false;
	m_blob_size = 0;
	m_blob_data_ptr = NULL;
}

void BlobData::clone(const BlobData& other)
{
	initBuffer(other.m_blob_data_ptr, other.m_blob_size);
}

i32 BlobData::memSize()
{
	i32 len = 0;
	len += m_blob_size;
	len += sizeof(m_blob_size);
	return len;
}

i32 BlobData::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	i32 len = 0;
	u8* tmp_buffer_pos = buffer_ptr;
	freeBuffer();

	CPOBase::memRead(len, buffer_ptr, buffer_size);
	if (CPOBase::checkRange(len, buffer_size))
	{
		m_is_external = false;
		m_blob_size = len;

		if (CPOBase::isCount(m_blob_size) && m_blob_size > 0)
		{
			m_blob_data_ptr = po_new u8[len];
			CPOBase::memRead(m_blob_data_ptr, m_blob_size, buffer_ptr, buffer_size);
		}
	}
	return buffer_ptr - tmp_buffer_pos;
}

i32 BlobData::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* tmp_buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_blob_size, buffer_ptr, buffer_size);
	if (m_blob_data_ptr && CPOBase::isCount(m_blob_size))
	{
		CPOBase::memWrite(m_blob_data_ptr, m_blob_size, buffer_ptr, buffer_size);
	}
	return buffer_ptr - tmp_buffer_pos;
}

bool BlobData::fileRead(FILE* fp)
{
	freeBuffer();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	i32 len = 0;
	CPOBase::fileRead(len, fp);
	if (CPOBase::isPositive(len))
	{
		m_blob_size = len;
		m_blob_data_ptr = po_new u8[len];
		CPOBase::fileRead(m_blob_data_ptr, len, fp);
	}

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool BlobData::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	
	CPOBase::fileWrite(m_blob_size, fp);
	if (CPOBase::isPositive(m_blob_size) && m_blob_data_ptr)
	{
		CPOBase::fileWrite(m_blob_data_ptr, m_blob_size, fp);
	}
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
UpdateInfo::UpdateInfo()
{
	init();
}

UpdateInfo::~UpdateInfo()
{
}

void UpdateInfo::init()
{
	is_update = false;
	is_update_ready = false;
	update_from = 0;
	update_packet_id = 0;
	read_size = 0;
	read_id = 0;

	model_name = "";
	update_version = "";
	extra_key.clear();
	extra_value.clear();
	lowlevel_file_vec.clear();
	lowlevel_dir_vec.clear();
	highlevel_file_vec.clear();
	highlevel_dir_vec.clear();
	update_compatibility = kPOAppUpdateCPTUnknown;
	is_update_highlevel = false;
	filesize[0] = 0;
	filesize[1] = 0;
}

//////////////////////////////////////////////////////////////////////////
DeviceInfo::DeviceInfo()
{
	device_id = PO_DEVICE_ID;
	device_name = PO_DEVICE_NAME;
	device_version = PO_DEVICE_VERSION;
	model_name = PO_DEVICE_MODELNAME;
	is_hl_embedded = false;
	is_auto_update = true;

	comm_port = PO_NETWORK_PORT;
	video_encoder = kPOEncoderNone;
	build_date.init();

#if defined(POR_EMULATOR)
	is_emulator = true;
#else
	is_emulator = false;
#endif
}

DeviceInfo::~DeviceInfo()
{
}

bool DeviceInfo::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		printlog_lv1("DeviceParam start-sign invalid in setting.dat");
		return false;
	}

	CPOBase::fileRead(fp, model_name);
	CPOBase::fileRead(fp, device_version);

	if (!CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE))
	{
		printlog_lv1("DeviceParam end-sign invalid in setting.dat");
		return false;
	}
	return true;
}

bool DeviceInfo::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	CPOBase::fileWrite(fp, model_name);
	CPOBase::fileWrite(fp, device_version);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

i32 DeviceInfo::memSize()
{
	i32 len = 0;
	len += sizeof(device_id);
	len += CPOBase::getStringMemSize(device_name);
	len += CPOBase::getStringMemSize(device_version);
	len += CPOBase::getStringMemSize(model_name);
	len += sizeof(is_hl_embedded);
	len += sizeof(is_emulator);

	len += sizeof(comm_port);
	len += sizeof(video_encoder);
	len += sizeof(build_date);
	return len;
}

i32 DeviceInfo::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(device_id, buffer_ptr, buffer_size);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_name);
	CPOBase::memWrite(buffer_ptr, buffer_size, device_version);
	CPOBase::memWrite(buffer_ptr, buffer_size, model_name);
	CPOBase::memWrite(is_hl_embedded, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_emulator, buffer_ptr, buffer_size);

	CPOBase::memWrite(comm_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(video_encoder, buffer_ptr, buffer_size);
	CPOBase::memWrite(build_date, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 DeviceInfo::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(device_id, buffer_ptr, buffer_size);
	CPOBase::memRead(buffer_ptr, buffer_size, device_name);
	CPOBase::memRead(buffer_ptr, buffer_size, device_version);
	CPOBase::memRead(buffer_ptr, buffer_size, model_name);
	CPOBase::memRead(is_hl_embedded, buffer_ptr, buffer_size);
	CPOBase::memRead(is_emulator, buffer_ptr, buffer_size);

	CPOBase::memRead(comm_port, buffer_ptr, buffer_size);
	CPOBase::memRead(video_encoder, buffer_ptr, buffer_size);
	CPOBase::memRead(build_date, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool DeviceInfo::isCompatibility(DeviceInfo& other)
{
	if (device_version != other.device_version || model_name != other.model_name)
	{
		return false;
	}
	return true;
}

void DeviceInfo::import(DeviceInfo& other)
{
	comm_port = other.comm_port;
	is_auto_update = other.is_auto_update;
}

//////////////////////////////////////////////////////////////////////////
NetAdapter::NetAdapter()
{
	init();
}

void NetAdapter::init()
{
	ip_address = 0;
	ip_subnet = 0;
	ip_gateway = 0;
	ip_dns_server = 0;
	is_loopback = false;
	is_conf_dhcp = false;
	adapter_name = "";
	mac_address = "";
}

i32 NetAdapter::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	init();
	CPOBase::memRead(ip_address, buffer_ptr, buffer_size);
	CPOBase::memRead(ip_subnet, buffer_ptr, buffer_size);
	CPOBase::memRead(ip_gateway, buffer_ptr, buffer_size);
	CPOBase::memRead(ip_dns_server, buffer_ptr, buffer_size);
	CPOBase::memRead(is_loopback, buffer_ptr, buffer_size);
	CPOBase::memRead(is_conf_dhcp, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 NetAdapter::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(ip_address, buffer_ptr, buffer_size);
	CPOBase::memWrite(ip_subnet, buffer_ptr, buffer_size);
	CPOBase::memWrite(ip_gateway, buffer_ptr, buffer_size);
	CPOBase::memWrite(ip_dns_server, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_loopback, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_conf_dhcp, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 NetAdapter::memSize()
{
	i32 len = 0;
	len += sizeof(ip_address);
	len += sizeof(ip_subnet);
	len += sizeof(ip_gateway);
	len += sizeof(ip_dns_server);
	len += sizeof(is_loopback);
	len += sizeof(is_conf_dhcp);
	return len;
}

bool NetAdapter::isValid()
{
	if (!ip_address || !ip_subnet)
	{
		return false;
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
CIPInfo::CIPInfo()
{
	m_server = 0;
	m_cmd_port = 0;
	m_cmd_address = 0;
	m_ivs_port = 0;
	m_data_port = 0;
	m_netadapter_vec.clear();

	m_high_adapter = 0;
	m_high_address = 0;
	m_high_id = 0;
}

i32 CIPInfo::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_server);
	len += sizeof(m_cmd_port);
	len += sizeof(m_cmd_address);
	len += sizeof(m_ivs_port);
	len += sizeof(m_data_port);

	len += sizeof(m_high_adapter);
	len += sizeof(m_high_address);
	len += sizeof(m_high_id);

	i32 i, count = (i32)m_netadapter_vec.size();
	for (i = 0; i < count; i++)
	{
		len += m_netadapter_vec[i].memSize();
	}
	len += sizeof(count);
	return len;
}

i32 CIPInfo::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_server, buffer_ptr, buffer_size);
	CPOBase::memRead(m_cmd_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_cmd_address, buffer_ptr, buffer_size);
	CPOBase::memRead(m_ivs_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_port, buffer_ptr, buffer_size);

	CPOBase::memRead(m_high_adapter, buffer_ptr, buffer_size);
	CPOBase::memRead(m_high_address, buffer_ptr, buffer_size);
	CPOBase::memRead(m_high_id, buffer_ptr, buffer_size);

	m_netadapter_vec.clear();
	i32 i, count = -1;
	CPOBase::memRead(count, buffer_ptr, buffer_size);
	if (CPOBase::isCount(count))
	{
		m_netadapter_vec.resize(count);
		for (i = 0; i < count; i++)
		{
			m_netadapter_vec[i].memRead(buffer_ptr, buffer_size);
		}
	}
	return buffer_ptr - buffer_pos;
}

i32 CIPInfo::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_server, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_cmd_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_cmd_address, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_ivs_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_data_port, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_high_adapter, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_high_address, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_high_id, buffer_ptr, buffer_size);

	i32 i, count = (i32)m_netadapter_vec.size();
	CPOBase::memWrite(count, buffer_ptr, buffer_size);
	for (i = 0; i < count; i++)
	{
		m_netadapter_vec[i].memWrite(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

bool CIPInfo::getFirstMacAddress(postring& mac_addr_str)
{
	i32 i, count = (i32)m_netadapter_vec.size();
	for (i=0; i<count; i++)
	{
		if(!m_netadapter_vec[i].isValid() || m_netadapter_vec[i].is_loopback)
		{
			continue;
		}
		mac_addr_str = m_netadapter_vec[i].mac_address;
		return true;
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////
CIODev::CIODev()
{
	reset();
}

CIODev::~CIODev()
{
}

void CIODev::init()
{
	lock_guard();
	reset();
}

void CIODev::reset()
{
	m_dev_address = PO_MODBUS_ADDRESS;

	m_port_name = PO_MODBUS_PORT;
	m_rs_mode = kPOSerialRs232;
	m_baud_rate = PO_MODBUS_BAUDBAND;
	m_data_bits = PO_MODBUS_DATABITS;
	m_parity = PO_MODBUS_PARITY;
	m_stop_bits = PO_MODBUS_STOPBITS;
	m_flow_control = PO_MODBUS_FLOWCTRL;

	m_time_out = PO_IO_TIMEOUT;
	m_retry_count = PO_IO_RETRY_COUNT;
}

CIODev CIODev::getValue()
{
	lock_guard();
	return *this;
}

void CIODev::setValue(CIODev& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32	CIODev::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_dev_address);

	len += CPOBase::getStringMemSize(m_port_name);
	len += sizeof(m_rs_mode);
	len += sizeof(m_baud_rate);
	len += sizeof(m_data_bits);
	len += sizeof(m_flow_control);
	len += sizeof(m_parity);
	len += sizeof(m_stop_bits);

	len += sizeof(m_time_out);
	len += sizeof(m_retry_count);
	return len;
}

i32 CIODev::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_dev_address, buffer_ptr, buffer_size);

	CPOBase::memWrite(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memWrite(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_parity, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CIODev::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

#if defined(POR_DEVICE)
	i32 tmp_dev_address;
	CPOBase::memRead(tmp_dev_address, buffer_ptr);
#else
	CPOBase::memRead(m_dev_address, buffer_ptr);
#endif

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CIODev::memImport(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_dev_address, buffer_ptr);

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CIODev::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_dev_address, fp);

	CPOBase::fileWrite(fp, m_port_name);
	CPOBase::fileWrite(m_rs_mode, fp);
	CPOBase::fileWrite(m_baud_rate, fp);
	CPOBase::fileWrite(m_data_bits, fp);
	CPOBase::fileWrite(m_flow_control, fp);
	CPOBase::fileWrite(m_parity, fp);
	CPOBase::fileWrite(m_stop_bits, fp);

	CPOBase::fileWrite(m_time_out, fp);
	CPOBase::fileWrite(m_retry_count, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

bool CIODev::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_dev_address, fp);

	CPOBase::fileRead(fp, m_port_name);
	CPOBase::fileRead(m_rs_mode, fp);
	CPOBase::fileRead(m_baud_rate, fp);
	CPOBase::fileRead(m_data_bits, fp);
	CPOBase::fileRead(m_flow_control, fp);
	CPOBase::fileRead(m_parity, fp);
	CPOBase::fileRead(m_stop_bits, fp);

	CPOBase::fileRead(m_time_out, fp);
	CPOBase::fileRead(m_retry_count, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

//////////////////////////////////////////////////////////////////////////
CPlainDevParam::CPlainDevParam()
{
	reset();
}

CPlainDevParam::~CPlainDevParam()
{
}

void CPlainDevParam::reset()
{
	m_used_device = kPOCommNone;

	m_tcp_port = PO_PLAINIO_TCP_PORT;
	m_udp_port = PO_PLAINIO_UDP_PORT;

	m_port_name = PO_PLAINIO_PORT;
	m_rs_mode = kPOSerialRs232;
	m_baud_rate = PO_PLAINIO_BAUDBAND;
	m_data_bits = PO_PLAINIO_DATABITS;
	m_parity = PO_PLAINIO_PARITY;
	m_stop_bits = PO_PLAINIO_STOPBITS;
	m_flow_control = PO_PLAINIO_FLOWCTRL;

	m_time_out = PO_IO_TIMEOUT;
	m_retry_count = PO_IO_RETRY_COUNT;
}

void CPlainDevParam::init()
{
	lock_guard();
	reset();
}

CPlainDevParam CPlainDevParam::getValue()
{
	lock_guard();
	return *this;
}

void CPlainDevParam::setValue(CPlainDevParam& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32	CPlainDevParam::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_used_device);

	len += sizeof(m_tcp_port);
	len += sizeof(m_udp_port);

	len += CPOBase::getStringMemSize(m_port_name);
	len += sizeof(m_rs_mode);
	len += sizeof(m_baud_rate);
	len += sizeof(m_data_bits);
	len += sizeof(m_flow_control);
	len += sizeof(m_parity);
	len += sizeof(m_stop_bits);
	len += sizeof(m_time_out);
	len += sizeof(m_retry_count);
	return len;
}

i32 CPlainDevParam::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_used_device, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memWrite(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memWrite(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_parity, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPlainDevParam::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

#if defined(POR_DEVICE)
	i32 tmp_used_device;
	CPOBase::memRead(tmp_used_device, buffer_ptr, buffer_size);
#else
	CPOBase::memRead(m_used_device, buffer_ptr, buffer_size);
#endif

	CPOBase::memRead(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPlainDevParam::memImport(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_used_device, buffer_ptr, buffer_size);

	CPOBase::memRead(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPlainDevParam::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_used_device, fp);

	CPOBase::fileWrite(m_tcp_port, fp);
	CPOBase::fileWrite(m_udp_port, fp);

	CPOBase::fileWrite(fp, m_port_name);
	CPOBase::fileWrite(m_rs_mode, fp);
	CPOBase::fileWrite(m_baud_rate, fp);
	CPOBase::fileWrite(m_data_bits, fp);
	CPOBase::fileWrite(m_flow_control, fp);
	CPOBase::fileWrite(m_parity, fp);
	CPOBase::fileWrite(m_stop_bits, fp);

	CPOBase::fileWrite(m_time_out, fp);
	CPOBase::fileWrite(m_retry_count, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

bool CPlainDevParam::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_used_device, fp);

	CPOBase::fileRead(m_tcp_port, fp);
	CPOBase::fileRead(m_udp_port, fp);

	CPOBase::fileRead(fp, m_port_name);
	CPOBase::fileRead(m_rs_mode, fp);
	CPOBase::fileRead(m_baud_rate, fp);
	CPOBase::fileRead(m_data_bits, fp);
	CPOBase::fileRead(m_flow_control, fp);
	CPOBase::fileRead(m_parity, fp);
	CPOBase::fileRead(m_stop_bits, fp);

	CPOBase::fileRead(m_time_out, fp);
	CPOBase::fileRead(m_retry_count, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

//////////////////////////////////////////////////////////////////////////
CModbusDevParam::CModbusDevParam()
{
	reset();
}

CModbusDevParam::~CModbusDevParam()
{
}

void CModbusDevParam::reset()
{
	m_used_device = kPOCommNone;

	m_dev_address = PO_MODBUS_ADDRESS;
	m_endian_mode = kPOModbusBigEndian;
	m_fmt_output_digits = 3;

	m_tcp_port = PO_MODNET_TCP_PORT;
	m_udp_port = PO_MODNET_UDP_PORT;

	m_port_name = PO_MODBUS_PORT;
	m_rs_mode = kPOSerialRs232;
	m_baud_rate = PO_MODBUS_BAUDBAND;
	m_data_bits = PO_MODBUS_DATABITS;
	m_parity = PO_MODBUS_PARITY;
	m_stop_bits = PO_MODBUS_STOPBITS;
	m_flow_control = PO_MODBUS_FLOWCTRL;

	m_time_out = PO_IO_TIMEOUT;
	m_retry_count = PO_IO_RETRY_COUNT;
}

void CModbusDevParam::init()
{
	lock_guard();
	reset();
}

CModbusDevParam CModbusDevParam::getValue()
{
	lock_guard();
	return *this;
}

void CModbusDevParam::setValue(CModbusDevParam& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32	CModbusDevParam::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_used_device);

	len += sizeof(m_dev_address);
	len += sizeof(m_fmt_output_digits);
	len += sizeof(m_endian_mode);

	len += sizeof(m_tcp_port);
	len += sizeof(m_udp_port);

	len += CPOBase::getStringMemSize(m_port_name);
	len += sizeof(m_rs_mode);
	len += sizeof(m_baud_rate);
	len += sizeof(m_data_bits);
	len += sizeof(m_flow_control);
	len += sizeof(m_parity);
	len += sizeof(m_stop_bits);
	len += sizeof(m_time_out);
	len += sizeof(m_retry_count);
	return len;
}

i32 CModbusDevParam::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;
	
	CPOBase::memWrite(m_used_device, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_dev_address, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_fmt_output_digits, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_endian_mode, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memWrite(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memWrite(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_parity, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CModbusDevParam::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;
	
#if defined(POR_DEVICE)
	i32 tmp_used_device, tmp_dev_address;
	CPOBase::memRead(tmp_used_device, buffer_ptr, buffer_size);
	CPOBase::memRead(tmp_dev_address, buffer_ptr, buffer_size);
#else
	CPOBase::memRead(m_used_device, buffer_ptr, buffer_size);
	CPOBase::memRead(m_dev_address, buffer_ptr, buffer_size);
#endif

	CPOBase::memRead(m_fmt_output_digits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_endian_mode, buffer_ptr, buffer_size);

	CPOBase::memRead(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CModbusDevParam::memImport(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;
	
	CPOBase::memRead(m_used_device, buffer_ptr, buffer_size);

	CPOBase::memRead(m_dev_address, buffer_ptr, buffer_size);
	CPOBase::memRead(m_fmt_output_digits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_endian_mode, buffer_ptr, buffer_size);

	CPOBase::memRead(m_tcp_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_udp_port, buffer_ptr, buffer_size);

	CPOBase::memRead(buffer_ptr, buffer_size, m_port_name);
	CPOBase::memRead(m_rs_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_baud_rate, buffer_ptr, buffer_size);
	CPOBase::memRead(m_data_bits, buffer_ptr, buffer_size);
	CPOBase::memRead(m_flow_control, buffer_ptr, buffer_size);
	CPOBase::memRead(m_parity, buffer_ptr, buffer_size);
	CPOBase::memRead(m_stop_bits, buffer_ptr, buffer_size);

	CPOBase::memRead(m_time_out, buffer_ptr, buffer_size);
	CPOBase::memRead(m_retry_count, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

CIODev CModbusDevParam::getIODevParam()
{
	lock_guard();
	CIODev io_dev_param;

	io_dev_param.m_dev_address = m_dev_address;

	io_dev_param.m_port_name = m_port_name;
	io_dev_param.m_rs_mode = m_rs_mode;
	io_dev_param.m_baud_rate = m_baud_rate;
	io_dev_param.m_data_bits = m_data_bits;
	io_dev_param.m_flow_control = m_flow_control;
	io_dev_param.m_parity = m_parity;
	io_dev_param.m_stop_bits = m_stop_bits;

	io_dev_param.m_time_out = m_time_out;
	io_dev_param.m_retry_count = m_retry_count;
	return io_dev_param;
}

bool CModbusDevParam::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_used_device, fp);

	CPOBase::fileWrite(m_dev_address, fp);
	CPOBase::fileWrite(m_fmt_output_digits, fp);
	CPOBase::fileWrite(m_endian_mode, fp);

	CPOBase::fileWrite(m_tcp_port, fp);
	CPOBase::fileWrite(m_udp_port, fp);

	CPOBase::fileWrite(fp, m_port_name);
	CPOBase::fileWrite(m_rs_mode, fp);
	CPOBase::fileWrite(m_baud_rate, fp);
	CPOBase::fileWrite(m_data_bits, fp);
	CPOBase::fileWrite(m_flow_control, fp);
	CPOBase::fileWrite(m_parity, fp);
	CPOBase::fileWrite(m_stop_bits, fp);

	CPOBase::fileWrite(m_time_out, fp);
	CPOBase::fileWrite(m_retry_count, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

bool CModbusDevParam::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_used_device, fp);

	CPOBase::fileRead(m_dev_address, fp);
	CPOBase::fileRead(m_fmt_output_digits, fp);
	CPOBase::fileRead(m_endian_mode, fp);

	CPOBase::fileRead(m_tcp_port, fp);
	CPOBase::fileRead(m_udp_port, fp);

	CPOBase::fileRead(fp, m_port_name);
	CPOBase::fileRead(m_rs_mode, fp);
	CPOBase::fileRead(m_baud_rate, fp);
	CPOBase::fileRead(m_data_bits, fp);
	CPOBase::fileRead(m_flow_control, fp);
	CPOBase::fileRead(m_parity, fp);
	CPOBase::fileRead(m_stop_bits, fp);

	CPOBase::fileRead(m_time_out, fp);
	CPOBase::fileRead(m_retry_count, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

//////////////////////////////////////////////////////////////////////////
COpcDev::COpcDev()
{
	reset();
}

COpcDev::~COpcDev()
{

}

void COpcDev::init()
{
	lock_guard();
	reset();
}

void COpcDev::reset()
{
	m_is_used = false;
	m_port = PO_OPC_PORT;
	m_interval = 100; //100ms
}

COpcDev COpcDev::getValue()
{
	lock_guard();
	return *this;
}

void COpcDev::setValue(COpcDev& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 COpcDev::memSize()
{
	lock_guard();
	i32 len = 0;
	len += sizeof(m_is_used);
	len += sizeof(m_port);
	len += sizeof(m_interval);
	return len;
}

i32 COpcDev::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_is_used, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_interval, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 COpcDev::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_is_used, buffer_ptr, buffer_size);
	CPOBase::memRead(m_port, buffer_ptr, buffer_size);
	CPOBase::memRead(m_interval, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool COpcDev::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_is_used, fp);
	CPOBase::fileRead(m_port, fp);
	CPOBase::fileRead(m_interval, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool COpcDev::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	CPOBase::fileWrite(m_is_used, fp);
	CPOBase::fileWrite(m_port, fp);
	CPOBase::fileWrite(m_interval, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CFtpDev::CFtpDev()
{
	reset();
}

CFtpDev::~CFtpDev()
{
}

void CFtpDev::init()
{
	lock_guard();
	reset();
}

void CFtpDev::reset()
{
	m_ftp_hostname = PO_FTP_HOSTNAME;
	m_ftp_username = PO_FTP_USERNAME;
	m_ftp_password = PO_FTP_PASSWORD;
	m_ftp_port = PO_FTP_PORT;
}

CFtpDev CFtpDev::getValue()
{
	lock_guard();
	return *this;
}

void CFtpDev::setValue(CFtpDev& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CFtpDev::memSize()
{
	lock_guard();
	i32 len = 0;

	len += CPOBase::getStringMemSize(m_ftp_hostname);
	len += CPOBase::getStringMemSize(m_ftp_username);
	len += CPOBase::getStringMemSize(m_ftp_password);
	len += sizeof(m_ftp_port);
	return len;
}

i32 CFtpDev::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(buffer_ptr, buffer_size, m_ftp_hostname);
	CPOBase::memWrite(buffer_ptr, buffer_size, m_ftp_username);
	CPOBase::memWrite(buffer_ptr, buffer_size, m_ftp_password);
	CPOBase::memWrite(m_ftp_port, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CFtpDev::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(buffer_ptr, buffer_size, m_ftp_hostname);
	CPOBase::memRead(buffer_ptr, buffer_size, m_ftp_username);
	CPOBase::memRead(buffer_ptr, buffer_size, m_ftp_password);
	CPOBase::memRead(m_ftp_port, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CFtpDev::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(fp, m_ftp_hostname);
	CPOBase::fileRead(fp, m_ftp_username);
	CPOBase::fileRead(fp, m_ftp_password);
	CPOBase::fileRead(m_ftp_port, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CFtpDev::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	
	CPOBase::fileWrite(fp, m_ftp_hostname);
	CPOBase::fileWrite(fp, m_ftp_username);
	CPOBase::fileWrite(fp, m_ftp_password);
	CPOBase::fileWrite(m_ftp_port, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CFTPDevGroup::CFTPDevGroup()
{
}

CFTPDevGroup::~CFTPDevGroup()
{
}

void CFTPDevGroup::init()
{
	lock_guard();
	for (i32 i = 0; i < POFTP_MAX_DEV; i++)
	{
		m_ftp_dev_param[i].init();
	}
}

i32 CFTPDevGroup::memSize()
{
	lock_guard();
	i32 len = 0;
	i32 i, count = POFTP_MAX_DEV;
	for (i = 0; i < count; i++)
	{
		len += m_ftp_dev_param[i].memSize();
	}
	len += sizeof(count);
	return len;
}

i32 CFTPDevGroup::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	i32 i, count = POFTP_MAX_DEV;
	CPOBase::memWrite(count, buffer_ptr, buffer_size);
	for (i = 0; i < count; i++)
	{
		m_ftp_dev_param[i].memWrite(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

i32 CFTPDevGroup::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	i32 i, count = -1;
	CPOBase::memRead(count, buffer_ptr, buffer_size);
	if (!CPOBase::checkRange(count, POFTP_MAX_DEV))
	{
		return buffer_ptr - buffer_pos;
	}

	for (i = 0; i < count; i++)
	{
		m_ftp_dev_param[i].memRead(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;

}

bool CFTPDevGroup::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	i32 i, count = -1;
	CPOBase::fileRead(count, fp);
	if (!CPOBase::checkRange(count, POFTP_MAX_DEV))
	{
		return false;
	}

	for (i = 0; i < count; i++)
	{
		m_ftp_dev_param[i].fileRead(fp);
	}
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CFTPDevGroup::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	
	i32 i, count = POFTP_MAX_DEV;
	CPOBase::fileWrite(count, fp);

	for (i = 0; i < count; i++)
	{
		m_ftp_dev_param[i].fileWrite(fp);
	}
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
DBConfig::DBConfig()
{
	init();
}

void DBConfig::init()
{
#if defined(POR_WITH_MYSQL)
	host_name = PO_MYSQL_HOSTNAME;
#elif defined(POR_WITH_SQLITE)
	host_name = PO_DATABASE_FILENAME;
#endif

	database_name = PO_MYSQL_DATABASE;
	username = PO_MYSQL_USERNAME;
	password = PO_MYSQL_PASSWORD;
	remote_username = PO_MYSQL_REMOTEUSERNAME;
	remote_password = PO_MYSQL_REMOTEPASSWORD;
	db_port = PO_MYSQL_PORT;

	limit_log = PO_MYSQL_LOGLIMIT;
	limit_record = PO_MYSQL_RECLIMIT;
}

i32 DBConfig::memSize()
{
	i32 len = 0;
	len += CPOBase::getStringMemSize(host_name);
	len += CPOBase::getStringMemSize(database_name);
	len += CPOBase::getStringMemSize(remote_username);
	len += CPOBase::getStringMemSize(remote_password);
	len += sizeof(db_port);
	len += sizeof(limit_log);
	len += sizeof(limit_record);
	return len;
}

i32 DBConfig::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(buffer_ptr, buffer_size, host_name);
	CPOBase::memWrite(buffer_ptr, buffer_size, database_name);
	CPOBase::memWrite(buffer_ptr, buffer_size, remote_username);
	CPOBase::memWrite(buffer_ptr, buffer_size, remote_password);
	CPOBase::memWrite(db_port, buffer_ptr, buffer_size);
	CPOBase::memWrite(limit_log, buffer_ptr, buffer_size);
	CPOBase::memWrite(limit_record, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 DBConfig::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(buffer_ptr, buffer_size, host_name);
	CPOBase::memRead(buffer_ptr, buffer_size, database_name);
	CPOBase::memRead(buffer_ptr, buffer_size, remote_username);
	CPOBase::memRead(buffer_ptr, buffer_size, remote_password);
	CPOBase::memRead(db_port, buffer_ptr, buffer_size);
	CPOBase::memRead(limit_log, buffer_ptr, buffer_size);
	CPOBase::memRead(limit_record, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool DBConfig::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(fp, host_name);
	CPOBase::fileRead(fp, database_name);
	CPOBase::fileRead(fp, remote_username);
	CPOBase::fileRead(fp, remote_password);
	CPOBase::fileRead(db_port, fp);
	CPOBase::fileRead(limit_log, fp);
	CPOBase::fileRead(limit_record, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool DBConfig::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(fp, host_name);
	CPOBase::fileWrite(fp, database_name);
	CPOBase::fileWrite(fp, remote_username);
	CPOBase::fileWrite(fp, remote_password);
	CPOBase::fileWrite(db_port, fp);
	CPOBase::fileWrite(limit_log, fp);
	CPOBase::fileWrite(limit_record, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
AlarmQueue::AlarmQueue()
{
	read_pos = write_pos = 0;
}

bool AlarmQueue::getAlarmLog(POAlarm& alarm)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	read_pos = (read_pos + 1) % PODB_MAX_ALARM;
	alarm = queue[read_pos];
	return true;
}

bool AlarmQueue::getAlarmLogVec(POAlarmVec& alarm_vec)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	alarm_vec.clear();
	while (read_pos != write_pos)
	{
		read_pos = (read_pos + 1) % PODB_MAX_ALARM;
		alarm_vec.push_back(queue[read_pos]);
	}
	return true;
}

bool AlarmQueue::setAlarmLog(const POAlarm& alarm)
{
	write_pos = (write_pos + 1) % PODB_MAX_ALARM;
	if (write_pos == read_pos)
	{
		return false;
	}

	queue[write_pos] = alarm;
	return true;
}

//////////////////////////////////////////////////////////////////////////
ActionQueue::ActionQueue()
{
	read_pos = 0;
	write_pos = 0;
}

bool ActionQueue::getActionLog(POAction& action)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	read_pos = (read_pos + 1) % PODB_MAX_ACTION;
	action = queue[read_pos];
	return true;
}

bool ActionQueue::getActionLogVec(POActionVec& action_vec)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	action_vec.clear();
	while (read_pos != write_pos)
	{
		read_pos = (read_pos + 1) % PODB_MAX_ACTION;
		action_vec.push_back(queue[read_pos]);
	}
	return true;
}

bool ActionQueue::setActionLog(const POAction& action)
{
	write_pos = (write_pos + 1) % PODB_MAX_ACTION;
	if (write_pos == read_pos)
	{
		return false;
	}

	queue[write_pos] = action;
	return true;
}

//////////////////////////////////////////////////////////////////////////
InfoQueue::InfoQueue()
{
	read_pos = 0;
	write_pos = 0;
}

bool InfoQueue::getInfoLog(POInfo& Info)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	read_pos = (read_pos + 1) % PODB_MAX_INFO;
	Info = queue[read_pos];
	return true;
}

bool InfoQueue::getInfoLogVec(POInfoVec& Info_vec)
{
	if (read_pos == write_pos)
	{
		return false;
	}

	Info_vec.clear();
	while (read_pos != write_pos)
	{
		read_pos = (read_pos + 1) % PODB_MAX_INFO;
		Info_vec.push_back(queue[read_pos]);
	}
	return true;
}

bool InfoQueue::setInfoLog(const POInfo& Info)
{
	write_pos = (write_pos + 1) % PODB_MAX_INFO;
	if (write_pos == read_pos)
	{
		return false;
	}

	queue[write_pos] = Info;
	return true;
}

//////////////////////////////////////////////////////////////////////////
POLineFormat::POLineFormat()
{
	line_width = kPOLineWidthAuto;
	line_style = kPOLineStyleAuto;
	line_color = kPOColorAuto;
}

POLineFormat::POLineFormat(i32 width, i32 style, i32 color)
{
	line_width = width;
	line_style = style;
	line_color = color;
}

//////////////////////////////////////////////////////////////////////////
CPOArc::CPOArc()
{
	init();
}

CPOArc::~CPOArc()
{
}

void CPOArc::init()
{
	m_center = vector2df();
	m_radius = 0;
	m_st_angle = 0;
	m_angle_len = 0;
	memset(&m_format, 0, sizeof(m_format));
}

void CPOArc::setValue(const CPOArc& other)
{
	*this = other;
}

void CPOArc::setValue(CPOArc* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	*this = *other_ptr;
}

i32 CPOArc::memSize()
{
	i32 len = 0;
	len += sizeof(m_center);
	len += sizeof(m_radius);
	len += sizeof(m_st_angle);
	len += sizeof(m_angle_len);
	len += sizeof(m_format);
	return len;
}

i32 CPOArc::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_center, buffer_ptr, buffer_size);
	CPOBase::memRead(m_radius, buffer_ptr, buffer_size);
	CPOBase::memRead(m_st_angle, buffer_ptr, buffer_size);
	CPOBase::memRead(m_angle_len, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPOArc::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_center, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_radius, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_st_angle, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_angle_len, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPOArc::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_center, fp);
	CPOBase::fileRead(m_radius, fp);
	CPOBase::fileRead(m_st_angle, fp);
	CPOBase::fileRead(m_angle_len, fp);
	CPOBase::fileRead(m_format, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_CODE);
}

bool CPOArc::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_center, fp);
	CPOBase::fileWrite(m_radius, fp);
	CPOBase::fileWrite(m_st_angle, fp);
	CPOBase::fileWrite(m_angle_len, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CPOCircle::CPOCircle()
{
	init();
}

CPOCircle::~CPOCircle()
{
}

void CPOCircle::init()
{
	m_center = vector2df();
	m_radius = 0;
	memset(&m_format, 0, sizeof(m_format));
}

void CPOCircle::setValue(const CPOCircle& other)
{
	*this = other;
}

void CPOCircle::setValue(CPOCircle* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	*this = *other_ptr;
}

i32 CPOCircle::memSize()
{
	i32 len = 0;
	len += sizeof(m_center);
	len += sizeof(m_radius);
	len += sizeof(m_format);
	return len;
}

i32 CPOCircle::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_center, buffer_ptr, buffer_size);
	CPOBase::memRead(m_radius, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPOCircle::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_center, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_radius, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPOCircle::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_center, fp);
	CPOBase::fileRead(m_radius, fp);
	CPOBase::fileRead(m_format, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_CODE);
}

bool CPOCircle::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_center, fp);
	CPOBase::fileWrite(m_radius, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CPOLine::CPOLine()
{
	init();
}

CPOLine::~CPOLine()
{
}

void CPOLine::init()
{
	m_st_point = vector2df();
	m_ed_point = vector2df();
	memset(&m_format, 0, sizeof(m_format));
}

void CPOLine::setValue(const CPOLine& other)
{
	*this = other;
}

void CPOLine::setValue(CPOLine* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	*this = *other_ptr;
}

i32 CPOLine::memSize()
{
	i32 len = 0;
	len += sizeof(m_st_point);
	len += sizeof(m_ed_point);
	len += sizeof(m_format);
	return len;
}

i32 CPOLine::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_st_point, buffer_ptr, buffer_size);
	CPOBase::memRead(m_ed_point, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPOLine::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_st_point, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_ed_point, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPOLine::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_st_point, fp);
	CPOBase::fileRead(m_ed_point, fp);
	CPOBase::fileRead(m_format, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_CODE);
}

bool CPOLine::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_st_point, fp);
	CPOBase::fileWrite(m_ed_point, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CPOCross::CPOCross()
{
	init();
}

CPOCross::~CPOCross()
{
}

void CPOCross::init()
{
	m_point = vector2df();
	m_angle = 0;
	m_length = 0;
	memset(&m_format, 0, sizeof(m_format));
}

void CPOCross::setValue(const CPOCross& other)
{
	*this = other;
}

void CPOCross::setValue(CPOCross* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	*this = *other_ptr;
}

i32 CPOCross::memSize()
{
	i32 len = 0;
	len += sizeof(m_point);
	len += sizeof(m_angle);
	len += sizeof(m_length);
	len += sizeof(m_format);
	return len;
}

i32 CPOCross::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_point, buffer_ptr, buffer_size);
	CPOBase::memRead(m_angle, buffer_ptr, buffer_size);
	CPOBase::memRead(m_length, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPOCross::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_point, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_angle, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_length, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPOCross::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_point, fp);
	CPOBase::fileRead(m_angle, fp);
	CPOBase::fileRead(m_length, fp);
	CPOBase::fileRead(m_format, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_CODE);
}

bool CPOCross::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_point, fp);
	CPOBase::fileWrite(m_angle, fp);
	CPOBase::fileWrite(m_length, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CPORotatedRect::CPORotatedRect()
{
	init();
}

CPORotatedRect::~CPORotatedRect()
{
}

void CPORotatedRect::init()
{
	m_rect.reset();
	m_angle = 0;
	memset(&m_format, 0, sizeof(m_format));
}

void CPORotatedRect::setValue(const CPORotatedRect& other)
{
	*this = other;
}

void CPORotatedRect::setValue(CPORotatedRect* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	*this = *other_ptr;
}

i32 CPORotatedRect::memSize()
{
	i32 len = 0;
	len += sizeof(m_rect);
	len += sizeof(m_angle);
	len += sizeof(m_format);
	return len;
}

i32 CPORotatedRect::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_rect, buffer_ptr, buffer_size);
	CPOBase::memRead(m_angle, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPORotatedRect::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_rect, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_angle, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPORotatedRect::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_rect, fp);
	CPOBase::fileRead(m_angle, fp);
	CPOBase::fileRead(m_format, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_CODE);
}

bool CPORotatedRect::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_rect, fp);
	CPOBase::fileWrite(m_angle, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CPOContours::CPOContours()
{
	init();
}

CPOContours::~CPOContours()
{
	freeBuffer();
}

void CPOContours::init(i32 reseved)
{
	m_format_vec.clear();
	m_is_closed_vec.clear();
	m_info_vec.clear();
	m_contour_vec.clear();

	if (reseved > 1)
	{
		m_format_vec.reserve(reseved);
		m_is_closed_vec.reserve(reseved);
		m_info_vec.resize(reseved);
		m_contour_vec.reserve(reseved);
	}
}

void CPOContours::freeBuffer()
{
	m_format_vec.clear();
	m_is_closed_vec.clear();
	m_info_vec.clear();
	m_contour_vec.clear();
}

void CPOContours::resize(i32 count)
{
	m_format_vec.resize(count);
	m_is_closed_vec.resize(count);
	m_info_vec.resize(count);
	m_contour_vec.resize(count);
}

void CPOContours::initContours(i32 count)
{
	freeBuffer();
	if (!CPOBase::isCount(count))
	{
		return;
	}

	m_format_vec.resize(count);
	m_is_closed_vec.resize(count);
	m_info_vec.resize(count);
	m_contour_vec.resize(count);
}

i32 CPOContours::getContoursCount()
{
	return (i32)m_contour_vec.size();
}

i32 CPOContours::getContourLength()
{
	i32 len = 0;
	i32 i, count = (i32)m_contour_vec.size();
	for (i = 0; i < count; i++)
	{
		len += (i32)m_contour_vec[i].size();
	}
	return len;
}

i32 CPOContours::getContourSize(i32 index)
{
	if (!CPOBase::checkIndex(index, (i32)m_contour_vec.size()))
	{
		return 0;
	}
	return (i32)m_contour_vec[index].size();
}

ptvector2df* CPOContours::getContour(i32 index)
{
	if (!CPOBase::checkIndex(index, (i32)m_contour_vec.size()))
	{
		return NULL;
	}
	return m_contour_vec.data()+ index;
}

void CPOContours::setValue(const CPOContours& other)
{
	m_format_vec = other.m_format_vec;
	m_is_closed_vec = other.m_is_closed_vec;
	m_info_vec= other.m_info_vec;
	m_contour_vec = other.m_contour_vec;
}

void CPOContours::setValue(CPOContours* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}
	m_format_vec = other_ptr->m_format_vec;
	m_is_closed_vec = other_ptr->m_is_closed_vec;
	m_info_vec = other_ptr->m_info_vec;
	m_contour_vec = other_ptr->m_contour_vec;
}

void CPOContours::addContour(bool is_closed, const vector2df* pt_ptr, i32 count)
{
	m_format_vec.push_back(POLineFormat());
	m_is_closed_vec.push_back(is_closed);
	m_info_vec.push_back(0);

	ptvector2df* pt_vec_ptr = CPOBase::pushBackNew(m_contour_vec);
	if (CPOBase::isPositive(count))
	{
		pt_vec_ptr->resize(count);
		CPOBase::memCopy(pt_vec_ptr->data(), pt_ptr, count);
	}
}

void CPOContours::addContour(const vector2df* pt_ptr, i32 count, bool is_closed, i32 info,
						i32 width, i32 style, i32 color)
{
	m_format_vec.push_back(POLineFormat(width, style, color));
	m_is_closed_vec.push_back(is_closed);
	m_info_vec.push_back(info);

	ptvector2df* pt_vec_ptr = CPOBase::pushBackNew(m_contour_vec);
	if (CPOBase::isPositive(count))
	{
		pt_vec_ptr->resize(count);
		CPOBase::memCopy(pt_vec_ptr->data(), pt_ptr, count);
	}
}

i32	CPOContours::memSize()
{
	i32 len = 0;
	len += CPOBase::getVectorMemSize(m_format_vec);
	len += CPOBase::getVectorMemSize(m_is_closed_vec);
	len += CPOBase::getVectorMemSize(m_info_vec);

	i32 i, count = (i32)m_contour_vec.size();
	for (i = 0; i < count; i++)
	{
		len += CPOBase::getVectorMemSize(m_contour_vec[i]);
	}
	len += sizeof(count);
	return len;
}

i32	CPOContours::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	freeBuffer();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memReadVector(m_format_vec, buffer_ptr, buffer_size);
	CPOBase::memReadVector(m_is_closed_vec, buffer_ptr, buffer_size);
	CPOBase::memReadVector(m_info_vec, buffer_ptr, buffer_size);

	i32 i, count = -1;
	CPOBase::memRead(count, buffer_ptr, buffer_size);
	if (CPOBase::isCount(count))
	{
		m_contour_vec.resize(count);
		for (i = 0; i < count; i++)
		{
			if (!CPOBase::memReadVector(m_contour_vec[i], buffer_ptr, buffer_size))
			{
				m_contour_vec.clear();
				return buffer_ptr - buffer_pos;
			}
		}
	}
	return buffer_ptr - buffer_pos;
}

i32	CPOContours::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memWriteVector(m_format_vec, buffer_ptr, buffer_size);
	CPOBase::memWriteVector(m_is_closed_vec, buffer_ptr, buffer_size);
	CPOBase::memWriteVector(m_info_vec, buffer_ptr, buffer_size);

	i32 i, count = (i32)m_contour_vec.size();
	CPOBase::memWrite(count, buffer_ptr, buffer_size);

	for (i = 0; i < count; i++)
	{
		if (!CPOBase::memWriteVector(m_contour_vec[i], buffer_ptr, buffer_size))
		{
			return buffer_ptr - buffer_pos;
		}
	}
	return buffer_ptr - buffer_pos;
}

bool CPOContours::fileRead(FILE* fp)
{
	freeBuffer();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileReadVector(m_format_vec, fp);
	CPOBase::fileReadVector(m_is_closed_vec, fp);
	CPOBase::fileReadVector(m_info_vec, fp);

	i32 i, count;
	CPOBase::fileRead(count, fp);
	if (CPOBase::isCount(count))
	{
		m_contour_vec.resize(count);
		for (i = 0; i < count; i++)
		{
			if (!CPOBase::fileReadVector(m_contour_vec[i], fp))
			{
				m_contour_vec.clear();
				return false;
			}
		}
	}
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CPOContours::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	CPOBase::fileWriteVector(m_format_vec, fp);
	CPOBase::fileWriteVector(m_is_closed_vec, fp);
	CPOBase::fileWriteVector(m_info_vec, fp);

	i32 i, count = (i32)m_contour_vec.size();
	CPOBase::fileWrite(count, fp);
	for (i = 0; i < count; i++)
	{
		CPOBase::fileWriteVector(m_contour_vec[i], fp);
	}

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////	
CPOShape::CPOShape()
{
	memset(this, 0, sizeof(CPOShape));
}

CPOShape::CPOShape(f32 x1, f32 y1, f32 x2, f32 y2) //edge
{
	m_shape_mode = kPOShapeEdge;
	memset(&m_format, 0, sizeof(m_format));
	d._edge.x1 = x1;
	d._edge.y1 = y1;
	d._edge.x2 = x2;
	d._edge.y2 = y2;
}

CPOShape::CPOShape(f32 x0, f32 y0, f32 dx, f32 dy, f32 len) //line
{
	m_shape_mode = kPOShapeLine;
	memset(&m_format, 0, sizeof(m_format));
	d._line.x0 = x0;
	d._line.y0 = y0;
	d._line.dx = dx;
	d._line.dy = dy;
	d._line.len = len;
}

CPOShape::CPOShape(f32 cx, f32 cy, f32 r) //circle
{
	m_shape_mode = kPOShapeCircle;
	memset(&m_format, 0, sizeof(m_format));
	d._circle.cx = cx;
	d._circle.cy = cy;
	d._circle.r = r;
	d._circle.st_angle = 0;
	d._circle.angle_len = PO_PI2;
}

void CPOShape::init()
{
	m_shape_mode = kPOShapeNone;
	memset(&m_format, 0, sizeof(m_format));
	memset(&d, 0, sizeof(d));
}

void CPOShape::setEdge(vector2df st_pos, vector2df ed_pos)
{
	m_shape_mode = kPOShapeEdge;
	d._edge.x1 = st_pos.x;
	d._edge.y1 = st_pos.y;
	d._edge.x2 = ed_pos.x;
	d._edge.y2 = ed_pos.y;
}

void CPOShape::setLine(vector2df line_pos, vector2df line_dir, f32 len)
{
	m_shape_mode = kPOShapeLine;
	d._line.x0 = line_pos.x;
	d._line.y0 = line_pos.y;
	d._line.dx = line_dir.x;
	d._line.dy = line_dir.y;
	d._line.len = len;
}

void CPOShape::setCircle(vector2df center_pos, f32 radius, f32 st_angle, f32 angle_len)
{
	m_shape_mode = kPOShapeCircle;
	d._circle.cx = center_pos.x;
	d._circle.cy = center_pos.y;
	d._circle.r = radius;
	d._circle.st_angle = st_angle;
	d._circle.angle_len = angle_len;
}

bool CPOShape::getEdge(vector2df& st_pos, vector2df& ed_pos) const
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			st_pos = vector2df(d._line.x0, d._line.y0);
			ed_pos = st_pos + vector2df(d._line.dx, d._line.dy)*d._line.len;
			break;
		}
		case kPOShapeEdge:
		{
			st_pos = vector2df(d._edge.x1, d._edge.y1);
			ed_pos = vector2df(d._edge.x2, d._edge.y2);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CPOShape::getCircle(vector2df& center_pos, f32& radius) const
{
	if (m_shape_mode != kPOShapeCircle)
	{
		return false;
	}
	center_pos = vector2df(d._circle.cx, d._circle.cy);
	radius = d._circle.r;
	return true;
}

i32 CPOShape::memSize()
{
	i32 len = 0;
	len += sizeof(m_shape_mode);
	len += sizeof(m_format);
	len += sizeof(d);
	return len;
}

i32 CPOShape::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memRead(m_shape_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_format, buffer_ptr, buffer_size);
	CPOBase::memRead(d, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CPOShape::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memWrite(m_shape_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_format, buffer_ptr, buffer_size);
	CPOBase::memWrite(d, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CPOShape::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}
	CPOBase::fileRead(m_shape_mode, fp);
	CPOBase::fileRead(m_format, fp);
	CPOBase::fileRead(d, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CPOShape::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	CPOBase::fileWrite(m_shape_mode, fp);
	CPOBase::fileWrite(m_format, fp);
	CPOBase::fileWrite(d, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

void CPOShape::updateCenter(vector2df center_offset)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			d._line.x0 += center_offset.x;
			d._line.y0 += center_offset.y;
			break;
		}
		case kPOShapeCircle:
		{
			d._circle.cx += center_offset.x;
			d._circle.cy += center_offset.y;
			break;
		}
	}
}

void CPOShape::updateShape(f32 min_dist, f32 max_dist)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			vector2df dir(d._line.dx, d._line.dy);
			vector2df pos2d(d._line.x0, d._line.y0);
			pos2d = pos2d + dir* min_dist;
			d._line.x0 = pos2d.x;
			d._line.y0 = pos2d.y;
			d._line.len = max_dist - min_dist;
			break;
		}
		case kPOShapeCircle:
		{
			d._circle.st_angle = min_dist;
			d._circle.angle_len = CPOBase::getAngleLen(min_dist, max_dist);
			break;
		}
	}
}

f32 CPOShape::distance(vector2df pos2d)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			pos2d = pos2d - vector2df(d._line.x0, d._line.y0);
			return CPOBase::distPt2Line(pos2d.x, pos2d.y, vector2df(d._line.dx, d._line.dy));
		}
		case kPOShapeCircle:
		{
			return std::abs((pos2d - vector2df(d._circle.cx, d._circle.cy)).getLength() - d._circle.r);
		}
	}
	return 0;
}

f32 CPOShape::distance(vector2df pos2d, f32& dist)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			vector2df dir(d._line.dx, d._line.dy);
			pos2d = pos2d - vector2df(d._line.x0, d._line.y0);
			dist = CPOBase::distPtInLine(pos2d.x, pos2d.y, dir);
			return CPOBase::distPt2Line(pos2d.x, pos2d.y, dir);
		}
		case kPOShapeCircle:
		{
			vector2df tmp2d(pos2d - vector2df(d._circle.cx, d._circle.cy));
			dist = CPOBase::getAngle(tmp2d);
			return std::abs(tmp2d.getLength() - d._circle.r);
		}
	}
	return 0;
}

f32 CPOShape::signedDistance(vector2df pos2d)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			pos2d = pos2d - vector2df(d._line.x0, d._line.y0);
			return CPOBase::signedDisPt2Line(pos2d.x, pos2d.y, vector2df(d._line.dx, d._line.dy));
		}
		case kPOShapeCircle:
		{
			vector2df tmp2d(pos2d - vector2df(d._circle.cx, d._circle.cy));
			return tmp2d.getLength() - d._circle.r;
		}
	}
	return 0;
}

f32 CPOShape::signedDistance(vector2df pos2d, f32& dist)
{
	switch (m_shape_mode)
	{
		case kPOShapeLine:
		{
			vector2df dir(d._line.dx, d._line.dy);
			pos2d = pos2d - vector2df(d._line.x0, d._line.y0);
			dist = CPOBase::distPtInLine(pos2d.x, pos2d.y, dir);
			return CPOBase::signedDisPt2Line(pos2d.x, pos2d.y, dir);
		}
		case kPOShapeCircle:
		{
			vector2df tmp2d(pos2d - vector2df(d._circle.cx, d._circle.cy));
			dist = CPOBase::getAngle(tmp2d);
			return tmp2d.getLength() - d._circle.r;
		}
	}
	return 0;
}

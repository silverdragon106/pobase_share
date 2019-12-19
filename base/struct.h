#pragma once

#include "define.h"
#include "lock_guide.h"
#include "struct/rect.h"
#include "struct/vector2d.h"
#include "struct/vector3d.h"
#include "struct/image.h"

struct DateTime
{
	u16					yy;
	u8					mm;
	u8					dd;
	u8					h;
	u8					m;
	u8					s;
	u16					ms;

public:
	DateTime();
	DateTime(u16 _yy, u8 _mm, u8 _dd, u8 _h = 0, u8 _m = 0, u8 _s = 0, u16 _ms = 0);

	void				init();

	bool				isEqual(const DateTime& r_datetime) const;
	void				setDateTime(u16 _yy, u8 _mm, u8 _dd, u8 _h = 0, u8 _m = 0, u8 _s = 0, u16 _ms = 0);
	postring			toString(i32 id = -1);
};

class BlobData
{
public:
	BlobData();
	~BlobData();

	void				initBuffer(const u8* buffer_ptr, i32 size);
	void				setBuffer(u8* buffer_ptr, i32 size);
	void				freeBuffer();
	void				clone(const BlobData& other);

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

	bool				isEmpty() const { return m_blob_size == 0 || m_blob_data_ptr == NULL; };
	u8*					getBuffer() const { return m_blob_data_ptr; };
	i32					getBufferSize() const { return m_blob_size; };

public:
	u8*					m_blob_data_ptr;
	i32					m_blob_size;
	bool				m_is_external;
};
typedef std::vector<BlobData> BlobVec;

struct UpdateInfo
{
	bool				is_update;
	bool				is_update_ready;
	i32					update_from;
	i32					update_packet_id;
	i32					read_size;
	i32					read_id;

	postring			model_name;
	postring			update_version;
	strvector			extra_key;
	strvector			extra_value;
	strvector			lowlevel_file_vec;
	strvector			lowlevel_dir_vec;
	strvector			highlevel_file_vec;
	strvector			highlevel_dir_vec;
	u8					update_compatibility;
	bool				is_update_highlevel;
	i32					filesize[2];

public:
	UpdateInfo();
	~UpdateInfo();

	void				init();

	inline bool			isUpdate() { return is_update; };
	inline bool			isUpdateReady() { return is_update_ready; };
	inline i32			getCompatibility() { return update_compatibility; };

	inline i32			getFrom() { return update_from; };
	inline i32			getCurPacketID() { return update_packet_id; };
	inline void			updatePacketID() { update_packet_id++; };
};

struct DeviceInfo
{
	i32					device_id;
	postring			device_name;
	postring			device_version;
	postring			model_name;
	bool				is_hl_embedded;
	bool				is_emulator;
	bool				is_auto_update;

	i32					comm_port;
	i32					video_encoder;
	DateTime			build_date;

public:
	DeviceInfo();
	~DeviceInfo();

	void				import(DeviceInfo& other);
	bool				isCompatibility(DeviceInfo& other);

	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);

	inline i32			getDeviceID() { return device_id; };
	inline postring&	getDeviceName() { return device_name; };
	inline postring&	getDeviceVersion() { return device_version; };
	inline postring&	getDeviceModel() { return model_name; };

	inline i32			getCommPort() { return comm_port; };
	inline i32			getVideoEncoder() { return video_encoder; };
	inline DateTime		getBuildDateTime() { return build_date; };
};

struct NetAdapter
{
	i32					ip_address;
	i32					ip_subnet;
	i32					ip_gateway;
	i32					ip_dns_server;
	bool				is_loopback;
	bool				is_conf_dhcp;
	postring			adapter_name;
	postring			mac_address;

public:
	NetAdapter();

	void				init();
	bool				isValid();

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
};
typedef std::vector<NetAdapter>	NetAdapterArray;

struct CIPInfo : public CLockGuard
{
public:
	CIPInfo();

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);

	bool				getFirstMacAddress(postring& mac_addr_str);

	inline i32			getCmdPort() { lock_guard(); return m_cmd_port; };
	inline i32			getIVSPort() { lock_guard(); return m_ivs_port; };
	inline i32			getDataPort() { lock_guard(); return m_data_port; };
	inline i64			getHighID() { lock_guard(); return m_high_id; };
	inline i32			getCmdAddress() { lock_guard(); return m_cmd_address; };
	inline i32			getHighAddress() { lock_guard(); return m_high_address; };

public:
	u8					m_server;
	i32					m_cmd_port;
	i32					m_cmd_address;
	i32					m_ivs_port;
	i32					m_data_port;
	NetAdapterArray		m_netadapter_vec;

	u8					m_high_adapter;
	i32					m_high_address;
	i64					m_high_id;
};

class CIODev : public CLockGuard
{
public:
	CIODev();
	~CIODev();

	void				init();
	void				reset();

	CIODev				getValue();
	void				setValue(CIODev& other);

	i32					memSize();
	i32 				memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32 				memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memImport(u8*& buffer_ptr, i32& buffer_size);

	bool				fileWrite(FILE* fp);
	bool				fileRead(FILE* fp);

public:
	i32					m_dev_address;
	
	postring			m_port_name;
	i32					m_rs_mode;
	i32					m_baud_rate;
	i32					m_data_bits;
	i32					m_flow_control;
	i32					m_parity;
	i32					m_stop_bits;

	i32					m_time_out;
	i32					m_retry_count;
};

class CPlainDevParam : public CLockGuard
{
public:
	CPlainDevParam();
	~CPlainDevParam();

	void				init();
	void				reset();

	CPlainDevParam 		getValue();
	void				setValue(CPlainDevParam& other);

	i32					memSize();
	i32 				memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32 				memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memImport(u8*& buffer_ptr, i32& buffer_size);

	bool				fileWrite(FILE* fp);
	bool				fileRead(FILE* fp);

	inline i32			getTcpPort() { lock_guard(); return m_tcp_port; };
	inline i32			getUdpPort() { lock_guard(); return m_udp_port; };
	inline postring		getSerialPortName() { lock_guard(); return m_port_name; };

public:
	i32					m_used_device;

	i32					m_tcp_port;
	i32					m_udp_port;

	postring			m_port_name;
	i32					m_rs_mode;
	i32					m_baud_rate;
	i32					m_data_bits;
	i32					m_flow_control;
	i32					m_parity;
	i32					m_stop_bits;

	i32					m_time_out;
	i32					m_retry_count;
};

class CModbusDevParam : public CLockGuard
{
public:
	CModbusDevParam();
	~CModbusDevParam();

	void				init();
	void				reset();

	CModbusDevParam		getValue();
	void				setValue(CModbusDevParam& other);

	i32					memSize();
	i32 				memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32 				memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memImport(u8*& buffer_ptr, i32& buffer_size);

	bool				fileWrite(FILE* fp);
	bool				fileRead(FILE* fp);

	CIODev				getIODevParam();

	inline i32			getDevAddress() { lock_guard(); return m_dev_address; };
	inline u8			getFmtDigitsCount() { lock_guard(); return m_fmt_output_digits; };
	inline u8			getEndianMode() { lock_guard(); return m_endian_mode; };

	inline i32			getTcpPort() { lock_guard(); return m_tcp_port; };
	inline i32			getUdpPort() { lock_guard(); return m_udp_port; };
	inline postring		getRS232PortName() { lock_guard(); return m_port_name; };

public:
	i32					m_used_device;

	i32					m_dev_address;
	u8					m_fmt_output_digits;
	u8					m_endian_mode;

	i32					m_tcp_port;
	i32					m_udp_port;

	postring			m_port_name;
	i32					m_rs_mode;
	i32					m_baud_rate;
	i32					m_data_bits;
	i32					m_flow_control;
	i32					m_parity;
	i32					m_stop_bits;

	i32					m_time_out;
	i32					m_retry_count;
};

struct DBConfig
{
	postring			host_name;			//접속할 자료기지가 설치된 IP
	postring			database_name;		//접속할 자료기지이름
	postring			username;			//자료쓰기(하위기)에 리용하는 사용자이름
	postring			password;			//자료쓰기(하위기)에 리용하는 사용자암호
	postring			remote_username;	//자료읽기(상위기)에 리용하는 사용자이름
	postring			remote_password;	//자료읽기(상위기)에 리용하는 사용자암호
	i32					db_port;			//접속할 포트번호

	i32					limit_log;			//로그자료(조작 및 오유)의 최대개수
	i32					limit_record;		//리력자료의 최대개수

public:
	DBConfig();

	void				init();

	i32					memSize();
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);

	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);
};

class COpcDev : public CLockGuard
{
public:
	COpcDev();
	~COpcDev();

	void				init();
	void				reset();

	COpcDev				getValue();
	void				setValue(COpcDev& other);

	i32					memSize();
	i32 				memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32 				memRead(u8*& buffer_ptr, i32& buffer_size);

	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

	inline u16			getOpcPort() { lock_guard(); return m_port; };
	inline u16			getOpcInterval() { lock_guard(); return m_interval; };
	inline bool			isOpcUsed() { lock_guard(); return m_is_used; };

public:
	bool				m_is_used;
	u16					m_port;
	u16					m_interval;
};

class CFtpDev : public CLockGuard
{
public:
	CFtpDev();
	~CFtpDev();

	void				init();
	void				reset();

	CFtpDev				getValue();
	void				setValue(CFtpDev& other);

	i32					memSize();
	i32 				memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32 				memRead(u8*& buffer_ptr, i32& buffer_size);

	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

public:
	postring			m_ftp_hostname;
	postring			m_ftp_username;
	postring			m_ftp_password;
	u16					m_ftp_port;
};

class CFTPDevGroup : public CLockGuard
{
public:
	CFTPDevGroup();
	~CFTPDevGroup();

	virtual void		init();

	virtual i32			memSize();
	virtual i32 		memWrite(u8*& buffer_ptr, i32& buffer_size);
	virtual i32 		memRead(u8*& buffer_ptr, i32& buffer_size);

	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

public:
	CFtpDev				m_ftp_dev_param[POFTP_MAX_DEV];
};

class CFileHeader
{
public:
	CFileHeader();
	CFileHeader(const char* sz_header);

	virtual bool		isValid() = 0;
	void				setSeek(i32 id, u64 pos);

public:
	char				header[32];
	u64					seek[5];
};

struct AlarmQueue
{
	i32					read_pos;
	i32					write_pos;
	POAlarm				queue[PODB_MAX_ALARM];

public:
	AlarmQueue();

	bool				getAlarmLog(POAlarm& alarm);
	bool				getAlarmLogVec(POAlarmVec& alarm_vec);
    bool				setAlarmLog(const POAlarm& alarm);
};

struct ActionQueue
{
	i32					read_pos;
	i32					write_pos;
	POAction			queue[PODB_MAX_ACTION];

public:
	ActionQueue();

	bool				getActionLog(POAction& action);
	bool				getActionLogVec(POActionVec& action_vec);
    bool				setActionLog(const POAction& action);
};

struct InfoQueue
{
	i32					read_pos;
	i32					write_pos;
	POInfo				queue[PODB_MAX_INFO];

public:
	InfoQueue();

	bool				getInfoLog(POInfo& action);
	bool				getInfoLogVec(POInfoVec& action_vec);
	bool				setInfoLog(const POInfo& action);
};

template <typename T>
struct Pixel
{
	T					x;
	T					y;

public:
	Pixel()
	{
		x = 0;  y = 0;
	}
	template <typename U>
	Pixel(U px, U py)
	{
		x = (T)px; y = (T)py;
	}
};

template <typename T>
struct Pixel3
{
	T					x;
	T					y;
	T					g;

public:
	Pixel3()
	{
		x = 0;  y = 0; g = 0;
	};
	template <typename U>
	Pixel3(U px, U py)
	{
		x = (T)px; y = (T)py; g = (T)0;
	};
	template <typename U>
	Pixel3(U px, U py, U pg)
	{
		x = (T)px; y = (T)py; g = (T)pg;
	};
};

template <typename T>
struct Line
{
	vector2d<T>			pt1;
	vector2d<T>			pt2;

public:
	Line()
	{
		pt1.x = 0; pt1.y = 0; pt2.x = 0; pt2.y = 0;
	};
	template <typename U>
	Line(vector2d<U> p1, vector2d<U> p2)
	{
		pt1.x = p1.x; pt1.y = p1.y;
		pt2.x = p2.x; pt2.y = p2.y;
	};
};

template <typename T>
class Contour
{
public:
	Contour()
	{
		memset(this, 0, sizeof(Contour));
	};
	Contour(Pixel<T>* pixels)
	{
		memset(this, 0, sizeof(Contour));
		m_pixel_ptr = pixels;
		m_malloc_external = true;
	};
	Contour(Pixel<T>* pixel_ptr, i32& malloc_pixels)
	{
		memset(this, 0, sizeof(Contour));
		m_pixel_ptr = pixel_ptr;
		m_malloc_pixel_count = malloc_pixels;
		m_malloc_external = true;
	};
	~Contour()
	{
		freeBuffer();
	}

	void setValue(Contour<T>& contour)
	{
		freeBuffer();
		m_is_closed = contour.m_is_closed;
		m_pixel_count = contour.m_pixel_count;

		if (m_pixel_count)
		{
			i32 i, index;
			i32 start_pixel_index = contour.m_start_pixel_index;
			i32 malloc_pixel_count = contour.m_malloc_pixel_count;
			Pixel<T>* other_pixel_ptr = contour.m_pixel_ptr;

			if (other_pixel_ptr)
			{
				m_pixel_ptr = po_new Pixel<T>[m_pixel_count];
				Pixel<T>* tmp_pixel_ptr = m_pixel_ptr;

				for (i = 0; i < m_pixel_count; i++)
				{
					index = (start_pixel_index + i) % malloc_pixel_count;
					*tmp_pixel_ptr = other_pixel_ptr[index];
					tmp_pixel_ptr++;
				}

				m_start_pixel_index = 0;
				m_end_pixel_index = m_pixel_count - 1;
			}
		}
	}

	void freeBuffer()
	{
		if (!m_malloc_external)
		{
			POSAFE_DELETE_ARRAY(m_pixel_ptr);
		}

		m_pixel_ptr = NULL;
		m_malloc_external = false;
	}

	inline bool			isClosedContour()	{ return m_is_closed; };
	inline i32			getContourPixelNum(){ return m_pixel_count; };
	inline Pixel<T>*	getContourPixel()	{ return m_pixel_ptr; };

public:
	i32					m_pixel_count;
	Pixel<T>*			m_pixel_ptr;

	bool				m_is_closed;
	bool				m_malloc_external;

	i32					m_start_pixel_index;
	i32					m_end_pixel_index;
	i32					m_malloc_pixel_count;
};

typedef Pixel<u16> Pixelu;
typedef Pixel<f32> Pixelf;
typedef Pixel3<u16> Pixel3u;
typedef Pixel3<f32> Pixel3f;
typedef std::vector<Pixelf>	ptfvector;
typedef std::vector<Pixelu> ptuvector;
typedef std::vector<vector2dd> ptvector2dd;
typedef std::vector<vector2df> ptvector2df;
typedef std::vector<vector2di> ptvector2di;
typedef std::vector<vector3dd> ptvector3dd;
typedef std::vector<vector3df> ptvector3df;
typedef std::vector<vector3di> ptvector3di;
typedef std::vector<ptuvector> shapeuvector;
typedef std::vector<Pixel3u> pt3uvector;
typedef std::vector<Pixel3f> pt3fvector;
typedef Contour<u16> Contouru;
typedef Contour<f32> Contourf;
typedef std::vector<Contouru*> ContouruVector;

//////////////////////////////////////////////////////////////////////////
// complex structs
struct POLineFormat
{
	POLineFormat();
	POLineFormat(i32 width, i32 style, i32 color);

	u8							line_width;
	u8							line_style;
	u8							line_color;
};
typedef std::vector<POLineFormat> POLineFormatVector;

class CPOArc
{
public:
	CPOArc();
	~CPOArc();

	void						init();
	void						setValue(const CPOArc& other);
	void						setValue(CPOArc* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	vector2df					m_center;
	f32							m_radius;
	f32							m_st_angle; //according to ccw-direction
	f32							m_angle_len;
	POLineFormat				m_format;
};

class CPOCircle
{
public:
	CPOCircle();
	~CPOCircle();

	void						init();
	void						setValue(const CPOCircle& other);
	void						setValue(CPOCircle* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	vector2df					m_center;
	f32							m_radius;
	POLineFormat				m_format;
};

class CPOLine
{
public:
	CPOLine();
	~CPOLine();

	void						init();
	void						setValue(const CPOLine& other);
	void						setValue(CPOLine* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	vector2df					m_st_point;
	vector2df					m_ed_point;
	POLineFormat				m_format;
};

class CPOCross
{
public:
	CPOCross();
	~CPOCross();

	void						init();
	void						setValue(const CPOCross& other);
	void						setValue(CPOCross* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	vector2df					m_point;
	f32							m_angle;
	f32							m_length;
	POLineFormat				m_format;
};

class CPORotatedRect
{
public:
	CPORotatedRect();
	~CPORotatedRect();

	void						init();
	void						setValue(const CPORotatedRect& other);
	void						setValue(CPORotatedRect* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	Rectf						m_rect;
	f32							m_angle;
	POLineFormat				m_format;
};

class CPOContours
{
public:
	CPOContours();
	~CPOContours();

	void						init(i32 reserved = 1);
	void						initContours(i32 count);
	void						freeBuffer();
	void						resize(i32 count);

	i32							getContoursCount();
	i32							getContourLength();
	i32							getContourSize(i32 index);
	ptvector2df*				getContour(i32 index);

	void						setValue(CPOContours* other_ptr);
	void						setValue(const CPOContours& other);
	void						addContour(bool is_closed, const vector2df* pt_ptr, i32 count);
	void						addContour(const vector2df* pt_ptr, i32 count, bool is_closed = false, i32 info = 0,
										i32 width = kPOLineWidthAuto, i32 style = kPOLineStyleAuto, i32 color = kPOColorAuto);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	POLineFormatVector			m_format_vec;
	u8vector					m_is_closed_vec;
	i32vector					m_info_vec;
	std::vector<ptvector2df>	m_contour_vec;
};

class CPOShape
{
public:
	CPOShape();
	CPOShape(f32 x1, f32 y1, f32 x2, f32 y2);			//edge
	CPOShape(f32 x0, f32 y0, f32 dx, f32 dy, f32 len);	//line
	CPOShape(f32 cx, f32 cy, f32 r);					//circle

	void						init();

	void						setEdge(vector2df st_pos, vector2df ed_pos);
	void						setLine(vector2df line_pos, vector2df line_dir, f32 len = 256.0f);
	void						setCircle(vector2df center_pos, f32 radius, f32 st_angle = 0, f32 angle_len = PO_PI2);
	bool						getEdge(vector2df& st_pos, vector2df& ed_pos) const;
	bool						getCircle(vector2df& center_pos, f32& radius) const;

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	void						updateCenter(vector2df center_offset);
	void						updateShape(f32 min_dist, f32 max_dist);

	f32							distance(vector2df pos2d);
	f32							distance(vector2df pos2d, f32& dist);
	f32							signedDistance(vector2df pos2d);
	f32							signedDistance(vector2df pos2d, f32& dist);

	inline i32					getShapeMode() { return m_shape_mode; };
	
public:
	i32							m_shape_mode;
	POLineFormat				m_format;
	union {
		struct { f32 x1, y1, x2, y2; } _edge;
		struct { f32 x0, y0, dx, dy, len; } _line;
		struct { f32 cx, cy, r, st_angle, angle_len;} _circle;
	} d;
};
typedef std::vector<CPOShape> POShapeVec;

class CVirtualEncoder
{
public:
	virtual bool				acquireEncoder(i32 encoder, i32 w, i32 h, i32 channel, i32 frate, i32 brate, i32 vid) = 0;
	virtual void				releaseEncoder() = 0;
	virtual void				setImageToEncoder(ImageData* img_data_ptr) = 0;
	virtual void				setImageToEncoder(ImageData* img_data_ptr, i32 cam_id) = 0;

protected:
	virtual void*				onEncodedFrame(u8* buffer_ptr, i32 size, i64 pts, ImageData* img_data_ptr) = 0;
	virtual void				onSendFrame(void* send_void_ptr) = 0;
};

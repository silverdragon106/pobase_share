#include "opc_ua_sdk.h"
#include "struct.h"
#include <QThread>

#define OPC_DEVICE_NAMESPACE_INDEX		2

#define OPC_STATUS_SUCCESS				0
#define OPC_STATUS_FAILURE				1
#define OPC_STATUS_ALREADY_RUNNING		2
#define OPC_STATUS_ALREADY_STOPPED		3
#define OPC_UNKNOWN_TYPE				4

#define OPC_STATUS_ONLINE				0
#define OPC_STATUS_OFFLINE				1

struct OpcValue
{
public:
	i32						value_type;
	union 
	{
		UA_Boolean			value_bool;
		UA_SByte			value_i8;
		UA_Int16			value_i16;
		UA_Int32			value_i32;
		UA_Int64			value_i64;
		UA_Byte				value_u8;
		UA_UInt16			value_u16;
		UA_UInt32			value_u32;
		UA_UInt64			value_u64;
		UA_Float			value_f32;
		UA_Double			value_f64;
		UA_String			value_str;
	} u;

public:
	OpcValue();
	
	inline bool				isValid() { return value_type >= UA_TYPES_BOOLEAN; };
};

class COpcUaServer : public QThread
{
public:
	COpcUaServer();
	~COpcUaServer();

	virtual bool				initInstance();
	virtual bool				exitInstance();
	virtual i32					initOpcNameSpace();

	virtual bool				restartOpc();

	virtual bool				readNodeValue(const postring& node_id, OpcValue& value);
	virtual bool				writeNodeValue(const postring& node_id, const UA_Variant* data_ptr);

	i32							createFolderNode(const postring& str_node_id);
	i32							createValueNode(const postring& str_node_id, bool is_readable, bool is_writable);
	i32							deleteSubFolderNodes(const postring& str_node_id);

	void						convertToUaString(const postring& str_value, OpcValue& opc_value);

	void						initAllSettings();
	bool						loadAllSettings(FILE* fp);
	bool						writeAllSettings(FILE* fp);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);

	inline bool					isOpcUsed() { return m_opc_setting.isOpcUsed(); };
	inline COpcDev*				getOpcDevParam() { return &m_opc_setting; };
	inline UA_Server*			getUaServer() { return m_server_ptr; };

protected:
	void						run() Q_DECL_OVERRIDE;
	
public:
	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;

	COpcDev						m_opc_setting;

	UA_Server*					m_server_ptr;
	UA_ServerNetworkLayer		m_server_netlayer;
	POMutex						m_server_mutex;
};
#include "opc_ua_server.h"
#include "logger/logger.h"
#include "base.h"

#define MAX_NOTIFICATIONS_PER_PUBLISH	10000

const i32 kOpcConnectionSendBufferSize = 655360;
const i32 kOpcConnectionRecvBufferSize = 655360;
const i32 kOpcConnectionMaxChunkCount = 32;
const i32 kOpcConnectionMaxMessageSize = 655360 * 32;

//////////////////////////////////////////////////////////////////////////
OpcValue::OpcValue()
{
	value_type = -1;
}

//////////////////////////////////////////////////////////////////////////
COpcUaServer* g_opc_server_ptr = NULL;

postring getUANodeIdString(const UA_NodeId& node_id)
{
	if (node_id.identifierType != UA_NODEIDTYPE_STRING)
	{
		return postring();
	}

	u8* buffer_pos = node_id.identifier.string.data;
	u8* buffer_end = buffer_pos + node_id.identifier.string.length;
	return postring(buffer_pos, buffer_end);
}

UA_StatusCode readNodeValueCallback(void* handle_ptr, const UA_NodeId nodeid, UA_Boolean source_time_stamp,
						const UA_NumericRange* range_ptr, UA_DataValue* data_value_ptr)
{
	if (!g_opc_server_ptr || !data_value_ptr)
	{
		return UA_STATUSCODE_BADUNEXPECTEDERROR;
	}
	
	OpcValue value;
	if (!g_opc_server_ptr->readNodeValue(getUANodeIdString(nodeid), value) ||!value.isValid())
	{
		return UA_STATUSCODE_GOOD;
	}

	switch (value.value_type)
	{
		case UA_TYPES_BOOLEAN:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_bool), &UA_TYPES[UA_TYPES_BOOLEAN]);
			break;
		}
		case UA_TYPES_SBYTE:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_i8), &UA_TYPES[UA_TYPES_SBYTE]);
			break;
		}
		case UA_TYPES_INT16:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_i16), &UA_TYPES[UA_TYPES_INT16]);
			break;
		}
		case UA_TYPES_INT32:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_i32), &UA_TYPES[UA_TYPES_INT32]);
			break;
		}
		case UA_TYPES_INT64:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_i64), &UA_TYPES[UA_TYPES_INT64]);
			break;
		}
		case UA_TYPES_BYTE:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_u8), &UA_TYPES[UA_TYPES_BYTE]);
			break;
		}
		case UA_TYPES_UINT16:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_u16), &UA_TYPES[UA_TYPES_UINT16]);
			break;
		}
		case UA_TYPES_UINT32:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_u32), &UA_TYPES[UA_TYPES_UINT32]);
			break;
		}
		case UA_TYPES_UINT64:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_u64), &UA_TYPES[UA_TYPES_UINT64]);
			break;
		}
		case UA_TYPES_FLOAT:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_f32), &UA_TYPES[UA_TYPES_FLOAT]);
			break;
		}
		case UA_TYPES_DOUBLE:
		{
			UA_Variant_setScalarCopy(&(data_value_ptr->value), &(value.u.value_f64), &UA_TYPES[UA_TYPES_DOUBLE]);
			break;
		}
		case UA_TYPES_STRING:
		{
			UA_String* string_ptr = &value.u.value_str;
			UA_Variant_setScalarCopy(&(data_value_ptr->value), string_ptr, &UA_TYPES[UA_TYPES_STRING]);
			UA_String_deleteMembers(string_ptr);
			break;
		}
		default:
		{
			return UA_STATUSCODE_GOOD;
		}
	}

	data_value_ptr->hasValue = true;
	return UA_STATUSCODE_GOOD;
}

UA_StatusCode writeNodeValueCallback(void* handle_ptr, const UA_NodeId nodeid,
						const UA_Variant* data_ptr, const UA_NumericRange* range_ptr)
{
	if (!g_opc_server_ptr || !data_ptr)
	{
		return UA_STATUSCODE_BADUNEXPECTEDERROR;
	}

	g_opc_server_ptr->writeNodeValue(getUANodeIdString(nodeid), data_ptr);
	return UA_STATUSCODE_GOOD;
}

UA_StatusCode deleteNodeCallback(UA_NodeId child_node_id, UA_Boolean is_inverse,
						UA_NodeId ref_type_id, void* handle_ptr)
{
	UA_NodeId reference_id = UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES);
	if (!g_opc_server_ptr || is_inverse || !UA_NodeId_equal(&ref_type_id, &reference_id))
	{
		return UA_STATUSCODE_BADUNEXPECTEDERROR;
	}

	UA_Server* server_ptr = g_opc_server_ptr->getUaServer();
	UA_Server_forEachChildNodeCall(server_ptr, child_node_id, deleteNodeCallback, handle_ptr);
	if (UA_Server_deleteNode(server_ptr, child_node_id, true) != UA_STATUSCODE_GOOD)
	{
		printlog_lvs2("DeleteNode Failed", LOG_SCOPE_OPC);
	}
	return UA_STATUSCODE_GOOD;
}

//////////////////////////////////////////////////////////////////////////
COpcUaServer::COpcUaServer()
{
	m_is_inited = false;
	m_is_thread_cancel = false;
	m_opc_setting.init();

	m_server_ptr = NULL;
	g_opc_server_ptr = this;
}

COpcUaServer::~COpcUaServer()
{
	exitInstance();
}

bool COpcUaServer::initInstance()
{
	if (!m_is_inited)
	{
		singlelog_lv0("OpcUaServer InitInstance");
		m_is_thread_cancel = false;

		UA_ServerConfig server_config = UA_ServerConfig_standard;
		UA_ConnectionConfig connection_config = UA_ConnectionConfig_standard;

		connection_config.sendBufferSize = kOpcConnectionSendBufferSize;
		connection_config.recvBufferSize = kOpcConnectionRecvBufferSize;
		connection_config.maxChunkCount = kOpcConnectionMaxChunkCount;
		connection_config.maxMessageSize = kOpcConnectionMaxMessageSize;

		m_server_netlayer = UA_ServerNetworkLayerTCP(connection_config, m_opc_setting.getOpcPort());
		server_config.networkLayers = &m_server_netlayer;
		server_config.networkLayersSize = 1;
		server_config.maxNotificationsPerPublish = MAX_NOTIFICATIONS_PER_PUBLISH;

		m_server_ptr = UA_Server_new(server_config);
		if (!m_server_ptr)
		{
			printlog_lvs1("Opc can't create", LOG_SCOPE_OPC);
			m_server_netlayer.deleteMembers(&m_server_netlayer);
			return false;
		}

		UA_Server_addNamespace(m_server_ptr, "Device address namespace");
		UA_Server_run_startup(m_server_ptr);
		QThreadStart();
		m_is_inited = true;
	}
	return true;
}

bool COpcUaServer::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0("OpcUaServer ExitInstance");

		m_is_thread_cancel = true;
		QThreadStop();

		//delete all sub nodes
		deleteSubFolderNodes("");

		if (m_server_ptr)
		{
			UA_Server_run_shutdown(m_server_ptr);
			UA_Server_delete(m_server_ptr);
			m_server_ptr = NULL;
		}
		m_server_netlayer.deleteMembers(&m_server_netlayer);
		m_is_thread_cancel = false;
		m_is_inited = false;
	}
	return true;
}

i32 COpcUaServer::initOpcNameSpace()
{
	return OPC_STATUS_FAILURE;
}

bool COpcUaServer::restartOpc()
{
	if (!m_is_inited)
	{
		return true;
	}

	if (!exitInstance())
	{
		return false;
	}
	return initInstance();
}

void COpcUaServer::initAllSettings()
{
	m_opc_setting.init();
}

bool COpcUaServer::loadAllSettings(FILE* fp)
{
	return m_opc_setting.fileRead(fp);
}

bool COpcUaServer::writeAllSettings(FILE* fp)
{
	return m_opc_setting.fileWrite(fp);
}

i32	COpcUaServer::memSize()
{
	return m_opc_setting.memSize();
}

i32	COpcUaServer::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	return m_opc_setting.memRead(buffer_ptr, buffer_size);
}

i32	COpcUaServer::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	return m_opc_setting.memWrite(buffer_ptr, buffer_size);
}

void COpcUaServer::run()
{
	if (!m_is_inited)
	{
		return;
	}

	singlelog_lv0("OpcUaServer Thread");
	i32 sleep_ms = m_opc_setting.getOpcInterval();
	UA_Boolean wait_internal = false;
	UA_UInt16 timeout;

	while (!m_is_thread_cancel)
	{
		{
			/* timeout is the maximum possible delay (in millisec) until the next
			_iterate call. Otherwise, the server might miss an internal timeout
			or cannot react to messages with the promised responsiveness. */
			/* If multicast discovery server is enabled, the timeout does not not consider new input data (requests) on the mDNS socket.
			* It will be handled on the next call, which may be too late for requesting clients.
			* if needed, the select with timeout on the multicast socket server->mdnsSocket (see example in mdnsd library)
			*/
			exlock_guard(m_server_mutex);
			timeout = UA_Server_run_iterate(m_server_ptr, wait_internal);
		}
		QThread::msleep(sleep_ms);
	}
}

i32 COpcUaServer::createFolderNode(const postring& str_node_id)
{
	if (!m_is_inited)
	{
		return OPC_STATUS_FAILURE;
	}
	postring str_parent, str_name;
	CPOBase::splitToPath(str_node_id, ".", str_parent, str_name);

	exlock_guard(m_server_mutex);

	UA_NodeId parent_node_id;
	UA_NodeId parent_reference_id = UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES);
	i32 ret_code = OPC_STATUS_FAILURE;

	if (!str_parent.empty())
	{
		//check parent folder node
		UA_NodeId node_id;
		parent_node_id = UA_NODEID_STRING(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_parent.c_str()));
		if (UA_Server_readNodeId(m_server_ptr, parent_node_id, &node_id) != UA_STATUSCODE_GOOD)
		{
			ret_code = createFolderNode(str_parent);
			if (ret_code != OPC_STATUS_SUCCESS)
			{
				return ret_code;
			}
		}
	}
	else
	{
		//use device root node
		parent_node_id = UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER);
	}

	//add folder node
	UA_ObjectAttributes attr;
	UA_ObjectAttributes_init(&attr);
	attr.description = UA_LOCALIZEDTEXT("en_US", const_cast<char*>(str_name.c_str()));
	attr.displayName = UA_LOCALIZEDTEXT("en_US", const_cast<char*>(str_name.c_str()));

	UA_NodeId node_id = UA_NODEID_STRING(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_node_id.c_str()));
	UA_QualifiedName node_name = UA_QUALIFIEDNAME(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_name.c_str()));

	ret_code = UA_Server_addObjectNode(m_server_ptr, node_id, parent_node_id,
						parent_reference_id, node_name, UA_NODEID_NULL, attr, NULL, NULL);

	return (ret_code == UA_STATUSCODE_GOOD) ? OPC_STATUS_SUCCESS : OPC_STATUS_FAILURE;
}

i32 COpcUaServer::createValueNode(const postring& str_node_id, bool is_readable, bool is_writable)
{
	if (!m_is_inited)
	{
		return OPC_STATUS_FAILURE;
	}

	exlock_guard(m_server_mutex);

	postring str_parent, str_name;
	CPOBase::splitToPath(str_node_id, ".", str_parent, str_name);

	UA_NodeId parent_node_id;
	UA_NodeId parent_reference_id = UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES);
	i32 ret_code = OPC_STATUS_FAILURE;

	if (!str_parent.empty())
	{
		//check parent folder node
		UA_NodeId node_id;
		parent_node_id = UA_NODEID_STRING(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_parent.c_str()));
		if (UA_Server_readNodeId(m_server_ptr, parent_node_id, &node_id) != UA_STATUSCODE_GOOD)
		{
			ret_code = createFolderNode(str_parent);
			if (ret_code != OPC_STATUS_SUCCESS)
			{
				return ret_code;
			}
		}
	}
	else
	{
		//use device root node
		parent_node_id = UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER);
	}

	//add value node
	UA_VariableAttributes attr;
	UA_VariableAttributes_init(&attr);
	attr.displayName = UA_LOCALIZEDTEXT("en_US", const_cast<char*>(str_name.c_str()));
	attr.accessLevel = (is_readable ? UA_ACCESSLEVELMASK_READ : 0) | (is_writable ? UA_ACCESSLEVELMASK_WRITE : 0);
	attr.userAccessLevel = attr.accessLevel;
	
	UA_NodeId data_node_id = UA_NODEID_STRING(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_node_id.c_str()));
	UA_QualifiedName data_node_name = UA_QUALIFIEDNAME(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_name.c_str()));
	UA_NodeId variable_type_node_id = UA_NODEID_NULL;

	UA_DataSource value_data_source;
	value_data_source.handle = NULL;
	value_data_source.read = readNodeValueCallback;
	value_data_source.write = writeNodeValueCallback;

	ret_code = UA_Server_addDataSourceVariableNode(m_server_ptr, data_node_id, parent_node_id,
						parent_reference_id, data_node_name, variable_type_node_id, attr, value_data_source, NULL);
	return (ret_code == UA_STATUSCODE_GOOD) ? OPC_STATUS_SUCCESS : OPC_STATUS_FAILURE;
}

i32 COpcUaServer::deleteSubFolderNodes(const postring& str_node)
{
	if (!m_is_inited)
	{
		return OPC_STATUS_FAILURE;
	}

	exlock_guard(m_server_mutex);

	UA_NodeId node_id;
	if (!str_node.empty())
	{
		node_id = UA_NODEID_STRING(OPC_DEVICE_NAMESPACE_INDEX, const_cast<char*>(str_node.c_str()));

		UA_Server_forEachChildNodeCall(m_server_ptr, node_id, deleteNodeCallback, NULL);
		UA_Server_deleteNode(m_server_ptr, node_id, true);
		createFolderNode(str_node);
	}
	else
	{
		//use device root node
		node_id = UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER);
		UA_Server_forEachChildNodeCall(m_server_ptr, node_id, deleteNodeCallback, NULL);
	}
	return OPC_STATUS_SUCCESS;
}

bool COpcUaServer::readNodeValue(const postring& node_id, OpcValue& value)
{
	return false;
}

bool COpcUaServer::writeNodeValue(const postring& node_id, const UA_Variant* data_ptr)
{
	return false;
}

void COpcUaServer::convertToUaString(const postring& str_value, OpcValue& opc_value)
{
	i32 size = (i32)str_value.size();
	if (CPOBase::isPositive(size))
	{
		opc_value.u.value_str.data = po_new u8[size];
		CPOBase::memCopy(opc_value.u.value_str.data, str_value.c_str(), size);
	}
	else
	{
		opc_value.u.value_str.data = NULL;
		opc_value.u.value_str.length = size;
	}
}

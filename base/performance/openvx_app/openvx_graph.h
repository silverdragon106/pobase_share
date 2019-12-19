#pragma once
#include "config.h"

#ifdef POR_WITH_OVX
#include <VX/vx.h>
#include <vector>
#include "struct.h"

/* 
OvxGraphBase
@brief
	base class for graph modules of mold protector engine.
*/

enum kGraphModeTypes
{
	kGraphModeNone = 0x00,
	kGraphModeSubGraph = 0x01,
	kGraphModeOutput = 0x02,
};

class COpenVxGraph
{
public:
	COpenVxGraph();
	virtual ~COpenVxGraph();

	virtual vx_status			release();

	virtual vx_status			schedule();
	virtual vx_status			process();
	virtual vx_status			waitFinish();
	virtual vx_status			finished();
	
	vx_status					createGraph();
	vx_status					verifyGraph();

	vx_graph					getGraph();
	vx_status					setGraphMode(vx_int32 mode);
	vx_status					setContext(vx_context context);
	vx_status					setParent(COpenVxGraph* parent_ptr);
	bool						isFree();

	vx_size						addNode(vx_node node);
	vx_reference				addReference(vx_reference ref);
	vx_node						lastNode();

	vx_array					addArray(vx_enum type, vx_size capacity);
	vx_image					addImage(vx_int32 w, vx_int32 h, vx_df_image format);
	vx_pyramid					addPyramid(vx_int32 w, vx_int32 h, vx_float32 scale, vx_size levels, vx_df_image format);
	vx_matrix					addMatrix(vx_enum type, vx_size rows, vx_size cols);

	vx_array					addVirtualArray(vx_enum type, vx_size capacity);
	vx_image					addVirtualImage(vx_int32 w, vx_int32 h, vx_df_image format);
	vx_pyramid					addVirtualPyramid(vx_int32 w, vx_int32 h, vx_float32 scale, vx_size levels, vx_df_image format);
	
	inline vx_context			getContext() { return m_context; };
	inline COpenVxGraph*		getParent() { return m_parent_ptr; };
	inline vx_status			getGraphResult() { return m_graph_result; };
	inline vx_int32				getGraphMode() { return m_graph_mode; };
	inline bool					hasParent() { return (m_parent_ptr != NULL); };
	
	template <typename T>
	vx_scalar addScalar(vx_enum type, T val)
	{
		vx_scalar ret = vxCreateScalar(m_context, type, &val);
		addReference((vx_reference)ret);
		return ret;
	}

public:
	vx_graph					m_graph;
	vx_status					m_graph_result;
	vx_context					m_context;

	std::vector<vx_node>		m_node_vec;
	std::vector<vx_reference>	m_references;

	vx_int32					m_graph_mode;
	COpenVxGraph*				m_parent_ptr;
};

class CVxGraphGroup
{
public:
	CVxGraphGroup();

	bool						initGraphGroup(i32 count);
	bool						clearGraphGroup();

	COpenVxGraph*				getFreeGraph();
	bool						waitGroup();

public:
	std::vector<COpenVxGraph*>	m_graph_vec;
};

//////////////////////////////////////////////////////////////////////////
/* get scalar parameter value by index from node. */
template<typename T>
T getNodeParameterByIndex(vx_node node, vx_int32 index, T def_val)
{
	vx_action ret = VX_ACTION_CONTINUE;

	T val_data;
	vx_scalar val;
	vx_parameter param;

	param = vxGetParameterByIndex(node, index);
	if (param)
	{
		vxQueryParameter(param, VX_PARAMETER_REF, &val, sizeof(val));
		if (val)
		{
			vxCopyScalar(val, &val_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
			if (val_data <= 0)
			{
				/* stop executing graph. */
				return def_val;
			}
		}
	}
	return val_data;
};

#endif

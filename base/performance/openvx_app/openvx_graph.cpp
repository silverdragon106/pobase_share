#include "openvx_graph.h"

#ifdef POR_WITH_OVX
#include <math.h>
#include <algorithm>
#include "base.h"

/* Implementation of OvxGraphBase */
COpenVxGraph::COpenVxGraph()
{
	m_context = NULL;
	m_graph = NULL;
	m_parent_ptr = NULL;
	m_graph_result = VX_SUCCESS;
	m_graph_mode = kGraphModeNone;

	m_node_vec.clear();
	m_references.clear();
}

COpenVxGraph::~COpenVxGraph()
{
	release();
}

vx_status COpenVxGraph::release()
{
	vx_status status = VX_SUCCESS;
	
	if (m_parent_ptr)
	{
		m_node_vec.clear();
		m_references.clear();
	}
	else
	{
		for each (vx_node node in m_node_vec)
		{
			status |= vxReleaseNode(&node);
		}
		m_node_vec.clear();

		status |= vxReleaseGraph(&m_graph);

		for each(vx_reference ref in m_references)
		{
			status |= vxReleaseReference(&ref);
		}
		m_references.clear();
	}
	return status;
}

vx_status COpenVxGraph::setGraphMode(vx_int32 mode)
{
	m_graph_mode = mode;
	return VX_SUCCESS;
}

vx_status COpenVxGraph::setContext(vx_context context)
{
	m_context = context;
	return VX_SUCCESS;
}

vx_status COpenVxGraph::setParent(COpenVxGraph* parent_ptr)
{
	if (parent_ptr)
	{
		m_parent_ptr = parent_ptr;
		m_context = parent_ptr->getContext();
	}
	return VX_SUCCESS;
}

vx_size COpenVxGraph::addNode(vx_node node)
{
	if (!node)
	{
		return m_node_vec.size();
	}

	if (hasParent())
	{
		return getParent()->addNode(node);
	}
	else
	{
		m_node_vec.push_back(node);
		return m_node_vec.size();
	}
	return 0;
}

vx_matrix COpenVxGraph::addMatrix(vx_enum type, vx_size rows, vx_size cols)
{
	vx_matrix ret = vxCreateMatrix(getContext(), type, cols, rows);
	addReference((vx_reference)ret);
	return ret;
}

vx_array COpenVxGraph::addArray(vx_enum type, vx_size capacity)
{
	vx_array ret = vxCreateArray(getContext(), type, capacity);
	addReference((vx_reference)ret);
	return ret;
}

vx_array COpenVxGraph::addVirtualArray(vx_enum type, vx_size capacity)
{
	vx_array ret = vxCreateVirtualArray(getGraph(), type, capacity);
	addReference((vx_reference)ret);
	return ret;
}

vx_image COpenVxGraph::addImage(vx_int32 w, vx_int32 h, vx_df_image format)
{
	vx_image ret = vxCreateImage(getContext(), w, h, format);
	addReference((vx_reference)ret);
	return ret;
}

vx_image COpenVxGraph::addVirtualImage(vx_int32 w, vx_int32 h, vx_df_image format)
{
	vx_image ret = vxCreateVirtualImage(getGraph(), w, h, format);
	addReference((vx_reference)ret);
	return ret;
}

vx_pyramid COpenVxGraph::addPyramid(vx_int32 w, vx_int32 h, vx_float32 scale, vx_size levels, vx_df_image format)
{
	vx_pyramid pyramid = vxCreatePyramid(getContext(), levels, scale, w, h, format);
	addReference((vx_reference)pyramid);
	return pyramid;
}

vx_pyramid COpenVxGraph::addVirtualPyramid(vx_int32 w, vx_int32 h, vx_float32 scale, vx_size levels, vx_df_image format)
{
	vx_pyramid pyramid = vxCreateVirtualPyramid(getGraph(), levels, scale, w, h, format);
	addReference((vx_reference)pyramid);
	return pyramid;
}

vx_reference COpenVxGraph::addReference(vx_reference ref)
{
	if (!ref)
	{
		return NULL;
	}
	if (hasParent())
	{
		return getParent()->addReference(ref);
	}
	else
	{
		m_references.push_back(ref);
	}
	return ref;
}

vx_node COpenVxGraph::lastNode()
{
	i32 node_count = (i32)m_node_vec.size();
	if (node_count == 0)
	{
		return NULL;
	}
	return m_node_vec[node_count - 1];
}

vx_graph COpenVxGraph::getGraph()
{
	if (hasParent())
	{
		return getParent()->getGraph();
	}
	return m_graph;
}

vx_status COpenVxGraph::createGraph()
{
	if (hasParent())
	{
		return VX_SUCCESS;
	}
	if (!m_graph)
	{
		m_graph = vxCreateGraph(m_context);
	}
	return VX_SUCCESS;
}

vx_status COpenVxGraph::verifyGraph()
{
	if (hasParent())
	{
		return VX_SUCCESS;
	}
	if (m_graph)
	{
		return vxVerifyGraph(m_graph);
	}
	return VX_ERROR_INVALID_GRAPH;
}

vx_status COpenVxGraph::schedule()
{
	if (!m_graph)
	{
		return VX_ERROR_INVALID_GRAPH;
	}
	return vxScheduleGraph(m_graph);
}

vx_status COpenVxGraph::process()
{
	m_graph_result = VX_FAILURE;
	if (!m_graph)
	{
		return VX_ERROR_INVALID_GRAPH;
	}
	
	return (m_graph_result = vxProcessGraph(m_graph));
}

vx_status COpenVxGraph::waitFinish()
{
	m_graph_result = VX_FAILURE;
	if (!m_graph)
	{
		return VX_ERROR_INVALID_GRAPH;
	}

	m_graph_result = vxWaitGraph(m_graph);
	if (m_graph_result == VX_SUCCESS)
	{
		finished();
	}
	return m_graph_result;
}

vx_status COpenVxGraph::finished()
{
	vx_graph_state_e graph_state;
	vx_status status = vxQueryGraph(m_graph, VX_GRAPH_STATE, &graph_state, sizeof(vx_graph_state_e));

	if (status != VX_SUCCESS || graph_state != VX_GRAPH_STATE_COMPLETED)
	{
		m_graph_result = VX_FAILURE;
		return VX_FAILURE;
	}
	return VX_SUCCESS;
}

bool COpenVxGraph::isFree()
{
	if (!m_graph)
	{
		return false;
	}

	vx_graph_state_e graph_state;
	vx_status status = vxQueryGraph(m_graph, VX_GRAPH_STATE, &graph_state, sizeof(vx_graph_state_e));
	if (status != VX_SUCCESS)
	{
		return false;
	}
	return (graph_state == VX_GRAPH_STATE_COMPLETED || graph_state == VX_GRAPH_STATE_VERIFIED
		|| graph_state == VX_GRAPH_STATE_ABANDONED);
}

//////////////////////////////////////////////////////////////////////////
CVxGraphGroup::CVxGraphGroup()
{
	clearGraphGroup();
}

bool CVxGraphGroup::initGraphGroup(i32 count)
{
	clearGraphGroup();
	if (!CPOBase::isPositive(count))
	{
		return false;
	}

	m_graph_vec.resize(count);
	for (i32 i = 0; i < count; i++)
	{
		m_graph_vec[i] = NULL;
	}
	return true;
}

bool CVxGraphGroup::clearGraphGroup()
{
	m_graph_vec.clear();
	return true;
}

COpenVxGraph* CVxGraphGroup::getFreeGraph()
{
	i32 i, count = (i32)m_graph_vec.size();
	if (!CPOBase::isPositive(count))
	{
		return NULL;
	}

	COpenVxGraph* graph_ptr = NULL;
	for (i = 0; i < count; i++)
	{
		graph_ptr = m_graph_vec[i];
		if (graph_ptr && graph_ptr->isFree())
		{
			return graph_ptr;
		}
	}

	graph_ptr = m_graph_vec[0];
	if (graph_ptr)
	{
		graph_ptr->waitFinish();
	}
	return graph_ptr;
}

bool CVxGraphGroup::waitGroup()
{
	COpenVxGraph* graph_ptr = NULL;
	i32 i, count = (i32)m_graph_vec.size();
	for (i = 0; i < count; i++)
	{
		graph_ptr = m_graph_vec[i];
		if (graph_ptr)
		{
			graph_ptr->waitFinish();
		}
	}
	return true;
}
#endif
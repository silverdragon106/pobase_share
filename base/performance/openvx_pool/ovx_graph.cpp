#include "ovx_graph.h"
#include "ovx_node.h"
#include "ovx_resource_pool.h"

#if defined(POR_WITH_OVX)
OvxGraph::OvxGraph(OvxResourcePool* resource_pool_ptr, i32 graph_type)
{
	m_graph_type = graph_type;
	m_res_pool_ptr = resource_pool_ptr;
	if (resource_pool_ptr)
	{
		m_context_ptr = resource_pool_ptr->getContext();
	}
	else
	{
		m_context_ptr = NULL;
	}
	
	m_graph = NULL;
	m_graph_seed = -1;
	m_status = kStatusNone;
	m_lifetime_ms = 0;
	m_nodes.clear();
	m_resources.clear();
}

OvxGraph::~OvxGraph()
{
    destroy();
}

bool OvxGraph::process()
{
    if (m_status == kStatusReady || m_status == kStatusFinished)
    {
        m_status = kStatusProcessing;
		if (vxProcessGraph(m_graph) == VX_SUCCESS)
		{
			m_graph_seed = -1;
			m_status = kStatusFinished;
			return finished();
		}

		m_graph_seed = -1;
		m_status = kStatusFinished;
    }
	return false;
}

bool OvxGraph::schedule(i32 seed)
{
    if (m_status == kStatusReady || m_status == kStatusFinished)
    {
        if (vxScheduleGraph(m_graph) == VX_SUCCESS)
        {
			m_graph_seed = seed;
            m_status = kStatusScheduling;
			return true;
        }
    }
	return false;
}

bool OvxGraph::waitFinish()
{
	if (m_status != kStatusScheduling)
	{
		return false;
	}

	if (vxWaitGraph(m_graph) != VX_SUCCESS)
	{
		m_graph_seed = -1;
		m_status = kStatusFinished;
		return false;
	}
	m_graph_seed = -1;
	m_status = kStatusFinished;
	return finished();
}

bool OvxGraph::finished()
{
	return false;
}

bool OvxGraph::create()
{
	if (!m_context_ptr)
	{
		return false;
	}
    m_graph = vxCreateGraph(m_context_ptr->getVxContext());
	if (!m_graph)
	{
		return false;
	}

	m_graph_seed = -1;
	m_status = kStatusCreated;
	return true;
}

bool OvxGraph::destroy()
{
	vx_status status = VX_SUCCESS;

	m_graph_seed = -1;
	m_status = kStatusNone;
	m_res_pool_ptr->releaseResource(this);
	POSAFE_CLEAR(m_nodes);
	
	if (m_graph)
	{
		//Releases a reference to a graph. 
		//The object may not be garbage collected until its total reference count is zero.
		//Once the reference count is zero, 
		//all node references in the graph are automatically released as well.
		//Data referenced by those nodes may not be released as the user may have external references to the data
		status = vxReleaseGraph(&m_graph);
	}
	return (status == VX_SUCCESS);
}

bool OvxGraph::verify()
{
	switch ((i32)m_status)
	{
		case kStatusNone:
		{
			return false;
		}
		case kStatusCreated:
		{
			if (vxVerifyGraph(m_graph) == VX_SUCCESS)
			{
				m_graph_seed = -1;
				m_status = kStatusReady;
				return true;
			}
			return false;
		}
	}
	return true;
}

bool OvxGraph::checkVerified()
{
	if (m_status < kStatusReady)
	{
		destroy();
		create();
		return false;
	}
	return true;
}

bool OvxGraph::checkOnly(va_list args)
{
	if (m_status < kStatusReady || !check(args))
	{
		return false;
	}
	return true;
}

bool OvxGraph::check(va_list args)
{
	return false;
}

bool OvxGraph::prepare(va_list args)
{
	return false;
}

void OvxGraph::printPerf()
{
    vx_perf_t perf;
    vxQueryGraph(m_graph, VX_GRAPH_PERFORMANCE, &perf, sizeof(perf));
	printlog_lvs2(QString(" Graph[%1] ExecutionTime: %2")
					.arg(m_graph_type).arg(perf.avg*0.000001f, 0, 'g', 3), LOG_SCOPE_OVX);

    for (i32 i = 0; i < m_nodes.size(); ++i)
    {
        m_nodes[i]->printPerf();
    }
}

bool OvxGraph::addNode(OvxNode* node_ptr)
{
	m_nodes.push_back(node_ptr);
	return true;
}

bool OvxGraph::addNode(vx_node node, const postring& name)
{
    return addNode(po_new OvxNode(this, name, node));
}

OvxNode* OvxGraph::lastNode()
{
	if (m_nodes.size() > 0)
	{
		return m_nodes.back();
	}
	return NULL;
}

OvxNode* OvxGraph::getNodeAt(i32 index)
{
	if (index >= 0 && index < m_nodes.size())
	{
		return m_nodes.at(index);
	}
	return NULL;
}

vx_context OvxGraph::getVxContext()
{
	if (!m_context_ptr)
	{
		return NULL;
	}
    return m_context_ptr->getVxContext();
}

vx_graph OvxGraph::getVxGraph()
{
    return m_graph;
}

void OvxGraph::addResource(OvxResource* resource_ptr)
{
	m_resources.push_back(resource_ptr);
}

std::vector<OvxResource*> OvxGraph::getResources()
{
	return m_resources;
}

void OvxGraph::freeResources()
{
	POSAFE_CLEAR(m_resources);
}

void OvxGraph::clearResources()
{
	m_resources.clear();
}

void OvxGraph::updateLifeTime(i64 cur_time_ms)
{
	if (!m_context_ptr)
	{
		return;
	}
	m_lifetime_ms = cur_time_ms + OVX_GRAPH_CACHED_MS;
}
#endif

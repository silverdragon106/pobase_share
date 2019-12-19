#include "ovx_node.h"
#include "ovx_graph.h"

#if defined(POR_WITH_OVX)
OvxNode::OvxNode()
{
	m_name = "";
	m_graph = NULL;
	m_node = NULL;
}

OvxNode::OvxNode(OvxGraph* graph, const postring& name, vx_node node)
{
	m_graph = graph;
	m_name = name;
	m_node = node;
}

OvxNode::~OvxNode()
{
	vxReleaseNode(&m_node);
}

void OvxNode::printPerf()
{
    vx_perf_t perf;
    vxQueryNode(m_node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
	printlog_lvs2(QString("		Node[%1] Time: %2").arg(QString::fromStdString(m_name))
					.arg(perf.avg*0.000001f, 0, 'g', 3), LOG_SCOPE_OVX);
}
#endif
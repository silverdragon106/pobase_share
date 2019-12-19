#pragma once
#include "ovx_base.h"
#include "ovx_object.h"

#if defined(POR_WITH_OVX)
class OvxGraph;
class OvxNode : public OvxObject
{
public:
	OvxNode();
	OvxNode(OvxGraph* graph, const postring& name, vx_node node = NULL);
	virtual ~OvxNode();

	/* typecase operators */
    operator					vx_node() { return m_node; }

	/* node methods */
    void						printPerf();

	inline vx_node				getVxNode() { return m_node; };
	inline postring				getNodeName() { return m_name; };
	inline OvxGraph*			getGraph() { return m_graph; };

private:
	vx_node						m_node;
	postring					m_name;
	OvxGraph*					m_graph;
};
typedef std::vector<OvxNode*> OvxNodeVec;
#endif
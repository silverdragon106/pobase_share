#pragma once
#include "ovx_base.h"
#include "ovx_object.h"
#include "ovx_context.h"
#include "ovx_node.h"

#if defined(POR_WITH_OVX)
class OvxResource;
class OvxResourcePool;
class OvxGraph : public OvxObject
{
public:
    enum GraphStatus
    {
        kStatusNone,
		kStatusCreated,
        kStatusReady,
        kStatusScheduling,
        kStatusProcessing,
        kStatusFinished,
    };

public:
	OvxGraph(OvxResourcePool* resource_pool_ptr, i32 graph_type);
	virtual ~OvxGraph();
	
	/* process methods for OpenVX graph*/
	virtual bool				process();
	virtual bool				schedule(i32 seed = -1);
	virtual bool				waitFinish();

	virtual bool				check(va_list args);
	virtual bool				prepare(va_list args);
	virtual bool				finished();

	bool						create();
	bool						destroy();
	bool						verify();
	
	void						printPerf();
	bool						checkVerified();
	bool						checkOnly(va_list args);
	
	/* child node related methods */
	bool						addNode(OvxNode* node_ptr);
    bool						addNode(vx_node node, const postring& name);
    OvxNode*					lastNode();
    OvxNode*					getNodeAt(i32 index);

	void						updateLifeTime(i64 cur_time_ms);
	void						addResource(OvxResource* resource_ptr);
	void						freeResources();
	void						clearResources();
	std::vector<OvxResource*>	getResources();

    vx_graph					getVxGraph();
    vx_context					getVxContext();

	/* typecast operators */
	inline operator				vx_graph() { return m_graph; }
	inline OvxResourcePool*		getResourcePool() { return m_res_pool_ptr; };

	inline vx_graph				getVxGraph() const { return m_graph; };
	inline i32					getType() const { return m_graph_type; };
	inline i64					getLifeTime() const { return m_lifetime_ms; };
	inline i32					getSeed() const { return m_graph_seed; };
	
	inline bool					isReady() const     { return m_status == kStatusReady; };
	inline bool					isNone() const      { return m_status == kStatusNone; };
	inline bool					isFinished() const  { return m_status == kStatusFinished; };
	inline bool					isWaiting() const   { return m_status == kStatusScheduling; };
	inline bool					isWaiting(i32 seed) const { return m_status == kStatusScheduling && m_graph_seed == seed; };

protected:
	OvxContext*					m_context_ptr;
	OvxResourcePool*			m_res_pool_ptr;

	std::vector<OvxNode*>		m_nodes;
	std::vector<OvxResource*>	m_resources;

    vx_graph					m_graph;
	i32							m_graph_type;
	std::atomic<i32>			m_graph_seed;
    std::atomic<i32>			m_status;
	i64							m_lifetime_ms;
};
#endif

#pragma once
#include "ovx_base.h"
#include "ovx_context.h"
#include "ovx_graph.h"
#include "ovx_resource_pool.h"
#include "performance/vx_kernels/vx_kernel_types.h"

enum GraphType
{
	kGImgProcCommon = 0,

	kGImgProcErode = 10,
	kGImgProcDilate,
	kGImgProcClose,
	kGImgProcOpen,
	kGImgProcRemap,
	kGImgProcCvtImg2RunTable,
	kGImgProcCvtRunTable2Img,
	kGImgProcCvtColor,
	kGImgProcCvtSplit,
	kGImgProcCvtHSVSplit,
	kGImgProcCvtIntensity,

	kGImgProcExtend = 100
};

#if defined(POR_WITH_OVX)
class OvxGraphPool : public OvxObject
{
public:
	static const i32 kGPMaxQueueSize10 = 10;
	static const i32 kGPMaxQueueSize20 = 20;
	static const i32 kGPMaxQueueSize40 = 40;
	static const i32 kGPMaxQueueSize60 = 60;
	static const i32 kGPMaxQueueSize80 = 80;

public:
	typedef std::vector<OvxGraph*>		GraphPoolVector;
	typedef std::list<OvxGraph*>		GraphPoolList;
	typedef GraphPoolList::iterator		GraphPoolListIter;

public:
	OvxGraphPool();
	virtual ~OvxGraphPool();

    bool						create(OvxContext* context_ptr, OvxResourcePool* resource_pool_ptr,
									i32 max_queue_size = kGPMaxQueueSize20);
	bool						destroy();

	void						printStats();

	OvxGraph*					fetchGraph(i32 graph_type, ...);
	bool						releaseGraph(i32 graph_type);
	bool						releaseGraph(OvxGraph* graph_ptr);
	bool						releaseSeedGraphs(i32 seed);
	bool						freeGraph(i32 graph_type);
	bool						freeGraph(OvxGraph* graph_ptr);

	static vx_status			addGradientGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_grad);
	static vx_status			addGaussianGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 kernel_size);
	static vx_status			addDilateGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 dilate_window);
	static vx_status			addErodeGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 erode_window);
	static vx_status			addOpenGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 erode_window, i32 dilate_window =  -1);
	static vx_status			addCloseGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 dilate_window, i32 erode_window = -1);

	virtual OvxGraph*			createGraph(i32 graph_type);
	
	inline i32					getNextSeed() { return m_seed++; };
	inline OvxContext*			getContext() { return m_context_ptr; }
	inline OvxResourcePool*		getResourcePool() { return m_resource_pool_ptr; }

protected:
	bool						_checkGraphPool(i64 cur_time_ms);
	bool						_isIdleFull() const;
	void						_freeOldestIdleGraph();

protected:
	bool						m_is_inited;
	i32							m_seed;
	i32							m_max_queue_size;

	POMutex						m_graph_mutex;
	GraphPoolList				m_fetch_graphs;
	GraphPoolList				m_idle_graphs;

	OvxContext*					m_context_ptr;
	OvxResourcePool*			m_resource_pool_ptr;
};
extern OvxGraphPool* g_vx_gpool_ptr;

//////////////////////////////////////////////////////////////////////////
class CGImgProcErode : public OvxGraph
{
public:
	CGImgProcErode(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcErode();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	u8*							m_dst_img_ptr;
	i32							m_erode_window;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcDilate : public OvxGraph
{
public:
	CGImgProcDilate(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcDilate();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	u8*							m_dst_img_ptr;
	i32							m_dilate_window;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcClose : public OvxGraph
{
public:
	CGImgProcClose(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcClose();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	u8*							m_dst_img_ptr;
	i32							m_dilate_window;
	i32							m_erode_window;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcOpen : public OvxGraph
{
public:
	CGImgProcOpen(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcOpen();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	u8*							m_dst_img_ptr;
	i32							m_dilate_window;
	i32							m_erode_window;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcRemap : public OvxGraph
{
public:
	CGImgProcRemap(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcRemap();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	i32							m_channel;
	u8*							m_dst_img_ptr;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtImg2RunTable : public OvxGraph
{
public:
	CGImgProcCvtImg2RunTable(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtImg2RunTable();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	void*						m_dst_run_table;

	vx_image					m_vx_src;
	vx_image					m_vx_src_mask;
	vx_array					m_vx_param;
	vx_array					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtRunTable2Img : public OvxGraph
{
public:
	CGImgProcCvtRunTable2Img(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtRunTable2Img();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	u8*							m_dst_img_ptr;

	vx_array					m_vx_src;
	vx_array					m_vx_param;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtColor : public OvxGraph
{
public:
	CGImgProcCvtColor(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtColor();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	i32							m_src_format;
	i32							m_dst_format;
	i32							m_dst_channel;
	void*						m_dst_img_ptr;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtSplit : public OvxGraph
{
public:
	CGImgProcCvtSplit(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtSplit();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	i32							m_src_format;
	i32							m_dst_channel;
	void*						m_dst_img_ptr;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtHSVSplit : public OvxGraph
{
public:
	CGImgProcCvtHSVSplit(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtHSVSplit();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	i32							m_src_format;
	i32							m_dst_channel;
	void*						m_dst_img_ptr;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};

//////////////////////////////////////////////////////////////////////////
class CGImgProcCvtIntensity : public OvxGraph
{
public:
	CGImgProcCvtIntensity(OvxResourcePool* resource_pool_ptr, i32 graph_type);
    virtual ~CGImgProcCvtIntensity();

	virtual bool				prepare(va_list args);
	virtual bool				check(va_list args);
	virtual	bool				finished();

public:
	i32							m_width;
	i32							m_height;
	i32							m_src_format;
	void*						m_dst_img_ptr;

	vx_image					m_vx_src;
	vx_image					m_vx_dst;
};
#endif

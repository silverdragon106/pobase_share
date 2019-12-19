#pragma once
#include "ovx_base.h"
#include "ovx_graph.h"
#include <mutex>

#if defined(POR_WITH_OVX)
enum OvxResourceType
{
	kOvxResourceNone,
	kOvxResourceImage,
	kOvxResourceScalar,
	kOvxResourceThreshold,
	kOvxResourceMatrix,
	kOvxResourceArray,
	kOvxResourceObjectArray,
	kOvxResourcePyramid
};

class OvxResourcePool;
class OvxResource : public OvxObject
{
public:
	OvxResource();
	~OvxResource();

	vx_image					getImage();
	vx_array					getArray();
	vx_matrix					getMatrix();
	vx_pyramid					getPyramid();

	void						updateLifeTime(i64 cur_time_ms);

	bool						operator==(const OvxResource& obj);
	
	inline i64					getLifeTime() const { return m_life_time_ms; };

	inline bool					isVirtual() const { return m_is_virtual; };
	inline bool					isFetchable() const { return m_is_fetch; };
	inline bool					isNull() const { return m_resource == NULL; };

public:
	OvxResourceType				m_type;			/* type of resource object. */
	vx_reference				m_resource;		/* resource object. */
	bool						m_is_virtual;	/* owned object of resource object. Can be one of OvxContext or OvxGraph. */
	bool						m_is_fetch;
	i64							m_life_time_ms;

	OvxResourcePool*			m_resource_pool;
};

class OvxContext;
class OvxResourcePool : public OvxObject
{
public:
	static const i32 kRPMaxQueueSize25		= 25;
	static const i32 kRPMaxQueueSize50		= 50;
	static const i32 kRPMaxQueueSize100		= 100;
	static const i32 kRPMaxQueueSize150		= 150;
	static const i32 kRPMaxQueueSize200		= 200;
	static const i32 kRPBaseImageSize64		= 64;
	static const i32 kRPBaseImageSize128	= 128;
	static const i32 kRPBaseImageSize256	= 256;
	static const i32 kRPBaseImageSize512	= 512;
	static const i32 kRPBaseArraySize16		= 16;
	static const i32 kRPBaseArraySize32		= 32;
	static const i32 kRPBaseArraySize64		= 64;
	static const i32 kRPBaseArraySize128	= 128;
	static const i32 kRPFetchMinSize		= 10240; //10k

	typedef std::vector<OvxResource*>		OvxResourceVec;
	typedef OvxResourceVec::iterator		OvxResourceVecIter;
	
	typedef std::list<OvxResource*>			OvxResourceList;
	typedef OvxResourceList::iterator		OvxResourceListIter;
		
public:
	OvxResourcePool();
	virtual ~OvxResourcePool();

	bool						create(OvxContext* context_ptr, i32 max_queue_size = kRPMaxQueueSize50,
									i32 base_img_size = kRPBaseImageSize128, i32 base_array_size = kRPBaseArraySize32,
									i32 fetch_min_size = kRPFetchMinSize);
	bool						destroy();

	void						printStats();

	bool						checkImageSize(vx_image vx_img, i32 w, i32 h);

	vx_image					fetchImage(OvxGraph* graph_ptr, i32 width, i32 height, i32 format, bool is_strict = false, bool is_virtual = false);
	vx_pyramid					fetchPyramid(OvxGraph* graph_ptr, i32 width, i32 height, f32 scale, i32 level, i32 format, bool is_virtual = false);
	vx_array					fetchArray(OvxGraph* graph_ptr, i32 type, i32 size, bool is_virtual = false);
	vx_matrix					fetchMatrix(OvxGraph* graph_ptr, i32 width, i32 height, i32 format);
	vx_scalar					fetchScalar(OvxGraph* graph_ptr, i32 type);
	vx_threshold				fetchThreshold(OvxGraph* graph_ptr, i32 threshold_type, i32 type);

	OvxResource*				fetchImage(vx_context context, i32 width, i32 height, i32 format, bool is_strict = false, bool is_virtual = false);
	OvxResource*				fetchPyramid(vx_context context, i32 width, i32 height, f32 scale, i32 level, i32 format, bool is_virtual = false);
	OvxResource*				fetchArray(vx_context context, i32 type, i32 size, bool is_virtual = false);
	OvxResource*				fetchMatrix(vx_context context, i32 width, i32 height, i32 format);
	OvxResource*				fetchScalar(vx_context context, i32 type);
	OvxResource*				fetchThreshold(vx_context context, i32 threshold_type, i32 type);

	OvxResource*				createPyramid(vx_context context, i32 width, i32 height, f32 scale, i32 level, i32 format);

	void						releaseResource(OvxGraph* graph_ptr);
	bool						releaseResource(OvxResource* resource);
	bool						freeResource(OvxGraph* graph_ptr);
	bool						freeResource(OvxResource* resource);

	inline OvxContext*			getContext() { return m_context_ptr; }

	template <typename T>
	vx_scalar fetchScalar(OvxGraph* graph_ptr, i32 type, T val)
	{
		if (!graph_ptr)
		{
			return NULL;
		}

		OvxResource* res_ptr = fetchScalar(graph_ptr->getVxContext(), type, val);
		graph_ptr->addResource(res_ptr);
		return (vx_scalar)res_ptr->m_resource;
	}

	template <typename T>
	OvxResource* fetchScalar(vx_context context, i32 type, T val)
	{
		OvxResource* res_ptr = po_new OvxResource();
		res_ptr->m_resource_pool = this;
		res_ptr->m_type = kOvxResourceScalar;
		res_ptr->m_resource = (vx_reference)vxCreateScalar(context, type, &val);
		if (!res_ptr->m_resource)
		{
			POSAFE_DELETE(res_ptr);
			return NULL;
		}
		return res_ptr;
	}

private:
	bool						_fetchResource(OvxResource* resource);

	bool						_checkResourcePool(i64 cur_time_ms);
	bool						_isIdleFull();
	bool						_freeOldestIdleResource();

	OvxResource*				_findIdleImage(i32 width, i32 height, i32 format, bool is_strict);
	OvxResource*				_findIdlePyramid(i32 width, i32 height, f32 scale, i32 level, i32 format);
	OvxResource*				_findIdleArray(i32 arr_size, i32 arr_item_type);
	OvxResource*				_findIdleMatrix(i32 width, i32 height, i32 format);
	
private:
	bool						m_is_inited;
	i32							m_max_queue_size;
	i32							m_base_img_size;
	i32							m_base_array_size;
	i32							m_fetch_min_size;

	POMutex						m_resource_mutex;
	OvxResourceList				m_fetch_resources;
	OvxResourceList				m_idle_resources;

	OvxContext*					m_context_ptr;
};
#endif

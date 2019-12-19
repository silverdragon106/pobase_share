#include "ovx_resource_pool.h"
#include "ovx_base.h"
#include "ovx_graph.h"
#include "ovx_context.h"

#if defined(POR_WITH_OVX)
//#define GRAPH_OR_CONTEXT_REF (is_virtual ? graph : graph->getContext())
/************************************************************************/
/* OvxResource                                                          */
/************************************************************************/
OvxResource::OvxResource()
{
	m_type = kOvxResourceNone;
	m_resource = NULL;
	m_is_virtual = false;
	m_is_fetch = false;
	m_life_time_ms = 0;
	m_resource_pool = NULL;
}

OvxResource::~OvxResource()
{
	POVX_RELEASE(m_resource);
}

bool OvxResource::operator==(const OvxResource& obj)
{
	return (m_object_id == obj.m_object_id &&
		m_type == obj.m_type && m_resource == obj.m_resource);
}

vx_image OvxResource::getImage()
{
	if (m_type != kOvxResourceImage)
	{
		return NULL;
	}
	return (vx_image)m_resource;
}

vx_array OvxResource::getArray()
{
	if (m_type != kOvxResourceArray)
	{
		return NULL;
	}
	return (vx_array)m_resource;
}

vx_matrix OvxResource::getMatrix()
{
	if (m_type != kOvxResourceMatrix)
	{
		return NULL;
	}
	return (vx_matrix)m_resource;
}

vx_pyramid OvxResource::getPyramid()
{
	if (m_type != kOvxResourcePyramid)
	{
		return NULL;
	}
	return (vx_pyramid)m_resource;
}

void OvxResource::updateLifeTime(i64 cur_time_ms)
{
	m_life_time_ms = cur_time_ms + OVX_RESOURCE_CACHED_MS;
}

//////////////////////////////////////////////////////////////////////////
/* OvxResourcePool                                                      */

OvxResourcePool::OvxResourcePool()
{
	m_is_inited = false;
	m_max_queue_size = 0;
	m_base_img_size = kRPBaseImageSize128;
	m_base_array_size = kRPBaseArraySize32;
	m_fetch_min_size = kRPFetchMinSize;

	m_fetch_resources.clear();
	m_idle_resources.clear();

	m_context_ptr = NULL;
}

OvxResourcePool::~OvxResourcePool()
{
	destroy();
}

bool OvxResourcePool::create(OvxContext* context_ptr, i32 max_queue_size,
						i32 base_img_size, i32 base_array_size, i32 fetch_min_size)
{
	destroy();
	if (!m_is_inited)
	{
		m_context_ptr = context_ptr;
		m_max_queue_size = max_queue_size;
		m_base_img_size = base_img_size;
		m_base_array_size = base_array_size;
		m_fetch_min_size = fetch_min_size;
		m_is_inited = true;
	}
	return true;
}

bool OvxResourcePool::destroy()
{
	if (m_is_inited)
	{
		exlock_guard(m_resource_mutex);
		m_is_inited = false;
		std::list<OvxResource*>::iterator iter;

		//remove all fetch resources
		for (iter = m_fetch_resources.begin(); iter != m_fetch_resources.end(); ++iter)
		{
			OvxResource* res_ptr = *iter;
			if (res_ptr->isNull())
			{
				continue;
			}
			POSAFE_DELETE(res_ptr);
		}
		m_fetch_resources.clear();

		//remove all idle resources
		for (iter = m_idle_resources.begin(); iter != m_idle_resources.end(); ++iter)
		{
			OvxResource* res_ptr = *iter;
			if (res_ptr->isNull())
			{
				continue;
			}
			POSAFE_DELETE(res_ptr);
		}
		m_idle_resources.clear();
	}
	return true;
}

bool OvxResourcePool::checkImageSize(vx_image vx_img, i32 w, i32 h)
{
	i32 max_w = po::_max(OVX_RESOURCE_MAX_RANGE * w, w + m_base_img_size);
	i32 max_h = po::_max(OVX_RESOURCE_MAX_RANGE * h, h + m_base_img_size);
	i32 vx_width = OvxHelper::getWidth(vx_img);
	i32 vx_height = OvxHelper::getHeight(vx_img);
	return (vx_width >= w && vx_width <= max_w && vx_height >= h && vx_height <= max_h);
}

OvxResource* OvxResourcePool::_findIdleImage(i32 width, i32 height, i32 format, bool is_strict)
{
	i32 max_w = po::_max(OVX_RESOURCE_MAX_RANGE * width, width + m_base_img_size);
	i32 max_h = po::_max(OVX_RESOURCE_MAX_RANGE * height, height + m_base_img_size);
	OvxResourcePool::OvxResourceListIter iter;
	for (iter = m_idle_resources.begin(); iter != m_idle_resources.end(); ++iter)
	{
		OvxResource* res_ptr = *iter;
		if (res_ptr->m_type == kOvxResourceImage)
		{
			vx_image vx_img = (vx_image)(res_ptr->m_resource);
			if (is_strict)
			{
				if (OvxHelper::getWidth(vx_img) == width && OvxHelper::getHeight(vx_img) == height &&
					OvxHelper::getFormat(vx_img) == format)
				{
					return res_ptr;
				}
			}
			else
			{
				i32 w = OvxHelper::getWidth(vx_img);
				i32 h = OvxHelper::getHeight(vx_img);
				if (OvxHelper::getFormat(vx_img) == format && w >= width && w <= max_w && h >= height && h <= max_h)
				{
					return res_ptr;
				}
			}
		}
	}
	return NULL;
}

OvxResource* OvxResourcePool::_findIdlePyramid(i32 width, i32 height, f32 scale, i32 level, i32 format)
{
	OvxResourcePool::OvxResourceListIter iter;
	for (iter = m_idle_resources.begin(); iter != m_idle_resources.end(); ++iter)
	{
		OvxResource* res_ptr = *iter;
		if (res_ptr->m_type == kOvxResourcePyramid)
		{
			vx_pyramid vx_pyr = (vx_pyramid)(res_ptr->m_resource);
			if (OvxHelper::getWidth(vx_pyr) == width && OvxHelper::getHeight(vx_pyr) == height &&
				OvxHelper::getFormat(vx_pyr) == format && OvxHelper::getLevel(vx_pyr) == level &&
				OvxHelper::getScale(vx_pyr) == scale)
			{
				return res_ptr;
			}
		}
	}
	return NULL;
}

OvxResource* OvxResourcePool::_findIdleArray(i32 arr_size, i32 arr_item_type)
{
	i32 max_size = po::_max(OVX_RESOURCE_MAX_RANGE * arr_size, arr_size + m_base_array_size);
	OvxResourcePool::OvxResourceListIter iter;
	for (iter = m_idle_resources.begin(); iter != m_idle_resources.end(); ++iter)
	{
		OvxResource* res_ptr = *iter;
		if (res_ptr->m_type == kOvxResourceArray)
		{
			vx_array vx_arr = (vx_array)(res_ptr->m_resource);
			i32 cap_size = OvxHelper::getCapacity(vx_arr);
			if (OvxHelper::getFormat(vx_arr) == arr_item_type && cap_size >= arr_size && cap_size <= max_size)
			{
				OvxHelper::clearArray(vx_arr);
				return res_ptr;
			}
		}
	}
	return NULL;
}

OvxResource* OvxResourcePool::_findIdleMatrix(i32 width, i32 height, i32 format)
{
	OvxResourcePool::OvxResourceListIter iter;
	for (iter = m_idle_resources.begin(); iter != m_idle_resources.end(); ++iter)
	{
		OvxResource* res_ptr = *iter;
		if (res_ptr->m_type == kOvxResourceMatrix)
		{
			vx_matrix vx_mat = (vx_matrix)(res_ptr->m_resource);
			if (OvxHelper::getWidth(vx_mat) == width && OvxHelper::getHeight(vx_mat) == height &&
				OvxHelper::getFormat(vx_mat) == format)
			{
				return (*iter);
			}
		}
	}
	return NULL;
}

vx_image OvxResourcePool::fetchImage(OvxGraph* graph_ptr, i32 width, i32 height, i32 format, bool is_strict, bool is_virtual)
{
	if (!graph_ptr || width * height <= 0)
	{
		return NULL;
	}
	
	vx_context context = is_virtual ? (vx_context)(graph_ptr->getVxGraph()) : graph_ptr->getVxContext();
	OvxResource* res_ptr = fetchImage(context, width, height, format, is_strict, is_virtual);
	graph_ptr->addResource(res_ptr);

	return (vx_image)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchImage(vx_context context, i32 width, i32 height, i32 format, bool is_strict, bool is_virtual)
{
	if (!context || width * height <= 0)
	{
		return NULL;
	}

	if (!is_strict)
	{
		width = CPOBase::round(width, m_base_img_size);
		height = CPOBase::round(height, m_base_img_size);
	}

	OvxResource* res_ptr = NULL;
	i32 fetch_size = width*height*OvxHelper::sizeImageElem(format);
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		/* Search re-usable reference in Free Resources */
		exlock_guard(m_resource_mutex);
		res_ptr = _findIdleImage(width, height, format, is_strict);
		if (_fetchResource(res_ptr))
		{
			return res_ptr;
		}
	}
		
	/* Add new resouce */
	res_ptr = po_new OvxResource();
	res_ptr->m_type = kOvxResourceImage;
	res_ptr->m_resource_pool = this;
	res_ptr->m_is_virtual = is_virtual;

	if (is_virtual)
	{
		res_ptr->m_resource = (vx_reference)vxCreateVirtualImage((vx_graph)context, width, height, format);
	}
	else
	{
		res_ptr->m_resource = (vx_reference)vxCreateImage(context, width, height, format);
	}

	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreateImage Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		exlock_guard(m_resource_mutex);
		res_ptr->m_is_fetch = true;
		m_fetch_resources.push_back(res_ptr);
	}
	return res_ptr;
}

vx_pyramid OvxResourcePool::fetchPyramid(OvxGraph* graph_ptr, i32 width, i32 height, f32 scale, i32 level, i32 format, bool is_virtual)
{
	if (!graph_ptr || width * height <= 0 || level <= 0)
	{
		return NULL;
	}

	vx_context context = is_virtual ? (vx_context)(graph_ptr->getVxGraph()) : graph_ptr->getVxContext();
	OvxResource* res_ptr = fetchPyramid(context, width, height, scale, level, format, is_virtual);
	graph_ptr->addResource(res_ptr);

	return (vx_pyramid)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchPyramid(vx_context context, i32 width, i32 height, f32 scale, i32 level, i32 format, bool is_virtual)
{
	if (!context || width * height <= 0 || level <= 0)
	{
		return NULL;
	}

	OvxResource* res_ptr = NULL;
	i32 fetch_size = 2 * width*height*OvxHelper::sizeImageElem(format);
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		/* Search re-usable reference in Free Resources */
		exlock_guard(m_resource_mutex);
		res_ptr = _findIdlePyramid(width, height, scale, level, format);
		if (_fetchResource(res_ptr))
		{
			return res_ptr;
		}
	}

	/* Add new resouce */
	res_ptr = po_new OvxResource();
	res_ptr->m_type = kOvxResourcePyramid;
	res_ptr->m_resource_pool = this;
	res_ptr->m_is_virtual = is_virtual;

	if (is_virtual)
	{
		res_ptr->m_resource = (vx_reference)vxCreateVirtualPyramid(
												(vx_graph)context, level, scale, width, height, format);
	}
	else
	{
		res_ptr->m_resource = (vx_reference)vxCreatePyramid(context, level, scale, width, height, format);
	}

	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreatePyramid Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		exlock_guard(m_resource_mutex);
		res_ptr->m_is_fetch = true;
		m_fetch_resources.push_back(res_ptr);
	}
	return res_ptr;
}

vx_array OvxResourcePool::fetchArray(OvxGraph* graph_ptr, i32 type, i32 size, bool is_virtual)
{
	if (!graph_ptr || size <= 0)
	{
		return NULL;
	}

	vx_context context = is_virtual ? (vx_context)(graph_ptr->getVxGraph()) : graph_ptr->getVxContext();
	OvxResource* res_ptr = fetchArray(context, type, size, is_virtual);
	graph_ptr->addResource(res_ptr);
	return (vx_array)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchArray(vx_context context, i32 type, i32 size, bool is_virtual)
{
	if (!context || size <= 0)
	{
		return NULL;
	}

	size = CPOBase::round(size, m_base_array_size);
	
	OvxResource* res_ptr = NULL;
	i32 fetch_size = size*OvxHelper::sizeArrayElem(type);
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		/* Search re-usable reference in Free Resources */
		exlock_guard(m_resource_mutex);
		res_ptr = _findIdleArray(size, type);
		if (_fetchResource(res_ptr))
		{
			return res_ptr;
		}
	}
	
	/* Add new resouce */
	res_ptr = po_new OvxResource();
	res_ptr->m_type = kOvxResourceArray;
	res_ptr->m_resource_pool = this;
	res_ptr->m_is_virtual = is_virtual;
	if (is_virtual)
	{
		res_ptr->m_resource = (vx_reference)vxCreateVirtualArray((vx_graph)context, type, size);
	}
	else
	{
		res_ptr->m_resource = (vx_reference)vxCreateArray(context, type, size);
	}

	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreateArray Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	if (!is_virtual && fetch_size > m_fetch_min_size)
	{
		exlock_guard(m_resource_mutex);
		res_ptr->m_is_fetch = true;
		m_fetch_resources.push_back(res_ptr);
	}
	return res_ptr;
}

vx_matrix OvxResourcePool::fetchMatrix(OvxGraph* graph_ptr, i32 width, i32 height, i32 format)
{
	if (!graph_ptr || width * height <= 0)
	{
		return NULL;
	}

	OvxResource* res_ptr = fetchMatrix(graph_ptr->getVxContext(), format, width, height);
	graph_ptr->addResource(res_ptr);
	return (vx_matrix)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchMatrix(vx_context context, i32 width, i32 height, i32 format)
{
	if (!context || width * height <= 0)
	{
		return NULL;
	}

	OvxResource* res_ptr = NULL;
	i32 fetch_size = width * height * OvxHelper::sizeMatrixElem(format);
	if (fetch_size > m_fetch_min_size)
	{
		/* Search re-usable reference in Free Resources */
		exlock_guard(m_resource_mutex);
		res_ptr = _findIdleMatrix(width, height, format);
		if (_fetchResource(res_ptr))
		{
			return res_ptr;
		}
	}

	/* Add new resouce */
	res_ptr = po_new OvxResource();
	res_ptr->m_type = kOvxResourceMatrix;
	res_ptr->m_resource_pool = this;
	res_ptr->m_resource = (vx_reference)vxCreateMatrix(context, format, width, height);
	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreateMatrix Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}

	if (fetch_size > m_fetch_min_size)
	{
		exlock_guard(m_resource_mutex);
		res_ptr->m_is_fetch = true;
		m_fetch_resources.push_back(res_ptr);
	}
	return res_ptr;
}

vx_scalar OvxResourcePool::fetchScalar(OvxGraph* graph_ptr, i32 type)
{
	if (!graph_ptr)
	{
		return NULL;
	}

	OvxResource* res_ptr = fetchScalar(graph_ptr->getVxContext(), type);
	graph_ptr->addResource(res_ptr);
	return (vx_scalar)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchScalar(vx_context context, i32 type)
{
	if (!context)
	{
		return NULL;
	}

	OvxResource* res_ptr = po_new OvxResource();
	res_ptr->m_resource_pool = this;
	res_ptr->m_type = kOvxResourceScalar;
	res_ptr->m_resource = (vx_reference)vxCreateScalar(context, type, NULL);
	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreateScalar Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	return res_ptr;
}

vx_threshold OvxResourcePool::fetchThreshold(OvxGraph* graph_ptr, i32 threshold_type, i32 type)
{
	if (!graph_ptr)
	{
		return NULL;
	}

	OvxResource* res_ptr = fetchThreshold(graph_ptr->getVxContext(), threshold_type, type);
	graph_ptr->addResource(res_ptr);
	return (vx_threshold)res_ptr->m_resource;
}

OvxResource* OvxResourcePool::fetchThreshold(vx_context context, i32 threshold_type, i32 type)
{
	if (!context)
	{
		return NULL;
	}

	/* Add new resouce */
	OvxResource* res_ptr = po_new OvxResource();
	res_ptr->m_resource_pool = this;
	res_ptr->m_type = kOvxResourceThreshold;
	res_ptr->m_resource = (vx_reference)vxCreateThreshold(context, threshold_type, type);
	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreateThreshold Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	return res_ptr;
}

OvxResource* OvxResourcePool::createPyramid(vx_context context, i32 width, i32 height, f32 scale, i32 level, i32 format)
{
	if (!context || width * height <= 0 || level <= 0)
	{
		return NULL;
	}

	OvxResource* res_ptr = po_new OvxResource();
	res_ptr->m_type = kOvxResourcePyramid;
	res_ptr->m_resource_pool = NULL;
	res_ptr->m_is_virtual = false;
	res_ptr->m_resource = (vx_reference)vxCreatePyramid(context, level, scale, width, height, format);

	if (!res_ptr->m_resource)
	{
        printlog_lvs2("vxCreatePyramid Failed.", LOG_SCOPE_OVX);
		POSAFE_DELETE(res_ptr);
		return NULL;
	}
	return res_ptr;
}

bool OvxResourcePool::_isIdleFull()
{
	return (m_idle_resources.size() >= m_max_queue_size);
}

bool OvxResourcePool::_checkResourcePool(i64 cur_time_ms)
{
	if (m_idle_resources.size() < m_max_queue_size)
	{
		return true;
	}

	OvxResource* res_ptr = m_idle_resources.front();
	if (res_ptr->getLifeTime() < cur_time_ms)
	{
		m_idle_resources.pop_front();
		POSAFE_DELETE(res_ptr);
		return true;
	}
	return false;
}

bool OvxResourcePool::_freeOldestIdleResource()
{
	if (m_idle_resources.size() > 0)
	{
		OvxResource* res_ptr = m_idle_resources.front();
		if (res_ptr)
		{
			POSAFE_DELETE(res_ptr);
		}
		m_idle_resources.pop_front();
		return true;
	}
	return false;
}

void OvxResourcePool::releaseResource(OvxGraph* graph_ptr)
{
	if (!graph_ptr)
	{
		return;
	}

	OvxResourceVec vec;
	i64 cur_time_ms = sys_cur_time;
	{
		exlock_guard(m_resource_mutex);
		OvxResourceVec res_vec = graph_ptr->getResources();
		OvxResourceVecIter iter;
		vec.reserve(res_vec.size());

		for (iter = res_vec.begin(); iter != res_vec.end(); ++iter)
		{
			OvxResource* res_ptr = *iter;
			if (!res_ptr || res_ptr->isNull())
			{
				continue;
			}
			if (res_ptr->isFetchable())
			{
				m_fetch_resources.remove(res_ptr);
				if (_checkResourcePool(cur_time_ms))
				{
					res_ptr->updateLifeTime(cur_time_ms);
					m_idle_resources.push_back(res_ptr);
					continue;
				}
			}
			vec.push_back(res_ptr);
		}
	}

	//free resource
	i32 i, count = (i32)vec.size();
	for (i = 0; i < count; i++)
	{
		POSAFE_DELETE(vec[i]);
	}
	graph_ptr->clearResources();
}

bool OvxResourcePool::releaseResource(OvxResource* res_ptr)
{
	if (!res_ptr || res_ptr->isNull())
	{
		return false;
	}
	i64 cur_time_ms = sys_cur_time;

	if (res_ptr->isFetchable())
	{
		exlock_guard(m_resource_mutex);
		m_fetch_resources.remove(res_ptr);
		if (_checkResourcePool(cur_time_ms))
		{
			res_ptr->updateLifeTime(cur_time_ms);
			m_idle_resources.push_back(res_ptr);
			return true;
		}
	}
	POSAFE_DELETE(res_ptr);
	return true;
}

bool OvxResourcePool::freeResource(OvxGraph* graph_ptr)
{
	if (!graph_ptr)
	{
		return false;
	}

	{
		exlock_guard(m_resource_mutex);
		OvxResourceVec res_vec = graph_ptr->getResources();
		OvxResourceVecIter iter;
		for (iter = res_vec.begin(); iter != res_vec.end(); ++iter)
		{
			OvxResource* res_ptr = *iter;
			if (!res_ptr || res_ptr->isNull())
			{
				continue;
			}
			if (res_ptr->isFetchable())
			{
				m_fetch_resources.remove(res_ptr);
			}
		}
	}
	graph_ptr->freeResources();
	return true;
}

bool OvxResourcePool::freeResource(OvxResource* res_ptr)
{
	if (!res_ptr || res_ptr->isNull())
	{
		return false;
	}
	if (res_ptr->isFetchable())
	{
		exlock_guard(m_resource_mutex);
		m_fetch_resources.remove(res_ptr);
	}
	POSAFE_DELETE(res_ptr);
	return true;
}

bool OvxResourcePool::_fetchResource(OvxResource* res_ptr)
{
	if (!res_ptr || !res_ptr->isFetchable())
	{
		return false;
	}

	m_idle_resources.remove(res_ptr);
	m_fetch_resources.push_back(res_ptr);
	return true;
}

void OvxResourcePool::printStats()
{
	exlock_guard(m_resource_mutex);
	printlog_lvs2(QString("Resource Pool Result"), LOG_SCOPE_OVX);
	printlog_lvs2(QString("		Free Resource Count: %1").arg(m_idle_resources.size()), LOG_SCOPE_OVX);
	printlog_lvs2(QString("		Using Resource Count: %1").arg(m_fetch_resources.size()), LOG_SCOPE_OVX);
}
#endif

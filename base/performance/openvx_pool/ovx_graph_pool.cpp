#include "ovx_graph_pool.h"
#include "ovx_graph.h"
#include "ovx_node.h"
#include "ovx_resource_pool.h"
#include "struct/run_table.h"
#include "performance/vx_kernels/vx_kernels.h"

#if defined(POR_WITH_OVX)
OvxGraphPool* g_vx_gpool_ptr = NULL;

//////////////////////////////////////////////////////////////////////////
OvxGraphPool::OvxGraphPool()
{
	m_is_inited = false;
	m_seed = 0;
	m_max_queue_size = 0;

	m_fetch_graphs.clear();
	m_idle_graphs.clear();

	m_context_ptr = NULL;
	m_resource_pool_ptr = NULL;
	g_vx_gpool_ptr = this;
}

OvxGraphPool::~OvxGraphPool()
{
	destroy();
}

bool OvxGraphPool::create(OvxContext* context_ptr, OvxResourcePool* resource_pool_ptr,
						i32 max_queue_size)
{
	if (!m_is_inited)
	{
		m_context_ptr = context_ptr;
		m_resource_pool_ptr = resource_pool_ptr;
		m_max_queue_size = max_queue_size;
		m_seed = 0;
		m_is_inited = true;
	}
	return true;
}

bool OvxGraphPool::destroy()
{
	if (m_is_inited)
	{
		OvxGraph* graph_ptr;
		GraphPoolListIter iter;
		exlock_guard(m_graph_mutex);

		//remove all idle graphs
		for (iter = m_idle_graphs.begin(); iter != m_idle_graphs.end(); ++iter)
		{
			graph_ptr = *iter;
			m_resource_pool_ptr->freeResource(graph_ptr);
			POSAFE_DELETE(graph_ptr);
		}
		m_idle_graphs.clear();

		//remove all fetch graphs
		for (iter = m_fetch_graphs.begin(); iter != m_fetch_graphs.end(); ++iter)
		{
			graph_ptr = *iter;
			m_resource_pool_ptr->freeResource(graph_ptr);
			POSAFE_DELETE(graph_ptr);
		}
		m_fetch_graphs.clear();
		m_is_inited = false;
	}
	return true;
}

OvxGraph* OvxGraphPool::fetchGraph(i32 graph_type, ...)
{
	va_list args;
	va_start(args, graph_type);

	m_graph_mutex.lock();

	//find graph
	OvxGraph* graph_ptr;
	GraphPoolListIter iter;
	for (iter = m_idle_graphs.begin(); iter != m_idle_graphs.end(); ++iter)
	{
		graph_ptr = *iter;
		if (graph_ptr->getType() == graph_type && graph_ptr->checkOnly(args))
		{
			break;
		}
	}

	//remove free graph and add graph
	if (iter != m_idle_graphs.end())
	{
		OvxGraph* graph_ptr = *iter;
		m_fetch_graphs.push_back(*iter);
		m_idle_graphs.erase(iter);
		m_graph_mutex.unlock();
			
		graph_ptr->prepare(args);
		va_end(args);
		return graph_ptr;
	}
	m_graph_mutex.unlock();
	
	graph_ptr = createGraph(graph_type);
	if (graph_ptr)
	{
		if (!graph_ptr->prepare(args))
		{
			va_end(args);
			POSAFE_DELETE(graph_ptr);
			return NULL;
		}

		m_graph_mutex.lock();
		m_fetch_graphs.push_back(graph_ptr);
		m_graph_mutex.unlock();
	}
	va_end(args);
	return graph_ptr;
}

bool OvxGraphPool::releaseSeedGraphs(i32 seed)
{
	OvxGraph* graph_ptr;
	GraphPoolListIter iter;
	GraphPoolVector usage_vec, garbage_vec;

	//check release graphs
	{
		exlock_guard(m_graph_mutex);
		usage_vec.reserve(m_fetch_graphs.size());
		for (iter = m_fetch_graphs.begin(); iter != m_fetch_graphs.end();)
		{
			graph_ptr = *iter;
			if (graph_ptr->isWaiting(seed))
			{
				usage_vec.push_back(graph_ptr);
				iter = m_fetch_graphs.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	i32 i, count = (i32)usage_vec.size();
	if (!CPOBase::isPositive(count))
	{
		return true;
	}

	//until finish...
	i64 cur_time_ms = sys_cur_time;
	for (i = 0; i < count; i++)
	{
		usage_vec[i]->waitFinish();
	}

	//queue operation
	{
		garbage_vec.reserve(count);
		exlock_guard(m_graph_mutex);
		for (i = 0; i < count; i++)
		{
			graph_ptr = usage_vec[i];
			if (_checkGraphPool(cur_time_ms))
			{
				graph_ptr->updateLifeTime(cur_time_ms);
				m_idle_graphs.push_back(graph_ptr);
			}
			else
			{
				garbage_vec.push_back(graph_ptr);
			}
		}
	}

	//release resources
	count = (i32)garbage_vec.size();
	for (i = 0; i < count; i++)
	{
		graph_ptr = garbage_vec[i];
		m_resource_pool_ptr->releaseResource(graph_ptr);
		POSAFE_DELETE(graph_ptr);
	}
	return true;
}

bool OvxGraphPool::releaseGraph(i32 graph_type)
{
	OvxGraph* graph_ptr;
	GraphPoolListIter iter;
	GraphPoolVector vec1, vec2;

	//check release graphs
	{
		exlock_guard(m_graph_mutex);
		vec1.reserve(m_fetch_graphs.size());
		for (iter = m_fetch_graphs.begin(); iter != m_fetch_graphs.end();)
		{
			graph_ptr = *iter;
			if (graph_ptr->getType() == graph_type)
			{
				vec1.push_back(graph_ptr);
				iter = m_fetch_graphs.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	i32 i, count = (i32)vec1.size();
	if (!CPOBase::isPositive(count))
	{
		return true;
	}

	//until finish...
	i64 cur_time_ms = sys_cur_time;
	for (i = 0; i < count; i++)
	{
		vec1[i]->waitFinish();
	}

	//queue operation
	{
		vec2.reserve(count);
		exlock_guard(m_graph_mutex);
		for (i = 0; i < count; i++)
		{
			graph_ptr = vec1[i];
			if (_checkGraphPool(cur_time_ms))
			{
				graph_ptr->updateLifeTime(cur_time_ms);
				m_idle_graphs.push_back(graph_ptr);
			}
			else
			{
				vec2.push_back(graph_ptr);
			}
		}
	}

	//release resources
	count = (i32)vec2.size();
	for (i = 0; i < count; i++)
	{
		graph_ptr = vec2[i];
		m_resource_pool_ptr->releaseResource(graph_ptr);
		POSAFE_DELETE(graph_ptr);
	}
	return true;
}

bool OvxGraphPool::releaseGraph(OvxGraph* graph_ptr)
{
	if (!graph_ptr)
	{
		return false;
	}
	graph_ptr->waitFinish();
	i64 cur_time_ms = sys_cur_time;

	m_graph_mutex.lock();
	if (_checkGraphPool(cur_time_ms))
	{
		graph_ptr->updateLifeTime(cur_time_ms);
		m_idle_graphs.push_back(graph_ptr);
		m_fetch_graphs.remove(graph_ptr);
		m_graph_mutex.unlock();
	}
	else
	{
		m_fetch_graphs.remove(graph_ptr);
		m_graph_mutex.unlock();

		m_resource_pool_ptr->releaseResource(graph_ptr);
		POSAFE_DELETE(graph_ptr);
	}
	return true;
}

bool OvxGraphPool::freeGraph(i32 graph_type)
{
	OvxGraph* graph_ptr;
	GraphPoolListIter iter;
	GraphPoolVector vec;

	//find free graphs
	{
		exlock_guard(m_graph_mutex);
		vec.reserve(m_fetch_graphs.size());
		for (iter = m_fetch_graphs.begin(); iter != m_fetch_graphs.end();)
		{
			graph_ptr = *iter;
			if (graph_ptr->getType() == graph_type)
			{
				vec.push_back(graph_ptr);
				iter = m_fetch_graphs.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	//free graphs
	i32 i, count = (i32)vec.size();
	for (i = 0; i < count; i++);
	{
		graph_ptr = vec[i];
		m_resource_pool_ptr->freeResource(graph_ptr);
		POSAFE_DELETE(graph_ptr);
	}
	vec.clear();
	return false;
}

bool OvxGraphPool::freeGraph(OvxGraph* graph_ptr)
{
	if (!graph_ptr)
	{
		return false;
	}

	{
		exlock_guard(m_graph_mutex);
		m_fetch_graphs.remove(graph_ptr);
	}
	m_resource_pool_ptr->freeResource(graph_ptr);
	POSAFE_DELETE(graph_ptr);
	return true;
}

bool OvxGraphPool::_checkGraphPool(i64 cur_time_ms)
{
	if (m_idle_graphs.size() < m_max_queue_size)
	{
		return true;
	}

	OvxGraph* graph_ptr = m_idle_graphs.front();
	if (graph_ptr->getLifeTime() < cur_time_ms)
	{
		m_resource_pool_ptr->releaseResource(graph_ptr);
		m_idle_graphs.pop_front();
		POSAFE_DELETE(graph_ptr);
		return true;
	}
	return false;
}

bool OvxGraphPool::_isIdleFull() const
{
	return (m_idle_graphs.size() >= m_max_queue_size);
}

void OvxGraphPool::_freeOldestIdleGraph()
{
	if (m_idle_graphs.size() > 0)
	{
		OvxGraph* graph_ptr = m_idle_graphs.front();
		m_resource_pool_ptr->releaseResource(graph_ptr);
		m_idle_graphs.pop_front();
		POSAFE_DELETE(graph_ptr);
	}
}

void OvxGraphPool::printStats()
{
	exlock_guard(m_graph_mutex);
	printlog_lvs2(QString("Graph Pool Stats"), LOG_SCOPE_OVX);
	printlog_lvs2(QString("		Free Graph Count: %1").arg(m_idle_graphs.size()), LOG_SCOPE_OVX);
	printlog_lvs2(QString("		Using Graph Count: %1").arg(m_fetch_graphs.size()), LOG_SCOPE_OVX);
}

OvxGraph* OvxGraphPool::createGraph(i32 graph_type)
{
	switch (graph_type)
	{
		case kGImgProcErode:
		{
			return po_new CGImgProcErode(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcDilate:
		{
			return po_new CGImgProcDilate(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcClose:
		{
			return po_new CGImgProcClose(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcOpen:
		{
			return po_new CGImgProcOpen(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcRemap:
		{
			return po_new CGImgProcRemap(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcCvtImg2RunTable:
		{
#if defined(POVX_USE_IMG_TO_RUNTABLE)
			return po_new CGImgProcCvtImg2RunTable(m_resource_pool_ptr, graph_type);
#else
			return NULL;
#endif
		}

		case kGImgProcCvtRunTable2Img:
		{
#if defined(POVX_USE_RUNTABLE_TO_IMG)
			return po_new CGImgProcCvtRunTable2Img(m_resource_pool_ptr, graph_type);
#else
			return NULL;
#endif
		}
		case kGImgProcCvtColor:
		{
			return po_new CGImgProcCvtColor(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcCvtSplit:
		{
			return po_new CGImgProcCvtSplit(m_resource_pool_ptr, graph_type);
		}
		case kGImgProcCvtHSVSplit:
		{
#if defined(POVX_USE_CONVERT_TO_HSV)
			return po_new CGImgProcCvtHSVSplit(m_resource_pool_ptr, graph_type);
#else
			return NULL;
#endif
		}
		case kGImgProcCvtIntensity:
		{
#if defined(POVX_USE_CONVERT_TO_AVG)
			return po_new CGImgProcCvtIntensity(m_resource_pool_ptr, graph_type);
#else
			return NULL;
#endif
		}
	}
	return NULL;
}

vx_status OvxGraphPool::addGradientGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_grad)
{
	if (!graph_ptr || !vxi_src || !vxi_grad)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add resources
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 shift = 3;
	vx_image vxi_sobel_x = res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_S16, true, true);
	vx_image vxi_sobel_y = res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_S16, true, true);
	vx_image vxi_mag = res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_S16, true, true);
	vx_scalar vxs_shift = res_pool_ptr->fetchScalar(graph_ptr, VX_TYPE_INT32, shift);

	//add nodes
	graph_ptr->addNode(vxSobel3x3Node(graph_ptr->getVxGraph(), vxi_src, vxi_sobel_x, vxi_sobel_y), "vx.sobel3x3");
	status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);

	graph_ptr->addNode(vxMagnitudeNode(graph_ptr->getVxGraph(), vxi_sobel_x, vxi_sobel_y, vxi_mag), "vx.magnitude");
	graph_ptr->addNode(vxConvertDepthNode(graph_ptr->getVxGraph(), vxi_mag, vxi_grad, VX_CONVERT_POLICY_SATURATE, vxs_shift), "vx.convert");
	return status;
}

vx_status OvxGraphPool::addGaussianGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 kernel_window)
{
	if (!graph_ptr || !vxi_src || !vxi_dst || !CPOBase::isPositive(kernel_window))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add nodes
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 format = OvxHelper::getFormat(vxi_src);

	switch (format)
	{
		case VX_DF_IMAGE_U8:
		{
			vx_image vxi_prev = vxi_src;
			i32 i, iter = (kernel_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = (i == iter) ? vxi_dst :
							res_pool_ptr->fetchImage(graph_ptr, nw, nh, format, true, true);

				graph_ptr->addNode(vxGaussian3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.gaussian3x3");
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
				vxi_prev = vxi_next;
			}
			break;
		}
		case VX_DF_IMAGE_U16:
		{
			vx_scalar vx_kernel_size = res_pool_ptr->fetchScalar(graph_ptr, VX_TYPE_INT32, kernel_window);
			graph_ptr->addNode(OvxCustomNode::vxGaussian2dNode(graph_ptr->getVxGraph(), vxi_src, vxi_dst,
							vx_kernel_size), POVX_KERNEL_NAME_FILTER_GAUSSIAN2D);
			break;
		}
		default:
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}
	}
	return status;
}

vx_status OvxGraphPool::addDilateGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 dilate_window)
{
	if (!graph_ptr || !vxi_src || !vxi_dst || !CPOBase::isPositive(dilate_window))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add nodes
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 format = OvxHelper::getFormat(vxi_src);

	switch (format)
	{
		case VX_DF_IMAGE_U8:
		{
			i32 nw = OvxHelper::getWidth(vxi_src);
			i32 nh = OvxHelper::getHeight(vxi_src);

			vx_image vxi_prev = vxi_src;
			i32 i, iter = (dilate_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = (i == iter) ? vxi_dst :
							res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);

				graph_ptr->addNode(vxDilate3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.dilate3x3");
				vxi_prev = vxi_next;
#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}
			break;
		}
		default:
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}
	}
	return status;
}

vx_status OvxGraphPool::addErodeGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 erode_window)
{
	if (!graph_ptr || !vxi_src || !vxi_dst || !CPOBase::isPositive(erode_window))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add nodes
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 format = OvxHelper::getFormat(vxi_src);

	switch (format)
	{
		case VX_DF_IMAGE_U8:
		{
			i32 nw = OvxHelper::getWidth(vxi_src);
			i32 nh = OvxHelper::getHeight(vxi_src);

			vx_image vxi_prev = vxi_src;
			i32 i, iter = (erode_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = (i == iter) ? vxi_dst :
						res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);

				graph_ptr->addNode(vxErode3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.erode3x3");
				vxi_prev = vxi_next;
#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}
			break;
		}
		default:
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}
	}
	return status;
}

vx_status OvxGraphPool::addOpenGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 erode_window, i32 dilate_window)
{
	if (!graph_ptr || !vxi_src || !vxi_dst || !CPOBase::isPositive(erode_window))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (dilate_window < 0)
	{
		dilate_window = erode_window;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add nodes
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 format = OvxHelper::getFormat(vxi_src);

	switch (format)
	{
		case VX_DF_IMAGE_U8:
		{
			i32 nw = OvxHelper::getWidth(vxi_src);
			i32 nh = OvxHelper::getHeight(vxi_src);

			vx_image vxi_prev = vxi_src;
			i32 i, iter = (erode_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);
				graph_ptr->addNode(vxErode3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.erode3x3");
				vxi_prev = vxi_next;

#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}

			iter = (dilate_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = (i == iter) ? vxi_dst :
						res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);
				graph_ptr->addNode(vxDilate3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.dilate3x3");
				vxi_prev = vxi_next;

#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}
			break;
		}
		default:
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}
	}
	return status;
}

vx_status OvxGraphPool::addCloseGraph(OvxGraph* graph_ptr, vx_image vxi_src, vx_image vxi_dst, i32 dilate_window, i32 erode_window)
{
	if (!graph_ptr || !vxi_src || !vxi_dst || !CPOBase::isPositive(dilate_window))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (erode_window < 0)
	{
		erode_window = dilate_window;
	}

	vx_status status = VX_SUCCESS;
	OvxResourcePool* res_pool_ptr = graph_ptr->getResourcePool();

	//add nodes
	i32 nw = OvxHelper::getWidth(vxi_src);
	i32 nh = OvxHelper::getHeight(vxi_src);
	i32 format = OvxHelper::getFormat(vxi_src);

	switch (format)
	{
		case VX_DF_IMAGE_U8:
		{
			i32 nw = OvxHelper::getWidth(vxi_src);
			i32 nh = OvxHelper::getHeight(vxi_src);

			vx_image vxi_prev = vxi_src;
			i32 i, iter = (dilate_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);
				graph_ptr->addNode(vxDilate3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.dilate3x3");
				vxi_prev = vxi_next;

#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}

			iter = (erode_window - 1) / 2;
			for (i = 1; i <= iter; i++)
			{
				vx_image vxi_next = (i == iter) ? vxi_dst :
						res_pool_ptr->fetchImage(graph_ptr, nw, nh, VX_DF_IMAGE_U8, true, true);
				graph_ptr->addNode(vxErode3x3Node(graph_ptr->getVxGraph(), vxi_prev, vxi_next), "vx.erode3x3");
				vxi_prev = vxi_next;

#if !defined(POR_IMVS2_ON_AM5728)
				status |= OvxHelper::setNodeBorder(graph_ptr->lastNode(), VX_BORDER_REPLICATE);
#endif
			}
			break;
		}
		default:
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}
	}
	return status;
}

//////////////////////////////////////////////////////////////////////////
CGImgProcErode::CGImgProcErode(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_img_ptr = NULL;
	m_erode_window = 0;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcErode::~CGImgProcErode()
{
}

bool CGImgProcErode::prepare(va_list args)
{
	i32 erode_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);

	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || erode_window < 3)
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_img_ptr = dst_img_ptr;
	m_erode_window = erode_window;

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
		vx_image vxi_prev = m_vx_src;
		i32 i, iter = (m_erode_window - 1) / 2;
		for (i = 1; i <= iter; i++)
		{
			vx_image vxi_next = (i == iter) ? m_vx_dst :
						m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true, true);

			addNode(vxErode3x3Node(getVxGraph(), vxi_prev, vxi_next), "vx.erode3x3");
			vxi_prev = vxi_next;
#if !defined(POR_IMVS2_ON_AM5728)
			status |= OvxHelper::setNodeBorder(lastNode(), VX_BORDER_REPLICATE);
#endif
		}

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcErode preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8);
#else
	i32 padding_size = (m_erode_window - 1) / 2;
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8, padding_size);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcErode::check(va_list args)
{
	i32 erode_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);

	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || erode_window < 3)
	{
		return false;
	}
	return (erode_window == m_erode_window && m_res_pool_ptr->checkImageSize(m_vx_src, w, h));
}

bool CGImgProcErode::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::getImage(m_vx_dst, m_dst_img_ptr, m_width, m_height, 8);
#else
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
#endif
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcDilate::CGImgProcDilate(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_img_ptr = NULL;
	m_dilate_window = 0;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcDilate::~CGImgProcDilate()
{
}

bool CGImgProcDilate::prepare(va_list args)
{
	i32 dilate_window = va_arg(args, i32); 
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);

	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3)
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_img_ptr = dst_img_ptr;
	m_dilate_window = dilate_window;

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
		status |= OvxGraphPool::addDilateGraph(this, m_vx_src, m_vx_dst, dilate_window);

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcDilate preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8);
#else
	i32 padding_size = (m_dilate_window - 1) / 2;
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8, padding_size);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcDilate::check(va_list args)
{
	i32 dilate_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);

	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3)
	{
		return false;
	}
	return (dilate_window == m_dilate_window && m_res_pool_ptr->checkImageSize(m_vx_src, w, h));
}

bool CGImgProcDilate::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::getImage(m_vx_dst, m_dst_img_ptr, m_width, m_height, 8);
#else
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
#endif
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcClose::CGImgProcClose(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_img_ptr = NULL;
	m_dilate_window = 0;
	m_erode_window = 0;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcClose::~CGImgProcClose()
{
}

bool CGImgProcClose::prepare(va_list args)
{
	i32 dilate_window = va_arg(args, i32); 
	u8* src_img_ptr = va_arg(args, u8*); 
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 erode_window = va_arg(args, i32);

	erode_window = (erode_window < 0) ? dilate_window : erode_window;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3 || erode_window < 3)
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_img_ptr = dst_img_ptr;
	m_dilate_window = dilate_window;
	m_erode_window = erode_window;

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
		status |= OvxGraphPool::addCloseGraph(this, m_vx_src, m_vx_dst, dilate_window, erode_window);
		
		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcClose preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8);
#else
	i32 padding_size = (m_erode_window + m_dilate_window - 2) / 2;
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8, padding_size);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcClose::check(va_list args)
{
	i32 dilate_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 erode_window = va_arg(args, i32);

	erode_window = (erode_window < 0) ? dilate_window : erode_window;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3 || erode_window < 3)
	{
		return false;
	}

	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) &&
		dilate_window == m_dilate_window && erode_window == m_erode_window);
}

bool CGImgProcClose::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::getImage(m_vx_dst, m_dst_img_ptr, m_width, m_height, 8);
#else
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
#endif
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcOpen::CGImgProcOpen(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_img_ptr = NULL;
	m_dilate_window = 0;
	m_erode_window = 0;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcOpen::~CGImgProcOpen()
{
}

bool CGImgProcOpen::prepare(va_list args)
{
	i32 erode_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 dilate_window = va_arg(args, i32);

	dilate_window = (dilate_window < 0) ? erode_window : dilate_window;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3 || erode_window < 3)
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_img_ptr = dst_img_ptr;
	m_dilate_window = dilate_window;
	m_erode_window = erode_window;

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
		status |= OvxGraphPool::addOpenGraph(this, m_vx_src, m_vx_dst, erode_window, dilate_window);

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcClose preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8);
#else
	i32 padding_size = (m_erode_window + m_dilate_window - 2) / 2;
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8, padding_size);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcOpen::check(va_list args)
{
	i32 erode_window = va_arg(args, i32);
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 dilate_window = va_arg(args, i32);

	dilate_window = (dilate_window < 0) ? erode_window : dilate_window;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || dilate_window < 3 || erode_window < 3)
	{
		return false;
	}

	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) && 
		dilate_window == m_dilate_window && erode_window == m_erode_window);
}

bool CGImgProcOpen::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::getImage(m_vx_dst, m_dst_img_ptr, m_width, m_height, 8);
#else
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
#endif
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcRemap::CGImgProcRemap(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_channel = 0;
	m_dst_img_ptr = NULL;
	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcRemap::~CGImgProcRemap()
{
}

bool CGImgProcRemap::prepare(va_list args)
{
	u8* src_img_ptr = va_arg(args, u8*); 
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32); 
	i32 h = va_arg(args, i32); 
	i32 channel = va_arg(args, i32); 
	vx_remap dist_remap = va_arg(args, vx_remap);

	if (!src_img_ptr || w*h*channel <= 0 || !dst_img_ptr || !dist_remap)
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_img_ptr = dst_img_ptr;

	if (!checkVerified())
	{
		//add nodes
		switch (channel)
		{
			case 1:
			{
				m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true);
				m_vx_dst = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true);

				addNode(vxRemapNode(getVxGraph(), m_vx_src, dist_remap, VX_INTERPOLATION_BILINEAR,
							m_vx_dst), "vx.remap.channel");
				break;
			}
			case 3:
			{
				m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_RGB, true);
				m_vx_dst = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_RGB, true);

				vx_image vxi_r_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);
				vx_image vxi_g_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);
				vx_image vxi_b_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);
				vx_image vxi_wr_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);
				vx_image vxi_wg_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);
				vx_image vxi_wb_channel = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8, true, true);

				addNode(vxChannelExtractNode(getVxGraph(), m_vx_src, VX_CHANNEL_R, vxi_r_channel), "vx.r_channel");
				addNode(vxChannelExtractNode(getVxGraph(), m_vx_src, VX_CHANNEL_G, vxi_g_channel), "vx.g_channel");
				addNode(vxChannelExtractNode(getVxGraph(), m_vx_src, VX_CHANNEL_B, vxi_b_channel), "vx.b_channel");
				addNode(vxRemapNode(getVxGraph(), vxi_r_channel, dist_remap, VX_INTERPOLATION_BILINEAR,
							vxi_wr_channel), "vx.remap.r.channel");
 				addNode(vxRemapNode(getVxGraph(), vxi_g_channel, dist_remap, VX_INTERPOLATION_BILINEAR,
 							vxi_wg_channel), "vx.remap.g.channel");
 				addNode(vxRemapNode(getVxGraph(), vxi_b_channel, dist_remap, VX_INTERPOLATION_BILINEAR,
							vxi_wb_channel), "vx.remap.b.channel");
				addNode(vxChannelCombineNode(getVxGraph(), vxi_wr_channel, vxi_wg_channel, vxi_wb_channel,
							NULL, m_vx_dst), "vx.channel.combine");
				break;
			}
			default:
			{
				return false;
			}
		}

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcRemap preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
		m_channel = channel;
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8*channel);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8*channel);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcRemap::check(va_list args)
{
	u8* src_img_ptr = va_arg(args, u8*);
	u8* dst_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	i32 channel = va_arg(args, i32);
	vx_remap dist_remap = va_arg(args, vx_remap);

	if (!src_img_ptr || w*h*channel <= 0 || !dst_img_ptr || !dist_remap)
	{
		return false;
	}
	return (m_channel == channel && m_res_pool_ptr->checkImageSize(m_vx_src, m_width, m_height));
}

bool CGImgProcRemap::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::getImage(m_vx_dst, m_dst_img_ptr, m_width, m_height, 8*m_channel);
#else
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
#endif
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtImg2RunTable::CGImgProcCvtImg2RunTable(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_run_table = NULL;

	m_vx_src = NULL;
	m_vx_src_mask = NULL;
	m_vx_param = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtImg2RunTable::~CGImgProcCvtImg2RunTable()
{
}

bool CGImgProcCvtImg2RunTable::prepare(va_list args)
{
	u8* src_img_ptr = va_arg(args, u8*); 
	i32 w = va_arg(args, i32); 
	i32 h = va_arg(args, i32); 
	void* dst_run_table = va_arg(args, void*);
	i32 flag = va_arg(args, i32); 
	i32 mask_val = va_arg(args, i32); 
	u8* src_mask_ptr = va_arg(args, u8*);

	bool use_src_mask = CPOBase::bitCheck(flag, kRunTableFlagMaskImg);
	if (!src_img_ptr || w*h <= 0 || !dst_run_table || (use_src_mask && !src_mask_ptr))
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = w;
	m_height = h;
	m_dst_run_table = dst_run_table;

	vxParamConvertImg2RunTable param;
	param.width = w;
	param.height = h;
	param.flag = flag;
	param.mask_val = mask_val;

	CImgRunTable* run_table_ptr = (CImgRunTable*)dst_run_table;
	i32 run_table_size = run_table_ptr->getMaxArraySize();

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);
		m_vx_src_mask = m_res_pool_ptr->fetchImage(this, w, h, VX_DF_IMAGE_U8);
		m_vx_param = m_res_pool_ptr->fetchArray(this, VX_TYPE_UINT8, sizeof(vxParamConvertImg2RunTable));
		m_vx_dst = m_res_pool_ptr->fetchArray(this, VX_TYPE_UINT8, run_table_size);

		//add nodes
#if defined(POVX_USE_IMG_TO_RUNTABLE)
		addNode(OvxCustomNode::vxConvertImgToRunTableNode(getVxGraph(), m_vx_src, m_vx_src_mask,
												m_vx_param, m_vx_dst), POVX_KERNEL_NAME_IMG_TO_RUNTABLE);
#endif

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcConvertImg2RunTable preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
	status |= OvxHelper::writeArray(m_vx_param, &param, sizeof(param), 1);

#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr, w, h, 8);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr, w, h, 8);
#endif
	if (use_src_mask)
	{
#if defined(POR_IMVS2_ON_AM5728)
		status |= OvxHelper::setImage(m_vx_src_mask, src_mask_ptr, w, h, 8);
#else
		status |= OvxHelper::writeImage(m_vx_src_mask, src_mask_ptr, w, h, 8);
#endif
	}
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtImg2RunTable::check(va_list args)
{
	u8* src_img_ptr = va_arg(args, u8*);
	i32 w = va_arg(args, i32);
	i32 h = va_arg(args, i32);
	void* dst_run_table = va_arg(args, void*);
	i32 flag = va_arg(args, i32);
	i32 mask_val = va_arg(args, i32);
	u8* src_mask_ptr = va_arg(args, u8*);

	bool use_src_mask = CPOBase::bitCheck(flag, kRunTableFlagMaskImg);
	if (!src_img_ptr || w*h <= 0 || !dst_run_table || (use_src_mask && !src_mask_ptr))
	{
		return false;
	}
	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h));
}

bool CGImgProcCvtImg2RunTable::finished()
{
	if (!m_dst_run_table || m_width <= 0 || m_height <= 0)
	{
		return false;
	}

	i32 pos = 0;
	vx_status status = VX_SUCCESS;
	CImgRunTable* dst_run_table_ptr = (CImgRunTable*)m_dst_run_table;

	status |= OvxHelper::readArray(&dst_run_table_ptr->m_pixels, pos, sizeof(i32), 1, m_vx_dst); pos += sizeof(i32);
	status |= OvxHelper::readArray(&dst_run_table_ptr->m_run_count, pos, sizeof(i32), 1, m_vx_dst); pos += sizeof(i32);
	status |= OvxHelper::readArray(dst_run_table_ptr->m_pxy_ptr, pos, (m_height + 1)*sizeof(i32), 1, m_vx_dst); pos += (m_height + 1)*sizeof(i32);
	status |= OvxHelper::readArray(dst_run_table_ptr->m_run2_ptr, pos, (dst_run_table_ptr->m_run_count)*sizeof(u16), 1, m_vx_dst);
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtRunTable2Img::CGImgProcCvtRunTable2Img(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_dst_img_ptr = NULL;

	m_vx_src = NULL;
	m_vx_param = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtRunTable2Img::~CGImgProcCvtRunTable2Img()
{
}

bool CGImgProcCvtRunTable2Img::prepare(va_list args)
{
	void* src_run_table = va_arg(args, void*); 
	u8* dst_img_ptr = va_arg(args, u8*); 
	u8 output_val = va_arg(args, u8);
	if (!src_run_table || !dst_img_ptr)
	{
		return false;
	}

	CImgRunTable* run_table_ptr = (CImgRunTable*)src_run_table;
	i32 rt_max_size = run_table_ptr->getMaxArraySize();

	//update param
	vx_status status = VX_SUCCESS;
	m_width = run_table_ptr->m_width;
	m_height = run_table_ptr->m_height;
	m_dst_img_ptr = dst_img_ptr;

	vxParamConvertRunTable2Img param;
	param.width = m_width;
	param.height = m_height;
	param.value = output_val;

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchArray(this, VX_TYPE_UINT8, rt_max_size);
		m_vx_param = m_res_pool_ptr->fetchArray(this, VX_TYPE_UINT8, sizeof(vxParamConvertRunTable2Img));
		m_vx_dst = m_res_pool_ptr->fetchImage(this, m_width, m_height, VX_DF_IMAGE_U8);

		//add nodes
#if defined(POVX_USE_RUNTABLE_TO_IMG)
		addNode(OvxCustomNode::vxConvertRunTableToImgNode(getVxGraph(), m_vx_src, m_vx_param, m_vx_dst),
									POVX_KERNEL_NAME_RUNTABLE_TO_IMG);
#endif

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcConvertRunTable2Img preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
	status |= OvxHelper::writeArray(m_vx_src, &run_table_ptr->m_pixels, sizeof(i32), 1);
	status |= OvxHelper::appendArray(m_vx_src, &run_table_ptr->m_run_count, sizeof(i32), 1);
	status |= OvxHelper::appendArray(m_vx_src, run_table_ptr->m_pxy_ptr, sizeof(i32)*(m_height + 1), 1);
	status |= OvxHelper::appendArray(m_vx_src, run_table_ptr->m_run2_ptr, sizeof(u16)*run_table_ptr->m_run_count, 1);
	status |= OvxHelper::writeArray(m_vx_param, &param, sizeof(param), 1);
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtRunTable2Img::check(va_list args)
{
	void* src_run_table = va_arg(args, void*);
	u8* dst_img_ptr = va_arg(args, u8*);
	u8 output_val = va_arg(args, u8);
	if (!src_run_table || !dst_img_ptr)
	{
		return false;
	}

	CImgRunTable* run_table_ptr = (CImgRunTable*)src_run_table;
	return (OvxHelper::getCapacity(m_vx_src) >= run_table_ptr->getArraySize() && 
		m_res_pool_ptr->checkImageSize(m_vx_dst, run_table_ptr->m_width, run_table_ptr->m_height));
}

bool CGImgProcCvtRunTable2Img::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr || m_width * m_height <= 0)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
	status |= OvxHelper::readImage(m_dst_img_ptr, m_width, m_height, m_vx_dst);
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtColor::CGImgProcCvtColor(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_src_format = VX_DF_IMAGE_VIRT;
	m_dst_format = VX_DF_IMAGE_VIRT;
	m_dst_channel = 0;
	m_dst_img_ptr = NULL;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtColor::~CGImgProcCvtColor()
{
}

bool CGImgProcCvtColor::prepare(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*);
	i32 output_format = va_arg(args, i32);

	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = src_img_ptr->w;
	m_height = src_img_ptr->h;
	m_dst_img_ptr = dst_img_ptr;

	switch (src_img_ptr->channel)
	{
		case kPOGrayChannels:	m_src_format = VX_DF_IMAGE_U8; break;	//GrayColor
		case kPORGBChannels:	m_src_format = VX_DF_IMAGE_RGB; break;	//RGBColor
		default: return false;
	}

	switch (output_format)
	{
		case kPOColorCvt2Gray:	m_dst_format = VX_DF_IMAGE_U8; m_dst_channel = 1; break;
		case kPOColorCvt2RGB:	m_dst_format = VX_DF_IMAGE_RGB; m_dst_channel = 3; break;
		case kPOColorCvt2YUV:	m_dst_format = VX_DF_IMAGE_YUV4; m_dst_channel = 3; break;
		default: return false;
	}

	if (!checkVerified())
	{
		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, m_width, m_height, m_src_format);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, m_dst_format, true);

		//add nodes
		addNode(vxColorConvertNode(getVxGraph(), m_vx_src, m_vx_dst), "vx.cvtColor");

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcCvtColor preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtColor::check(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*);
	i32 output_format = va_arg(args, i32);

	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	i32 w = src_img_ptr->w;
	i32 h = src_img_ptr->h;
	i32 src_format, dst_format;
	switch (src_img_ptr->channel)
	{
		case kPOGrayChannels:	src_format = VX_DF_IMAGE_U8; break;	//GrayColor
		case kPORGBChannels:	src_format = VX_DF_IMAGE_RGB; break;//RGBColor
		default: return false;
	}
	switch (output_format)
	{
		case kPOColorCvt2Gray:	dst_format = VX_DF_IMAGE_U8; break;
		case kPOColorCvt2RGB:	dst_format = VX_DF_IMAGE_RGB; break;
		case kPOColorCvt2YUV:	dst_format = VX_DF_IMAGE_YUV4; break;
		default: return false;
	}

	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) &&
		OvxHelper::getFormat(m_vx_src) == src_format && OvxHelper::getFormat(m_vx_dst) == dst_format);
}

bool CGImgProcCvtColor::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr || m_width * m_height <= 0)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
	ImageData* img_data_ptr = (ImageData*)m_dst_img_ptr;
	if (img_data_ptr)
	{
		img_data_ptr->initInternalBuffer(m_width, m_height, m_dst_channel);
		status |= OvxHelper::readImage(img_data_ptr->img_ptr, m_width, m_height, m_vx_dst);
	}
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtSplit::CGImgProcCvtSplit(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_src_format = VX_DF_IMAGE_VIRT;
	m_dst_channel = VX_CHANNEL_0;
	m_dst_img_ptr = NULL;
	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtSplit::~CGImgProcCvtSplit()
{
}

bool CGImgProcCvtSplit::prepare(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*); 
	ImageData* dst_img_ptr = va_arg(args, ImageData*); 
	i32 output_format = va_arg(args, i32);

	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = src_img_ptr->w;
	m_height = src_img_ptr->h;
	m_dst_img_ptr = dst_img_ptr;

	if (!checkVerified())
	{
		switch (src_img_ptr->channel)
		{
			case kPORGBChannels:	m_src_format = VX_DF_IMAGE_RGB; break;	//RGBColor
			default: return false;
		}
		switch (output_format)
		{
			case kPOColorCvt2Red:	m_dst_channel = VX_CHANNEL_R; break;
			case kPOColorCvt2Green:	m_dst_channel = VX_CHANNEL_G; break;
			case kPOColorCvt2Blue:	m_dst_channel = VX_CHANNEL_B; break;
			default:				return false;
		}

		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, m_width, m_height, m_src_format);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
		addNode(vxChannelExtractNode(getVxGraph(), m_vx_src, m_dst_channel, m_vx_dst), "vx.cvtSplit");

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcCvtSplit preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtSplit::check(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*);
	i32 output_format = va_arg(args, i32);

	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	i32 w = src_img_ptr->w;
	i32 h = src_img_ptr->h;
	i32 src_format, dst_channel;

	switch (src_img_ptr->channel)
	{
		case kPORGBChannels:	src_format = VX_DF_IMAGE_RGB; break;	//RGBColor
		default: return false;
	}
	switch (output_format)
	{
		case kPOColorCvt2Red:	dst_channel = VX_CHANNEL_R; break;
		case kPOColorCvt2Green:	dst_channel = VX_CHANNEL_G; break;
		case kPOColorCvt2Blue:	dst_channel = VX_CHANNEL_B; break;
		default:				return false;
	}

	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) && 
		OvxHelper::getFormat(m_vx_src) == src_format && m_dst_channel == dst_channel);
}

bool CGImgProcCvtSplit::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr || m_width * m_height <= 0)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
	ImageData* img_data_ptr = (ImageData*)m_dst_img_ptr;
	if (img_data_ptr)
	{
		img_data_ptr->initInternalBuffer(m_width, m_height);
		status |= OvxHelper::readImage(img_data_ptr->img_ptr, m_width, m_height, m_vx_dst);
	}
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtHSVSplit::CGImgProcCvtHSVSplit(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_src_format = VX_DF_IMAGE_VIRT;
	m_dst_channel = VX_CHANNEL_0;
	m_dst_img_ptr = NULL;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtHSVSplit::~CGImgProcCvtHSVSplit()
{
}

bool CGImgProcCvtHSVSplit::prepare(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*); 
	i32 output_format = va_arg(args, i32);
	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount)) 
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = src_img_ptr->w;
	m_height = src_img_ptr->h;
	m_dst_img_ptr = dst_img_ptr;

	if (!checkVerified())
	{
		switch (src_img_ptr->channel)
		{
			case kPORGBChannels:			m_src_format = VX_DF_IMAGE_RGB; break;	//RGBColor
			default:						return false;
		}
		switch (output_format)
		{
			case kPOColorCvt2Hue:			m_dst_channel = VX_CHANNEL_R; break;
			case kPOColorCvt2Saturation:	m_dst_channel = VX_CHANNEL_G; break;
			default:						return false;
		}

		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, m_width, m_height, m_src_format);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);
		vx_image vxi_hsv = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_RGB, true, true);

		//add nodes
#if defined(POVX_USE_CONVERT_TO_HSV)
		addNode(OvxCustomNode::vxConvertToHSVNode(getVxGraph(), m_vx_src, vxi_hsv), POVX_KERNEL_NAME_CONVERT_TO_HSV);
		addNode(vxChannelExtractNode(getVxGraph(), vxi_hsv, m_dst_channel, m_vx_dst), "vx.cvtHSVSplit");
#endif

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcCvtHSVSplit preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtHSVSplit::check(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*);
	i32 output_format = va_arg(args, i32);
	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	i32 w = src_img_ptr->w;
	i32 h = src_img_ptr->h;
	i32 src_format, dst_channel;
	switch (src_img_ptr->channel)
	{
		case kPORGBChannels:			src_format = VX_DF_IMAGE_RGB; break;	//RGBColor
		default:						return false;
	}
	switch (output_format)
	{
		case kPOColorCvt2Hue:			dst_channel = VX_CHANNEL_R; break;
		case kPOColorCvt2Saturation:	dst_channel = VX_CHANNEL_G; break;
		default:						return false;
	}
	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) && 
		OvxHelper::getFormat(m_vx_src) == src_format && m_dst_channel == dst_channel);
}

bool CGImgProcCvtHSVSplit::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr || m_width * m_height <= 0)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
	ImageData* img_data_ptr = (ImageData*)m_dst_img_ptr;
	if (img_data_ptr)
	{
		img_data_ptr->initInternalBuffer(m_width, m_height);
		status |= OvxHelper::readImage(img_data_ptr->img_ptr, m_width, m_height, m_vx_dst);
	}
	return (status == VX_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////
CGImgProcCvtIntensity::CGImgProcCvtIntensity(OvxResourcePool* resource_pool_ptr, i32 graph_type)
	: OvxGraph(resource_pool_ptr, graph_type)
{
	m_width = 0;
	m_height = 0;
	m_src_format = VX_DF_IMAGE_VIRT;
	m_dst_img_ptr = NULL;

	m_vx_src = NULL;
	m_vx_dst = NULL;
}

CGImgProcCvtIntensity::~CGImgProcCvtIntensity()
{
}

bool CGImgProcCvtIntensity::prepare(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*); 
	ImageData* dst_img_ptr = va_arg(args, ImageData*); 
	i32 output_format = va_arg(args, i32);
	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	//update param
	vx_status status = VX_SUCCESS;
	m_width = src_img_ptr->w;
	m_height = src_img_ptr->h;
	m_dst_img_ptr = dst_img_ptr;

	if (!checkVerified())
	{
		switch (src_img_ptr->channel)
		{
			case kPO2Channels:		m_src_format = VX_DF_IMAGE_NV12; break; //NV12
			case kPORGBChannels:	m_src_format = VX_DF_IMAGE_RGB; break;	//RGB
			case kPORGBXChannels:	m_src_format = VX_DF_IMAGE_RGBX; break;	//RGBA
			default:				return false;
		}

		//add resources
		m_vx_src = m_res_pool_ptr->fetchImage(this, m_width, m_height, m_src_format);

		i32 nw = OvxHelper::getWidth(m_vx_src);
		i32 nh = OvxHelper::getHeight(m_vx_src);
		m_vx_dst = m_res_pool_ptr->fetchImage(this, nw, nh, VX_DF_IMAGE_U8, true);

		//add nodes
#if defined(POVX_USE_CONVERT_TO_AVG)
		addNode(OvxCustomNode::vxConvertToAvgNode(getVxGraph(), m_vx_src, m_vx_dst), POVX_KERNEL_NAME_CONVERT_TO_AVG);
#endif

		//graph verify
		if (!verify())
		{
			printlog_lvs2("CGImgProcCvtIntensity preparing failed.", LOG_SCOPE_OVX);
			return false;
		}
	}

	//update vx param
#if defined(POR_IMVS2_ON_AM5728)
	status |= OvxHelper::setImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#else
	status |= OvxHelper::writeImage(m_vx_src, src_img_ptr->img_ptr, m_width, m_height, src_img_ptr->channel * 8);
#endif
	return (status == VX_SUCCESS);
}

bool CGImgProcCvtIntensity::check(va_list args)
{
	ImageData* src_img_ptr = va_arg(args, ImageData*);
	ImageData* dst_img_ptr = va_arg(args, ImageData*);
	i32 output_format = va_arg(args, i32);
	if (!src_img_ptr || !dst_img_ptr || !src_img_ptr->isValid() ||
		!CPOBase::checkIndex(output_format, kPOColorCvtTypeCount))
	{
		return false;
	}

	i32 w = src_img_ptr->w;
	i32 h = src_img_ptr->h;
	i32 src_format;
	switch (src_img_ptr->channel)
	{
		case kPO2Channels:		src_format = VX_DF_IMAGE_NV12; break;	//NV12
		case kPORGBChannels:	src_format = VX_DF_IMAGE_RGB; break;	//RGB
		case kPORGBXChannels:	src_format = VX_DF_IMAGE_RGBX; break;	//RGBA
		default:				return false;
	}

	return (m_res_pool_ptr->checkImageSize(m_vx_src, w, h) && OvxHelper::getFormat(m_vx_src) == src_format);
}

bool CGImgProcCvtIntensity::finished()
{
	if (!m_vx_dst || !m_dst_img_ptr || m_width * m_height <= 0)
	{
		return false;
	}

	vx_status status = VX_SUCCESS;
	ImageData* img_data_ptr = (ImageData*)m_dst_img_ptr;
	if (img_data_ptr)
	{
		img_data_ptr->initInternalBuffer(m_width, m_height);
		status |= OvxHelper::readImage(img_data_ptr->img_ptr, m_width, m_height, m_vx_dst);
	}
	return (status == VX_SUCCESS);
}
#endif

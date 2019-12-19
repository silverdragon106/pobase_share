#include "define.h"
#include "lock_guide.h"
#include "logger/logger.h"
#include "base.h"
#include "proc/image_proc.h"
#include "ti_c66_dsp_app.h"

#if defined(POR_WITH_DSP)
CDspNode::CDspNode()
{
	m_id = -1;
	m_cmd = DSP_CMD_NONE;
	m_status = kDspNodeNone;

	m_dsp_target = kDspTargetAny;
	m_resource_count = 1;
	m_semaphore_ptr = new QSemaphore();
	m_dsp_app_ptr = NULL;

	m_src_buffer_ptr = NULL;
	m_dst_buffer_ptr = NULL;
	m_width = 0;
	m_height = 0;
	m_src_step = 0;
	m_dst_step = 0;
	m_gap_size = 1;
}

CDspNode::CDspNode(i32 id, u16 cmd, u8 dsp_target, CBaseDspApp* dsp_app_ptr)
{
	//calc resource count for this node
	i32 resource_count = 1;
	switch (dsp_target)
	{
		case kDspTargetDsp1:
		case kDspTargetDsp2:
		case kDspTargetAny:
		{
			resource_count = 1;
			break;
		}
		case kDspTargetMultiCore:
		default:
		{
			resource_count = DSP_MULTI_CORES;
			break;
		}
	}

	//init dsp node
	m_id = id;
	m_cmd = cmd;
	m_status = kDspNodeNone;

	m_dsp_target = dsp_target;
	m_resource_count = resource_count;
	m_semaphore_ptr = new QSemaphore(resource_count);
	m_dsp_app_ptr = dsp_app_ptr;

	m_src_buffer_ptr = NULL;
	m_dst_buffer_ptr = NULL;
	m_width = 0;
	m_height = 0;
	m_src_step = 0;
	m_dst_step = 0;
	m_gap_size = 1;
}

CDspNode::~CDspNode()
{
	//wait until finished
	m_semaphore_ptr->acquire(m_resource_count);
	POSAFE_DELETE(m_semaphore_ptr);
	setStatus(kDspNodeFinished);
}

void CDspNode::prepare(u8* src_img_ptr, u8* dst_img_ptr, i32 w, i32 h, i32 src_step, i32 dst_step, i32 gap_size)
{
	m_src_buffer_ptr = src_img_ptr;
	m_dst_buffer_ptr = dst_img_ptr;
	m_width = w;
	m_height = h;
	m_src_step = src_step;
	m_dst_step = dst_step;
	m_gap_size = gap_size;
}

i32 CDspNode::process()
{
	schedule();
	if (isFinished())
	{
		return m_status;
	}
	return wait();
}

i32 CDspNode::schedule()
{
	if (m_status == kDspNodeNone || !m_dsp_app_ptr || !m_dsp_app_ptr->addDspNode2Queue(this))
	{
		m_status = kDspNodeInvalid;
	}

	return m_status;
}

i32 CDspNode::wait()
{
	if (isFinished())
	{
		printlog_lv4(QString("Node[%1] finished").arg(m_id));
		return m_status;
	}

	//wait until finished
	m_semaphore_ptr->acquire(m_resource_count);
	m_semaphore_ptr->release(m_resource_count);
	setStatus(kDspNodeFinished);

	printlog_lv4(QString("Node[%1] wait & finished").arg(m_id));
	return kDspNodeFinished;
}

void CDspNode::setStatus(i32 status)
{
	m_status = status;
}

void CDspNode::acquireResource(i32 count)
{
	if (m_semaphore_ptr)
	{
		count = po::_min(count, m_resource_count);
		m_semaphore_ptr->acquire(count);
	}
}

void CDspNode::releaseResource(i32 count)
{
	if (m_semaphore_ptr)
	{
		count = po::_min(count, m_resource_count);
		m_semaphore_ptr->release(count);
	}
}

//////////////////////////////////////////////////////////////////////////
CDspCoreThread::CDspCoreThread()
{
	m_is_inited = false;
	m_is_thread_cancel = false;

	exlock_guard(m_queue_mutex);
	m_queue.clear();
}

CDspCoreThread::~CDspCoreThread()
{
	exitDspCoreThread();
	freeBuffer();
}

bool CDspCoreThread::initDspCoreThread(u16 dsp_core_id)
{
	if (!m_is_inited)
	{
		singlelog_lv1("DspCoreThread InitInstance");

		memset(&m_ipc_client, 0, sizeof(m_ipc_client));
		m_ipc_client.target_id = dsp_core_id;

		if (ipc_start(&m_ipc_client) != IPC_SUCCESS ||
			ipc_send_init_msg(&m_ipc_client) != IPC_SUCCESS ||
			ipc_wait_msg(&m_ipc_client, DSP_WAIT_FOREVER, NULL) != IPC_SUCCESS) //TIMEOUT_INIT_CMD
		{
			exitDspCoreThread();
			return false;
		}

		freeBuffer();

		m_is_thread_cancel = false;
		m_is_inited = true;
		QThreadStart();
	}
	return true;
}

bool CDspCoreThread::exitDspCoreThread()
{
	if (m_is_inited)
	{
		singlelog_lv1("DspCoreThread ExitInstance");
		m_is_thread_cancel = true;
		QThreadStop();
				
		freeBuffer();
		ipc_end(&m_ipc_client);
		m_is_inited = false;
	}

	return true;
}

void CDspCoreThread::freeBuffer()
{
	exlock_guard(m_queue_mutex);
	for (DspQueueList::iterator iter = m_queue.begin(); iter != m_queue.end(); ++iter)
	{
		if (iter->is_allocated)
		{
			POSAFE_DELETE_ARRAY(iter->src_buffer_ptr);
		}
	}
	m_queue.clear();
}

bool CDspCoreThread::isValidDspItem(DspQueueItem& item)
{
	if (!item.src_buffer_ptr || !item.dst_buffer_ptr)
	{
		return false;
	}
	if (!CPOBase::isPositive(item.src_buffer_size) || !CPOBase::isPositive(item.dst_buffer_size))
	{
		return false;
	}
	return true;
}

void CDspCoreThread::removeDspQueue(i32 id)
{
	exlock_guard(m_queue_mutex);
	for (DspQueueList::iterator iter = m_queue.begin(); iter != m_queue.end(); ++iter)
	{
		if (iter->node_id == id)
		{
			if (iter->is_allocated)
			{
				POSAFE_DELETE_ARRAY(iter->src_buffer_ptr);
			}
			iter = m_queue.erase(iter);
			break;
		}
	}
}

i32 CDspCoreThread::getFreeQueueSize()
{
	exlock_guard(m_queue_mutex);
	return DSP_NODE_QUEUE_SIZE - (i32)m_queue.size();
}

void CDspCoreThread::run()
{
	singlelog_lv1("DspCoreThread Loop");

	bool has_process;
	DspQueueItem dsp_item;

	while (!m_is_thread_cancel)
	{
		has_process = false;
		{
			exlock_guard(m_queue_mutex);
			if (m_queue.size() > 0)
			{
				has_process = true;
				dsp_item = m_queue.front();
				m_queue.pop_front();
			}
		}

		if (has_process)
		{
			m_ipc_client.item = dsp_item;
			i32 ret_code = ipc_send_cmd_msg(&m_ipc_client);

			printlog_lv3(QString("DSP[%1] send data: nid=%2,%3").arg(m_ipc_client.target_id).arg(dsp_item.node_id).arg(ret_code));
			switch (ret_code)
			{
				case IPC_SUCCESS:
				{
					break;
				}
				case IPC_QUEUE_FULL:
				{
					checkDspResult(DSP_WAIT_FOREVER);
					// retry send
					i32 ret_code = ipc_send_cmd_msg(&m_ipc_client);
					printlog_lv3(QString("DSP[%1] send data: nid=%2,%3").arg(m_ipc_client.target_id).arg(dsp_item.node_id).arg(ret_code));
					break;
				}
				default:
				{
					QThread::msleep(1);
					continue;
				}
			}
		}
		checkDspResult(1);
	}
}

void CDspCoreThread::checkDspResult(i32 timeout_ms)
{
	IpcClientInfo ipc_item;
	u8* ipc_buffer_ptr = NULL;

	ipc_item.target_id = m_ipc_client.target_id;

	if (ipc_wait_msg(&ipc_item, timeout_ms, &ipc_buffer_ptr) == IPC_SUCCESS)
	{
		DspQueueItem* dsp_item_ptr = &ipc_item.item;
		printlog_lv3(QString("DSP[%1] loading %2 percent").arg(m_ipc_client.target_id).arg(ipc_item.dsp_load));

		//restore backup
		i32 w = dsp_item_ptr->content.width;
		i32 dst_height = dsp_item_ptr->content.dst_height;
		i32 src_step =  dsp_item_ptr->content.src_step;
		i32 dst_step =  dsp_item_ptr->content.dst_step;
		i32 dy_pos = dsp_item_ptr->content.dy_pos;
		memcpy(dsp_item_ptr->dst_buffer_ptr, ipc_buffer_ptr + dy_pos*w*dst_step, dst_height*w*dst_step);

		//release
		if (dsp_item_ptr->is_allocated)
		{
			POSAFE_DELETE_ARRAY(dsp_item_ptr->src_buffer_ptr);
		}
		if (dsp_item_ptr->semaphore_ptr)
		{
			((QSemaphore*)(dsp_item_ptr->semaphore_ptr))->release();
		}
	}
}

bool CDspCoreThread::addDspNode2Queue(CDspNode* dsp_node_ptr, i32 index, i32 count)
{
	if (!dsp_node_ptr || count <= 0)
	{
		return false;
	}

	{
		exlock_guard(m_queue_mutex);

		//check queue size
		i32 free_item = DSP_NODE_QUEUE_SIZE - (i32)m_queue.size();
		if (free_item <= 0)
		{
			return false;
		}

		//add queue item
		DspQueueItem dsp_item;
		memset(&dsp_item, 0, sizeof(DspQueueItem));

		dsp_item.cmd = dsp_node_ptr->getCmd();
		dsp_item.node_id = dsp_node_ptr->getNodeID();
		dsp_item.semaphore_ptr = (void*)dsp_node_ptr->m_semaphore_ptr;

		//check image range
		i32 w = dsp_node_ptr->m_width;
		i32 h = dsp_node_ptr->m_height;
		i32 src_step = dsp_node_ptr->m_src_step;
		i32 dst_step = dsp_node_ptr->m_dst_step;
		i32 gap_size = dsp_node_ptr->m_gap_size;
		u8* src_buffer_ptr = dsp_node_ptr->m_src_buffer_ptr;
		u8* dst_buffer_ptr = dsp_node_ptr->m_dst_buffer_ptr;

		i32 py0 = h * index / count;
		i32 py1 = h * (index + 1)/count;
		i32 py2 = po::_max(py0 - gap_size, 0); // top - gap
		i32 py3 = po::_min(py1 + gap_size, h); // bottom + gap
		i32 dh = py3 - py2; // height of extracted rect
		i32 dy = py0 - py2; // height of top gap

		//check memory allocation
		bool new_src_allocated = false;
		if (count > 1 && src_buffer_ptr == dst_buffer_ptr)
		{
			new_src_allocated = true;
		}

		u8* new_src_buffer_ptr = src_buffer_ptr + py2*w*src_step;
		u8* new_dst_buffer_ptr = dst_buffer_ptr + py0*w*dst_step;
		if (new_src_allocated)
		{
			new_src_buffer_ptr = new u8[dh*w*src_step];
			memcpy(new_src_buffer_ptr, src_buffer_ptr + py2*w*src_step, dh*w*src_step);
		}

		dsp_item.is_allocated = new_src_allocated;
		dsp_item.src_buffer_ptr = new_src_buffer_ptr;
		dsp_item.dst_buffer_ptr = new_dst_buffer_ptr;
		dsp_item.src_buffer_size = w*dh*src_step;
		dsp_item.dst_buffer_size = w*dh*dst_step;
		dsp_item.content.width = w;					//width
		dsp_item.content.src_height = dh;			//src_height: equal or lager than image-height / core-count
		dsp_item.content.dst_height = py1 - py0;	//dst_height: required height
		dsp_item.content.src_step = src_step;		//src_step
		dsp_item.content.dst_step = dst_step;		//dst_step
		dsp_item.content.dy_pos = dy;				//dy_pos: difference between src-rect and dst-rect at top

		if (!isValidDspItem(dsp_item))
		{
			return false;
		}
		m_queue.push_back(dsp_item);
	}

	dsp_node_ptr->setStatus(kDspNodeRunning);
	printlog_lv3(QString("DSP[%1]: add item[%2], qsize=%3").arg(dsp_node_ptr->getTarget() + 1)
					.arg(dsp_node_ptr->getNodeID()).arg((i32)m_queue.size()));
	return true;
}

//////////////////////////////////////////////////////////////////////////
CBaseDspApp::CBaseDspApp()
{
	m_node_id = 0;
	m_is_inited = false;
}

CBaseDspApp::~CBaseDspApp()
{
	exitDspApp();
}

bool CBaseDspApp::initDspApp()
{
	if (!m_is_inited)
	{
		singlelog_lv1("BaseDspApp InitInstance");

		if (ipc_init(1) != IPC_SUCCESS)
		{
			return false;
		}

		for (i32 i = 0; i < DSP_MULTI_CORES; i++)
		{
			if (!m_dsp_thread[i].initDspCoreThread(i + 1))
			{
				printlog_lv1(QString("DspThread%1 Initialize Failed.").arg(i));
				return false;
			}
		}
		m_node_id = 0;
		m_is_inited = true;
	}

	return true;
}

bool CBaseDspApp::exitDspApp()
{
	if (m_is_inited)
	{
		singlelog_lv1("BaseDspApp ExitInstance");
		for (i32 i = 0; i < DSP_MULTI_CORES; i++)
		{
			if (!m_dsp_thread[i].exitDspCoreThread())
			{
				return false;
			}
		}
		m_node_id = 0;
		m_is_inited = false;

		ipc_uninit();
	}

	return true;
}

CDspNode* CBaseDspApp::createDspNode(u16 cmd, u8 dsp_target)
{
	CDspNode* node_ptr = new CDspNode(m_node_id++, cmd, dsp_target, this);
	node_ptr->m_status = kDspNodeNotStarted;
	return node_ptr;
}

void CBaseDspApp::removeDspNode(i32 id)
{
	for (i32 i = 0; i < DSP_MULTI_CORES; i++)
	{
		m_dsp_thread[i].removeDspQueue(id);
	}
}

void CBaseDspApp::clearDspNodes()
{
	for (i32 i = 0; i < DSP_MULTI_CORES; i++)
	{
		m_dsp_thread[i].freeBuffer();
	}
}

bool CBaseDspApp::addDspNode2Queue(CDspNode* dsp_node_ptr)
{
	if (!dsp_node_ptr)
	{
		return false;
	}

	switch (dsp_node_ptr->getTarget())
	{
		case kDspTargetDsp1:
		{
			if (!m_dsp_thread[0].addDspNode2Queue(dsp_node_ptr))
			{
				dsp_node_ptr->setStatus(kDspNodeInvalid);
				return false;
			}
			dsp_node_ptr->acquireResource();
			break;
		}
		case kDspTargetDsp2:
		{
			if (!m_dsp_thread[1].addDspNode2Queue(dsp_node_ptr))
			{
				dsp_node_ptr->setStatus(kDspNodeInvalid);
				return false;
			}
			dsp_node_ptr->acquireResource();
			break;
		}
		case kDspTargetMultiCore:
		{
			for (i32 i = 0; i < DSP_MULTI_CORES; i++)
			{
				if (!m_dsp_thread[i].addDspNode2Queue(dsp_node_ptr, i, DSP_MULTI_CORES))
				{
					removeDspNode(dsp_node_ptr->getNodeID());
					return false;
				}
			}
			dsp_node_ptr->acquireResource(DSP_MULTI_CORES);
			break;
		}
		case kDspTargetAny:
		{
			i32 dsp_index = -1;
			i32 free_queue_size, max_free_size = 0;
			for (i32 i = 0; i < DSP_MULTI_CORES; i++)
			{
				free_queue_size = m_dsp_thread[i].getFreeQueueSize();
				if (free_queue_size > max_free_size)
				{
					max_free_size = free_queue_size;
					dsp_index = i;
				}
			}
			if (dsp_index < 0 || !m_dsp_thread[dsp_index].addDspNode2Queue(dsp_node_ptr))
			{
				dsp_node_ptr->setStatus(kDspNodeInvalid);
				return false;
			}
			dsp_node_ptr->acquireResource();
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}
#endif

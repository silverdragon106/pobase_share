#pragma once
#include "types.h"
#include <QThread>
#include <QSemaphore>
#include <mutex>

#ifdef POR_WITH_DSP
extern "C"
{
	#include "dsp_ipc_client.h"
}
typedef std::list<DspQueueItem> DspQueueList;

#define DSP_MULTI_CORES			2
#define DSP_NODE_QUEUE_SIZE		16

enum DspNodeStatusType
{
	kDspNodeNone = 0,
	kDspNodeNotStarted,
	kDspNodeRunning,
	kDspNodeFinished,
	kDspNodeFailed,
	kDspNodeInvalid,
	
	kDspNodeStatusCount,
};

enum DspTargetType
{
	kDspTargetDsp1 = 0,
	kDspTargetDsp2,
	kDspTargetAny,
	kDspTargetMultiCore,

	kDspTargetTypeCount
};

class CBaseDspApp;
class CDspNode
{
public:
	CDspNode();
	CDspNode(i32 id, u16 cmd, u8 dsp_target, CBaseDspApp* dsp_app_ptr);
	~CDspNode();

	void					prepare(u8* src_img_ptr, u8* dst_img_ptr, i32 w, i32 h, i32 src_step = 1, i32 dst_step = 1, i32 gap_size = 1);
	i32						process();
	i32						schedule();
	i32						wait();

	void					setStatus(i32 status);
	void					acquireResource(i32 count = 1);
	void					releaseResource(i32 count = 1);

	inline u16				getCmd() { return m_cmd; };
	inline i32				getNodeID() { return m_id; };
	inline i32				getStatus() { return m_status; };
	inline i32				getTarget() { return m_dsp_target; };

	inline bool				isInvalid() { return m_status == kDspNodeInvalid; };
	inline bool				isFinished() { return m_status >= kDspNodeFinished; };

public:
	i32						m_id;
	u16						m_cmd;
	std::atomic<i32>		m_status;

	u8						m_dsp_target;
	u8						m_resource_count;
	QSemaphore*				m_semaphore_ptr;
	CBaseDspApp*			m_dsp_app_ptr;

	u8*						m_src_buffer_ptr;
	u8*						m_dst_buffer_ptr;
	u16						m_width;
	u16						m_height;
	u8						m_src_step;
	u8						m_dst_step;
	u8						m_gap_size;
};

class CDspCoreThread : public QThread
{
public:
	CDspCoreThread();
	~CDspCoreThread();

	bool					initDspCoreThread(u16 dsp_core_id);
	bool					exitDspCoreThread();
	
	bool					addDspNode2Queue(CDspNode* dsp_node_ptr, i32 index = 0, i32 count = 1);
	void					removeDspQueue(i32 id);
	void					freeBuffer();
	
	void					checkDspResult(i32 timeout_ms);
	i32						getFreeQueueSize();

private:
	bool					isValidDspItem(DspQueueItem& item);

	void					run() Q_DECL_OVERRIDE;

private:
	bool					m_is_inited;
	std::atomic<bool>		m_is_thread_cancel;

	std::recursive_mutex	m_queue_mutex;
	DspQueueList			m_queue;

	IpcClientInfo			m_ipc_client;
};

class CBaseDspApp
{
public:
	CBaseDspApp();
	virtual ~CBaseDspApp();

	virtual bool			initDspApp();
	virtual bool			exitDspApp();
	
	CDspNode*				createDspNode(u16 cmd, u8 dsp_target = kDspTargetAny);
	void					removeDspNode(i32 id);
	void					clearDspNodes();

	bool					addDspNode2Queue(CDspNode* dsp_node_ptr);

private:
	bool					m_is_inited;
	std::atomic<i32>		m_node_id;
	CDspCoreThread			m_dsp_thread[DSP_MULTI_CORES];
};
#endif

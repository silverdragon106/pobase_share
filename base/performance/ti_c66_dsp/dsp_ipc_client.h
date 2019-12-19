#pragma once
#include "define.h"

#if defined(POR_IMVS2_ON_AM5728) && defined(POR_WITH_DSP)

#define DSP_CMD_NONE					(0)
#define DSP_WAIT_FOREVER				((~(0)))

#define IPC_SUCCESS						(0)
#define IPC_INIT_TIME_OUT				(1)
#define IPC_INIT_MSGQ_OPEN_FAIL			(2)
#define IPC_INIT_CMEM_INIT_FAIL			(3)
#define IPC_INIT_CMEM_ALLOC_FAIL		(4)
#define IPC_INIT_CMEM_GET_POOL_FAIL		(5)
#define IPC_INIT_CMEM_ALLOC_POOL_FAIL	(6)
#define IPC_INIT_SR_SETUP_FAIL			(7)
#define IPC_INIT_HEAP_SETUP_FAIL		(8)
#define IPC_INIT_HEAP_CREATE_FAIL		(9)
#define IPC_INIT_HEAP_ALLOC_FAIL		(10)
#define IPC_MSG_ALLOC_FAIL				(11)
#define IPC_CMEM_GET_PHY_FAIL			(12)
#define IPC_MSG_GET_FAIL				(13)
#define IPC_XL_GLOBT_FAIL				(14)
#define IPC_GLOBT_XL_FAIL				(15)
#define IPC_MSG_CREATE_FAIL				(16)
#define IPC_START_FAIL					(17)
#define IPC_INVALID_PARAM				(18)
#define IPC_QUEUE_FULL					(19)

typedef struct _DspItemContent
{
	u16					width;
	u16					src_height;
	u16					dst_height;
	u8					src_step;
	u8					dst_step;
	i16					dy_pos;
	u8					ret_offsetx;
	u8					ret_offsety;
} DspItemContent;

typedef struct _DspQueueItem
{
	i32					node_id;
	bool				is_allocated;

	// *: related with me
	u16					cmd;			//*
	u8*					src_buffer_ptr;	//*
	u8*					dst_buffer_ptr;	//*
	i32					src_buffer_size;
	i32					dst_buffer_size;
	_DspItemContent		content;

	void*				semaphore_ptr;
} DspQueueItem;

typedef struct _IpcClientInfo
{
	u16					target_id;		// 1,2
	DspQueueItem		item;

	u16					dsp_load;
} IpcClientInfo;

extern int ipc_init(i32 init_param);
extern int ipc_uninit();

// all functions: ipc_ci, ipc_ci->target_id must be valid
extern int ipc_start(IpcClientInfo* ipc_client_info);
extern int ipc_end(IpcClientInfo* ipc_client_info);
extern int ipc_send_init_msg(IpcClientInfo *ipc_client_info);
extern int ipc_send_cmd_msg(IpcClientInfo *ipc_client_info);
extern int ipc_send_quit_msg(IpcClientInfo *ipc_client_info);
extern int ipc_wait_msg(IpcClientInfo *ipc_client_info, u32 timeout_ms, u8** receive_buff);

#endif


#pragma once

#include <QObject>
#include <QSharedPointer>
#include "queue.h"
#include "ring_buffer.h"
#include "types.h"

/**
* Pipeline
* @brief
*	for process from jpeg or raw data to image can be drawn.
*
* @code
*
* @endcode
**/
class Pipeline : public QObject
{
	Q_OBJECT

	const int		kMaxOutQueueSize = 3;		/* max size of RawQueue | OutQueue. */
												/* 주의 : MaxOutQueueSize만큼 기억구역을 할당하라수 있도록 
												rawRingBuffer와 outRingBuffer크기를 설정해야 한다. 
												이 값은 CQueue::kMaxQueueSize보다는 작아야 한다.*/
												
	const int		kMaxFrameWidth = 4000;
	const int		kMaxFrameHeight = 3000;
public:
	Pipeline(int nVix, QObject* parent = NULL);
	~Pipeline();

	bool			initInstance();
	void			exitInstance();

	void			start(int decoder_type);
	void			stop();

	void			setDecoderParams(int width, int height);
	int				getCameraId();

	bool			isRawMode();

	/* thread-safe functions */
	bool			addToInQueue(void *p);
	bool			addToOutQueue(void *p);
	bool			addToRawQueue(void *p);

	void*			takeOneFromInQueue();
	void*			takeOneFromOutQueue();
	void*			takeOneFromRawQueue();

	int				getInQueueSize();
	int				getOutQueueSize();
	int				getRawQueueSize();

	/* thread-unsafe functions */

	/* ring buffer functions */
	u8*				allocInFrame(int len);
	void			freeInFrame(u8* p);

	u8*				allocOutFrame(int len);
	void			freeOutFrame(u8* p);

	u8*				allocRawFrame(int len);
	void			freeRawFrame(u8* p, bool bRemoveFromQueue = true);

	/* main thread callback */
	void			animate(i64 timestamp);
public:
	/* encode-decode channel */
	CQueue<void*>	m_incoming_queue;
	CQueue<void*>	m_outgoing_queue;
	CRingBuffer		m_in_buffer;
	CRingBuffer		m_out_buffer;

	/* raw channel */
	CQueue<void*>	m_raw_queue;
	CRingBuffer		m_raw_buffer;

	i64				m_last_timestamp;			/* 마지막으로 발송된 프레임의 timestamp */
	int				m_vid;						/* camera index. */
	int				m_decoder_type;
};

typedef QSharedPointer<Pipeline> PipelineRef;
#include "pipeline.h"
#include "queue.h"
#include "base_decoder.h"
#include "video_streamer.h"

Pipeline::Pipeline(int nVix, QObject* parent)
	: QObject(parent), 
	  m_vid(nVix),
	  m_decoder_type(0),
	  m_last_timestamp(0)
{
}

Pipeline::~Pipeline()
{
	stop();
}

bool Pipeline::initInstance()
{
	return true;
}

void Pipeline::exitInstance()
{
	stop();
}

void Pipeline::start(int decoder_type)
{
	if (decoder_type == kPOEncoderNetworkRaw || decoder_type == kPOEncoderIPCRaw)
	{
		/* rawFrame*/
		int raw_header_size = sizeof(RawFrame);
		/* 출력바퍼크기를 최대대기렬크기+1 개의 프레임을 담을수 있게 생성한다. */
		int one_frame_size = kMaxFrameWidth * kMaxFrameHeight + raw_header_size;
		m_raw_buffer.Create((kMaxOutQueueSize + 1) * one_frame_size);
	}
	else
	{
		int enc_header_size = sizeof(EncodedFrame);
		int dec_header_size = sizeof(DecodedFrame);
		/* 입력바퍼크기를 최대대기렬크기 개의 프레임을 담을수 있게 생성한다. */
		/* 출력바퍼크기를 최대대기렬크기+1 개의 프레임을 담을수 있게 생성한다. */

		int one_enc_frame_size = kMaxFrameWidth * kMaxFrameHeight + enc_header_size;
		int one_dec_frame_size = kMaxFrameWidth * kMaxFrameHeight + dec_header_size;

		m_in_buffer.Create(kMaxOutQueueSize * one_enc_frame_size);
		m_out_buffer.Create((kMaxOutQueueSize + 1) * one_dec_frame_size);
	}
	m_decoder_type = decoder_type;
	m_last_timestamp = 0;
}

void Pipeline::stop()
{
	m_incoming_queue.clear();
	m_outgoing_queue.clear();
	m_raw_queue.clear();

	m_in_buffer.Delete();
	m_out_buffer.Delete();
	m_raw_buffer.Delete();
}

int Pipeline::getCameraId()
{
	return m_vid;
}

bool Pipeline::addToInQueue(void *p)
{
	return m_incoming_queue.put(p);
}

bool Pipeline::addToOutQueue(void *p)
{
	return m_outgoing_queue.put(p);
}

bool Pipeline::addToRawQueue(void *p)
{
	return m_raw_queue.put(p);
}

void* Pipeline::takeOneFromInQueue()
{
	void* p = NULL;
	if (m_incoming_queue.pop(&p, 50))
		return p;
	return NULL;
}

void* Pipeline::takeOneFromOutQueue()
{
	void* p = NULL;
	if (m_outgoing_queue.pop(&p))
		return p;
	return NULL;
}

void* Pipeline::takeOneFromRawQueue()
{
	void* p = NULL;
	if (m_raw_queue.pop(&p))
		return p;
	return NULL;
}

int Pipeline::getInQueueSize()
{
	return m_incoming_queue.getSize();
}

int Pipeline::getOutQueueSize()
{
	return m_outgoing_queue.getSize();
}

int Pipeline::getRawQueueSize()
{
	return m_raw_queue.getSize();
}

u8* Pipeline::allocInFrame(int len)
{
	return m_in_buffer.Malloc(len);
}

void Pipeline::freeInFrame(u8* p)
{
	m_in_buffer.Free(p);
}

u8* Pipeline::allocOutFrame(int len)
{
	return m_out_buffer.Malloc(len);
}

void Pipeline::freeOutFrame(u8* p)
{
	m_out_buffer.Free(p);
}

u8* Pipeline::allocRawFrame(int len)
{
	return m_raw_buffer.Malloc(len);
}

void Pipeline::freeRawFrame(u8* p, bool bRemoveFromQueue)
{
	if (bRemoveFromQueue)
		m_raw_queue.remove(p);
	m_raw_buffer.Free(p);
}

void Pipeline::animate(i64 timestamp)
{
	/* Video Synchronize */
	/* @ToDo
	1. 파이프라인의 timestamp가 0이면 최신프레임이 있을때 신호를 보낸다.
	2. timestamp가 값이 있으면 timestamp를 가진 프레임을 얻어 신호를 보낸다.
	3. 파이프라인의 출력대기렬개수가 kMaxOutQueueSize개 이상이면 낡은 프레임자료를 뽑아없앤다.
	*/

	/*
	주의: newRawFrame()과 newFrame() 신호전송부분을 조사하여 없애고 이 함수에서 신호를 전송한다.
		이때 frame buffer의 해방이 정확히 동작하는가를 따져보아야 한다.
	*/

	CQueue<void*>* queue = isRawMode()? &m_raw_queue : &m_outgoing_queue;
	
	/* timestamp==0이면 최신프레임을 뽑는다. */
	if (timestamp == 0)
	{
		/* Queue에 출력자료가 있는가를 검사한다. 
		만일 출력자료중에 현재의 timestamp보다 큰프레임이 있다면 
		새프레임으로 간주하고 최신프레임이 도착했다는 신호를 날린다. */
		/* queue는 Decoding스레드에서 접근하기때문에 lock를 하고 사용해야 한다.*/
		queue->lock();

		CQueue<void*>::Container &buffer = queue->getBuffer();
		if (buffer.size() > 0)
		{
			if (isRawMode())
			{
				/* Raw프레임방식에서 마지막프레임의 timestamp를 검사하여 
				새프레임이면 신호를 날린다. */
				RawFrame* frame = (RawFrame*)buffer.last();
				if (m_last_timestamp < frame->timestamp)
				{
					/* Direct Connection으로 련결되였다고 가정한다. */
					/* 같은 스레드안에서 동작한다고 가정한다. */
#ifdef USE_STREAMER
					emit g_stream_mgr->newFrameReady(m_vid, isRawMode(), (void*)frame);
#endif
					m_last_timestamp = frame->timestamp;
				}
			}
			else
			{
				/* Decoding프레임방식에서 마지막프레임의 timestamp를 검사하여
				새프레임이면 신호를 날린다. */
				DecodedFrame* frame = (DecodedFrame*)buffer.last();
				if (m_last_timestamp < frame->timestamp)
				{
#ifdef USE_STREAMER
					emit g_stream_mgr->newFrameReady(m_vid, isRawMode(), (void*)frame);
#endif
					m_last_timestamp = frame->timestamp;
				}
			}
		}
		queue->unlock();
	}
	/* 아니면 지정한 timestamp를 가진 프레임을 찾는다. */
	else
	{
		/* Queue에 timestamp를 가진 출력자료가 있는가를 검사한다.
		  있다면 timestamp를 가진 프레임을 신호로 날린다. */
		/* queue는 Decoding스레드에서 접근하기때문에 lock를 하고 사용해야 한다.*/
		queue->lock();

		CQueue<void*>::Container &buffer = queue->getBuffer();

		if (isRawMode())
		{
			foreach(void* frame, buffer)
			{
				RawFrame* raw_frame = (RawFrame*)frame;
				if (timestamp == raw_frame->timestamp)
				{
#ifdef USE_STREAMER
					emit g_stream_mgr->timestampFrameReady(m_vid, isRawMode(), frame);
#endif
					break;
				}
			}
		}
		else
		{
			foreach(void* frame, buffer)
			{
				DecodedFrame* decoded_frame = (DecodedFrame*)frame;
				if (timestamp == decoded_frame->timestamp)
				{
#ifdef USE_STREAMER
					emit g_stream_mgr->timestampFrameReady(m_vid, isRawMode(), frame);
#endif
					break;
				}
			}
		}

		queue->unlock();
	}

	/* flush queue */
	/* Queue의 개수가 kMaxOutQueueSize개 이상이면 Queue에서 오래된 프레임을 해방한다. */
	queue->lock();

	CQueue<void*>::Container &buffer = queue->getBuffer();
	/* remove old frame */
	if (isRawMode())
	{
		CQueue<void*>::Container::Iterator it = buffer.begin();
		for (; it != buffer.end(); )
		{
			RawFrame* raw = (RawFrame*)(*it);
			if (raw && raw->timestamp < m_last_timestamp - 1000)
			{
				POSAFE_DELETE(raw);
				it = buffer.erase(it);
			}
			else
			{
				break;
			}
		}
	}
	else
	{
		CQueue<void*>::Container::Iterator it = buffer.begin();
		for (; it != buffer.end();)
		{
			DecodedFrame* decoded = (DecodedFrame*)(*it);
			if (decoded && decoded->timestamp < m_last_timestamp - 1000)
			{
				POSAFE_DELETE(decoded);
				it = buffer.erase(it);
			}
			else
			{
				break;
			}
		}
	}

	/* overflow max queue size. */
	if (buffer.size() >= kMaxOutQueueSize)
	{
		/* 초과개수 */
		int overed_count = buffer.size() - kMaxOutQueueSize + 1;

		/* 처음부터 초과된 개수만큼  기억공간을 해방하고 대기렬에서 삭제한다. */
		/* 대기렬에서 처음요소가 제일 오래된 요소이고 제일 마지막요소가 최신요소이다. */
		CQueue<void*>::Container::Iterator it = buffer.begin();
		for (int i = 0; i < overed_count; i++)
		{
			/* 기억공간 해방 */
			if (isRawMode())
			{
				RawFrame* raw = (RawFrame*)(*it);
				POSAFE_DELETE(raw);
			}
			else
			{
				DecodedFrame* decoded = (DecodedFrame*)(*it);
				POSAFE_DELETE(decoded);
			}

			/* 대기렬에서 삭제 */
			it = buffer.erase(it);
		}
	}
	
	queue->unlock();
}

bool Pipeline::isRawMode()
{
	return m_decoder_type == kPOEncoderIPCRaw || m_decoder_type == kPOEncoderNetworkRaw;
}

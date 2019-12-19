#include "tcp_client.h"
#include "ipc.h"
#include "video_streamer.h"
#include "logger.h"
#include "pipeline.h"
#include "mvh_base.h"
#include "mvh_data.h"
#include "base.h"
#include "time_tracker.h"
#include "packet.h"
#include <QtCore>

#ifndef WIN32
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#include <wspiapi.h>
#endif

VideoStreamer* g_stream_mgr = NULL;
VideoStreamer::VideoStreamer()
{
	m_bStarted = false;
}

VideoStreamer::~VideoStreamer()
{
	stop();
}

bool VideoStreamer::prepare(int camera_count, int decoder_type)
{
	/* 로컬에서는 생화상자료를 통신하기때문에 decoder를 실행시킬필요 없다. */

#ifdef POR_WITH_STREAM
	if (g_tcp_cmd->getHostIP() == "127.0.0.1")
	{
		decoder_type = kPOEncoderIPCRaw;
	}
#endif

	for (int i = 0; i < camera_count; i++)
	{
		m_decoders << new CBaseDecoder(i, decoder_type);
		m_decoders[i]->initInstance();
	}

	m_bStarted = false;
	return true;
}

void VideoStreamer::start()
{
	m_bStarted = true;

	for (int i = 0; i < m_decoders.size(); i++)
	{
		m_decoders[i]->start();
	}
}

void VideoStreamer::stop()
{
	m_bStarted = false;
	for (int i = 0; i < m_decoders.size(); i++)
	{
		CBaseDecoder* decoder = m_decoders[i];
		POSAFE_DELETE(decoder);
	}
	m_decoders.clear();
}

CBaseDecoder* VideoStreamer::getDecoder(int vid)
{
	if (vid >= m_decoders.size())
		return NULL;
	return m_decoders[vid];
}

void VideoStreamer::onReceive(Packet* pak)
{
	if (!m_bStarted)
		return;

	/* 1. if encoded stream */
	if (pak->getSubCmd() == kMVSubTypeImageStream)
	{
		u8* data = pak->getData();
		int vid = 0;
		/* read only camera index. */
		CPOBase::memRead(vid, data);				   /* camera index */

		/* check camera index. if camera index is out of pipelines */
		if (vid >= m_decoders.size())
		{
			logerror("video index is out of pipelines.");
			return;
		}

		/* send to jpeg data do decode thread. */
		m_decoders[vid]->pushEncodedFrameDataMV(pak->getData(), pak->getDataLen());
	}
	// 2. else raw stream type
	else if (pak->getSubCmd() == kMVSubTypeImageRaw)
	{
		/* Tcp통신에서는 생화상자료전송방식을 리용하지 않는다. */
		logerror("Unsupport raw image process on Tcp.");
	}
}

void VideoStreamer::onReceiveIPC(Packet* pak)
{
	if (!m_bStarted)
		return;
	
	// check video stream type.
	if (pak->getSubCmd() == kMVSubTypeImageRaw)
	{
		u8* data = pak->getData();
		i64 timestamp = 0;
		int vid = 0;
		int width = 0;
		int height = 0;
		int bpp = 0;

		/* read raw frame info */
		// read raw data.
		CPOBase::memRead(vid, data);
		CPOBase::memRead(timestamp, data);
		CPOBase::memRead(width, data);
		CPOBase::memRead(height, data);
		CPOBase::memRead(bpp, data);

		/* validate raw frame info */
		if (vid < 0 || vid >= m_decoders.size())
		{
			logerror("vid is out of pipelines.");
			return;
		}
		if (width < 0 || width > 3000 || height < 0 || height > 3000)
		{
			logerror("too large image size.");
			return;
		}
		/* read ffmpeg raw data */
		/* allocate raw frame buffer
		   after use, must release. */
		/* pak->len = data_len + 20(width,height,timestamp, len) bytes. *
		/* maybe datalen is width*height */
		int metalen = (int)(data - pak->getData());
		int datalen = pak->getDataLen() - metalen; 
		m_decoders[vid]->pushRawFrame(data, datalen, width, height, bpp, timestamp);
	}
}

/*
* animate
*	파이프라인의 출력프레임대기렬을 조사하여 프레임의 timestamp가 현재의 timestamp보다
*	작은 모든 프레임들을 대기렬에서 뽑는다.
* @timestamp:
*	ng화상을 현시해야하는경우 ng화상의 timestamp
*	ng화상이 없는경우 현재의 timestamp이다.
*/
void VideoStreamer::animate(i64 timestamp)
{
	if (!m_bStarted)
		return;

	foreach(CBaseDecoder* decoder, m_decoders)
	{
		decoder->getPipeline()->animate(timestamp);
	}
}

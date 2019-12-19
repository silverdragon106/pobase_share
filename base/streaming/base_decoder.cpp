#include "base_decoder.h"
#include "base.h"
#include "pipeline.h"
//#include "mvh_common.h"

#ifdef POR_SUPPORT_FFMPEG
#ifdef POR_WINDOWS
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "avformat.lib")
#endif

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
}
#endif

#ifdef POR_SUPPORT_GSTREAMER
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

bool CBaseDecoder::_is_gst_inited = false;
#endif

//////////////////////////////////////////////////////////////////////////
// EncodedFrame
//////////////////////////////////////////////////////////////////////////
EncodedFrame::EncodedFrame() : data(NULL), width(0), height(0), timestamp(0), length(0), bpp(1)
{

}

EncodedFrame::~EncodedFrame()
{
	if (!m_pAllocator.isNull())
	{
		m_pAllocator->freeInFrame(data);
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);
	}
}

bool EncodedFrame::alloc(int len)
{
	if (len <= 0)
		return false;
	if (!m_pAllocator.isNull())
	{
		data = m_pAllocator->allocInFrame(len);
		if (!data)
			return false;

		length = len;
	}
	else
	{
		data = new u8[len];
		if (!data)
			return false;

		length = len;
	}
	return true;
}

void EncodedFrame::setAllocator(PipelineRef pPipeline)
{
	m_pAllocator = pPipeline;
}

//////////////////////////////////////////////////////////////////////////
// DecodedFrame
//////////////////////////////////////////////////////////////////////////
DecodedFrame::DecodedFrame() : data(NULL), width(0), height(0), timestamp(0), bpp(1)
{
}

DecodedFrame::~DecodedFrame()
{
	if (!m_pAllocator.isNull())
	{
		m_pAllocator->freeOutFrame(data);
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);
	}
}

void DecodedFrame::setAllocator(PipelineRef pAllocator)
{
	m_pAllocator = pAllocator;
}

bool DecodedFrame::setYuvData(u8* decoded[DECODED_CHANNEL_COUNT], int linesize[DECODED_CHANNEL_COUNT], int width, int height)
{
	if (!m_pAllocator.isNull())
	{
		if (!data)
		{
			data = m_pAllocator->allocOutFrame(width * height * 3 / 2);
		}
		
		if (!data)
		{
			simplelog("[Error] Failed to allocate frame data.[1]");
			return false;
		}
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);
		data = new u8[width*height * 3 / 2];
		if (!data)
		{
			simplelog("[Error] Failed to allocate frame data.[2]");
			return false;
		}
	}

	int	nLength[DECODED_CHANNEL_COUNT];
	int	nStride[DECODED_CHANNEL_COUNT];

	this->width = width;
	this->height = height;
	this->length = width * height * 3 / 2;

	nStride[0] = width;
	nStride[1] = width >> 1;
	nStride[2] = width >> 1;

	int heights[] = { height, height / 2, height / 2 };

	nLength[0] = nStride[0] * heights[0];
	nLength[1] = nStride[1] * heights[1];
	nLength[2] = nStride[2] * heights[2];

	/* copy data */
	u8* dst = data;
	u8* src = NULL;
	for (int i = 0; i < DECODED_CHANNEL_COUNT; i++)
	{
		src = decoded[i];
		for (int j = 0; j < heights[i]; j++)
		{
			memcpy(dst, src, nStride[i]);
			dst += nStride[i];
			src += linesize[i];
		}
	}

	return true;
}

bool DecodedFrame::setGrayData(u8* decoded, int /*decoded_len*/, int width, int height)
{
	if (!m_pAllocator.isNull())
	{
		if (!data)
		{
			data = m_pAllocator->allocOutFrame(width * height);
		}

		if (!data)
		{
			simplelog("[Error] Failed to allocate frame data.[1]");
			return false;
		}
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);
		data = new u8[width*height];
		if (!data)
		{
			simplelog("[Error] Failed to allocate frame data.[2]");
			return false;
		}
	}

	this->width = width;
	this->height = height;
	this->length = width * height;

	memcpy(data, decoded, this->length);
	return true;
}

//////////////////////////////////////////////////////////////////////////
// CBaseDecoder
//////////////////////////////////////////////////////////////////////////
CBaseDecoder::CBaseDecoder(int decoder_index, int decoder_type)
{
	_decoder_index = decoder_index;
	_decoder_type = decoder_type;

	m_bAbort = false;
	m_pPipeline = PipelineRef();

#ifdef POR_SUPPORT_FFMPEG
	_avcodec = NULL;
	_avframe = NULL;
	_avcodec_context = NULL;
#endif
#ifdef POR_SUPPORT_GSTREAMER
	_pipeline = NULL;
	_source = NULL;
	_decoder = NULL;
	_sink = NULL;
	_buffer_pool = NULL;
	_is_enabled_checking_caps = false;
#endif
}

CBaseDecoder::~CBaseDecoder()
{
	exitInstance();
}

void CBaseDecoder::registerCodecs()
{
#ifdef POR_SUPPORT_FFMPEG
	avcodec_register_all();
#endif
}

bool CBaseDecoder::initInstance()
{
	if (!createDecoder())
	{
		return false;
	}

	if (!createPipeline())
	{
		return false;
	}

	return true;
}

void CBaseDecoder::exitInstance()
{
	stop();
}

bool CBaseDecoder::createDecoder()
{
#ifdef POR_WITH_STREAM
#ifdef POR_SUPPORT_FFMPEG
	if (_decoder_type == kPOEncoderFFMpegH264 || _decoder_type == kPOEncoderFFMpegMJpeg)
	{
		return createDecoderFFmpeg();
	}
#endif
#ifdef POR_SUPPORT_GSTREAMER
	else if (_decoder_type == kPOEncoderGStreamerH264 || _decoder_type == kPOEncoderFFMpegMJpeg)
	{
		return createDecoderGstreamer();
	}
#endif

	return false;
#else
	return true;
#endif
}

void CBaseDecoder::releaseDecoder()
{
#ifdef POR_SUPPORT_FFMPEG
	if (_decoder_type == kPOEncoderFFMpegH264 || _decoder_type == kPOEncoderFFMpegMJpeg)
	{
		releaseDecoderFFmpeg();
	}
#endif
#ifdef POR_SUPPORT_GSTREAMER
	if (_decoder_type == kPOEncoderGStreamerH264 || _decoder_type == kPOEncoderGStreamerMJpeg)
	{
		releaseDecoderGstreamer();
	}
#endif
}

bool CBaseDecoder::createPipeline()
{
	m_pPipeline = PipelineRef(new Pipeline(_decoder_index));
	if (m_pPipeline.isNull())
		return false;

	return m_pPipeline->initInstance();
}

void CBaseDecoder::releasePipeline()
{
	if (!m_pPipeline.isNull())
	{
		{
			EncodedFrame* encoded_frame = NULL;
			while ((encoded_frame = (EncodedFrame*)m_pPipeline->takeOneFromInQueue()))
			{
				POSAFE_DELETE(encoded_frame);
			}
		}
		
		{
			DecodedFrame* decoded_frame = NULL;
			while ((decoded_frame = (DecodedFrame*)m_pPipeline->takeOneFromOutQueue()))
			{
				POSAFE_DELETE(decoded_frame);
			}
		}

		{
			RawFrame* raw_frame = NULL;
			while ((raw_frame = (RawFrame*)m_pPipeline->takeOneFromRawQueue()))
			{
				POSAFE_DELETE(raw_frame);
			}
		}
		m_pPipeline->exitInstance();
	}
}

bool CBaseDecoder::isRawMode()
{
	return _decoder_type == kPOEncoderIPCRaw || _decoder_type == kPOEncoderNetworkRaw;
}
void CBaseDecoder::start()
{
	if (!m_pPipeline.isNull())
	{
		m_pPipeline->start(_decoder_type);
	}

	if (!isRawMode())
	{
		if (!isRunning())
		{
			m_bAbort = false;

			QThread::start();
		}
	}
}
void CBaseDecoder::stop()
{
	if (isRunning())
	{
		m_bAbort = true;
		if (!wait(500))
		{
			terminate();
			wait();
		}
	}

	releasePipeline();
	releaseDecoder();
}

PipelineRef CBaseDecoder::getPipeline()
{
	return m_pPipeline;
}
/*
* setParam
*	set decode parameters
*/
bool CBaseDecoder::setParams(int width, int height, RawDataFormat format)
{
#ifdef POR_SUPPORT_FFMPEG
	/* Nothing to do */
#endif
#ifdef POR_SUPPORT_GSTREAMER
	if (!_buffer_pool)
		return false;

	/* check data format. */
	if (format != kRawDataGray8)
	{
		simplelog("Decoder only support kDrawDataGray8.");
		return false;
	}

	/* calc bytes per pixel from data format. */
	int bpp = 1;
	if (format == kRawDataGray8)
		bpp = 1;
	else if (format == kRawDataRGB)
		bpp = 3;

	/* set buffer pool params for decoding. */
	/* set buffer size to use buffer pool */
	guint size = 0;
	guint min_buffers = 0;
	guint max_buffers = 0;
	GstCaps *caps = NULL;
	GstStructure *pool_config = NULL;
	pool_config = gst_buffer_pool_get_config(_buffer_pool);
	gst_buffer_pool_config_get_params(pool_config, &caps, &size, &min_buffers, &max_buffers);
	gst_buffer_pool_config_set_params(pool_config, caps, width * height * bpp, kMinBufferPoolCount, kMaxBufferPoolCount);
	gst_buffer_pool_set_config(_buffer_pool, pool_config);
	gst_buffer_pool_set_active(_buffer_pool, true);
#endif
	/* set values */
	_width = width;
	_height = height;
	_format = format;

	return true;
}

/*
* updateParams
*/
bool CBaseDecoder::updateParams(int width, int height, RawDataFormat format)
{
	if (_width == width && _height == height && _format == format)
		return true;

	return setParams(width, height, format);
}

bool CBaseDecoder::pushEncodedFrameDataSC(u8* data, int /*nSize*/)
{
	if (!isRunning())
	{
		simplelog("[Warning] Decoding thread is not started.");
		return false;
	}

	bool bCalibed;
	f32 tPixel2mm = 0.0f;
	i32 vid, w, h, datalen;
	i64 timestamp, pts;
	i32 bpp = 1;

	u8* pbuffer = data;
	CPOBase::memRead(vid, pbuffer);
	CPOBase::memRead(w, pbuffer);
	CPOBase::memRead(h, pbuffer);
	CPOBase::memRead(bpp, pbuffer);				/* 1: gray, 2: yuv, 3: rgb */ /* not used this parameter */
	CPOBase::memRead(bCalibed, pbuffer);
	CPOBase::memRead(tPixel2mm, pbuffer);
	CPOBase::memRead(timestamp, pbuffer);
	CPOBase::memRead(pts, pbuffer);
	CPOBase::memRead(datalen, pbuffer);

	return pushEncodedFrame(pbuffer, datalen, w, h, bpp, timestamp);
}

bool CBaseDecoder::pushEncodedFrameDataMV(u8* data, int /*nSize*/)
{
	u8* pbuffer = data;
	int vid = 0;
	int width = 0;
	int height = 0;
	bool is_snaped = false;
	i64 pts = 0;
	i64 timestamp = 0;
	int datalen = 0;
	int bpp = 0;
	
	/* read frame packet header. */
	CPOBase::memRead(vid, pbuffer);				/* camera index */
	CPOBase::memRead(width, pbuffer);			/* frame width */
	CPOBase::memRead(height, pbuffer);			/* frame height */
	CPOBase::memRead(bpp, pbuffer);				/* 1: gray, 2: yuv, 3: rgb */ /* not used this parameter */
	CPOBase::memRead(is_snaped, pbuffer);
	CPOBase::memRead(timestamp, pbuffer);		/* frame timestamp */
	CPOBase::memRead(pts, pbuffer);		/* frame timestamp */
	CPOBase::memRead(datalen, pbuffer);			/* encoded frame data size */

	if (datalen <= 0)
		return false;

	/* 인코딩기억공간을 파이프라인의 입력RingBuffer로부텉 할당한다.
	할당된 기억공간은 사용후 반드시 해방하여야 한다.
	만일 해방하지 않으면 다음 인코딩에서 기억공간할당오유가 나오게 된다.
	아래의 인코딩기억공간은 run()에서 해방한다.
	입력대기렬에 넣기실패나 자료에 오유가 있으면 즉시 해방한다. */

#if defined(POR_SUPPORT_GSTREAMER) || defined(POR_SUPPORT_FFMPEG)
	if (!isRunning())
	{
		simplelog("[Warning] Decoding thread is not started.");
		return false;
	}

	return pushEncodedFrame(pbuffer, datalen, width, height, bpp, timestamp);
#else /* Raw data format */
	return pushRawFrame(pbuffer, datalen, width, height, bpp, timestamp);
#endif
}

bool CBaseDecoder::pushEncodedFrame(u8* pbuffer, int datalen, int width, int height, int bpp, i64 timestamp)
{
	EncodedFrame* encoded_frame = new EncodedFrame();
	if (encoded_frame)
	{
		encoded_frame->setAllocator(m_pPipeline);
		if (encoded_frame->alloc(datalen))
		{
			CPOBase::memRead(encoded_frame->data, datalen, pbuffer);
			encoded_frame->width = width;
			encoded_frame->height = height;
			encoded_frame->timestamp = timestamp;
			encoded_frame->bpp = bpp;

			if (!m_pPipeline->addToInQueue(encoded_frame))
			{
				simplelog("[Error] Encoded Queue is full.[1]");
				POSAFE_DELETE(encoded_frame);
				return false;
			}
		}
		else
		{
			simplelog("[Error] Encoded Frame Alloc Error.[1]");
			POSAFE_DELETE(encoded_frame);
			return false;
		}
	}
	else
	{
		simplelog("[Error] Encoded Frame Alloc Error.[2]");
		return false;
	}

	return true;
}

bool CBaseDecoder::pushRawFrame(u8* pbuffer, int datalen, int width, int height, int bpp, i64 timestamp)
{
	RawFrame* raw_frame = new RawFrame();
	if (raw_frame)
	{
		raw_frame->setAllocator(m_pPipeline);
		if (raw_frame->alloc(width*height))
		{
			raw_frame->width = width;
			raw_frame->height = height;
			raw_frame->timestamp = timestamp;
			raw_frame->bpp = bpp;

			// pak->len = data_len + 20(width,height,timestamp, len) bytes.
			if (raw_frame->length == datalen)
			{
				CPOBase::memRead(raw_frame->data, datalen, pbuffer);
				if (!m_pPipeline->addToRawQueue(raw_frame))
				{
					simplelog("[Error]Full raw queue.");
					POSAFE_DELETE(raw_frame);
					return false;
				}
			}
			else
			{
				simplelog(QString("[Error]raw frame data length is wrong. len=%1, width=%2, height=%3").arg(datalen).arg(width).arg(height));
				POSAFE_DELETE(raw_frame);
				return false;
			}
		}
		else
		{
			simplelog("[Error]Failed to allocate raw frame.");
			POSAFE_DELETE(raw_frame);
			return false;
		}
	}
	else
	{
		simplelog("[Error]Failed to new RawFrame.");
		return false;
	}

	return true;
}

DecodedFrame* CBaseDecoder::decode(u8*& encoded_buf, int& encoded_len)
{
#ifdef POR_SUPPORT_FFMPEG
	if (_decoder_type == kPOEncoderFFMpegH264 || _decoder_type == kPOEncoderFFMpegMJpeg)
	{
		return decodeFFmpeg(encoded_buf, encoded_len);
	}
#endif
#ifdef POR_SUPPORT_GSTREAMER
	if (_decoder_type == kPOEncoderGStreamerH264 || _decoder_type == kPOEncoderGStreamerMJpeg)	
	{
		return decodeGstreamer(encoded_buf, encoded_len);
	}
#endif
	return NULL;
}

void CBaseDecoder::run()
{
	if (!m_pPipeline)
		return;

	while (!m_bAbort)
	{
		EncodedFrame* encoded_frame = (EncodedFrame*)m_pPipeline->takeOneFromInQueue();
		if (encoded_frame)
		{
			u8* encoded_buf = encoded_frame->data;
			int encoded_len = encoded_frame->length;
			int color_mode = kRawDataGray8;
			if (encoded_frame->bpp > 1)
			{
				color_mode = kRawDataRGB;
			}

			if (updateParams(encoded_frame->width, encoded_frame->height, kRawDataGray8))
			{
				while (encoded_len > 0)
				{
					DecodedFrame* decoded_frame = decode(encoded_buf, encoded_len);
					if (decoded_frame)
					{
						decoded_frame->timestamp = encoded_frame->timestamp;

#ifdef USE_STREAMER
						/* VideoStreamer::animate()에서 프레임동기화를 진행하면서
						새프레임이 추가되였다는 신호를 날린다. */
						if (!m_pPipeline->addToOutQueue(decoded_frame))
						{
							simplelog("[Warning] Full to Decoded queue.");
							POSAFE_DELETE(decoded_frame);
						}
#else
						/* theStreamer->animate에서 프레임 notify를 진행하기 때문에 아래의 함수는 필요가 없다. */
						onDecodedFrame(decoded_frame);
#endif
					}
					else
					{
						break;
					}
				}
			}

			POSAFE_DELETE(encoded_frame);
		}

		QThread::msleep(1);
	}
}

bool CBaseDecoder::createDecoderFFmpeg()
{
#ifdef POR_SUPPORT_FFMPEG
	AVCodecID codec_id = AV_CODEC_ID_MJPEG;
	_avcodec = avcodec_find_decoder(codec_id);
	if (!_avcodec)
	{
		simplelog("[Error] Failed to Create h264 Decoder.");
		return false;
	}
	_avcodec_context = avcodec_alloc_context3(_avcodec);
	if (!_avcodec_context)
	{
		simplelog("[Error] Failed to Create Decode Context.");
		return false;
	}

	if (_avcodec->capabilities & AV_CODEC_CAP_TRUNCATED)
		_avcodec_context->flags |= AV_CODEC_FLAG_TRUNCATED; // we do not send complete frames

	if (codec_id == AV_CODEC_ID_H264)
	{
		av_opt_set(_avcodec_context, "threads", "1", 0);
	}

	/* open it */
	if (avcodec_open2(_avcodec_context, _avcodec, NULL) < 0)
	{
		simplelog("[Error] Failed to Load Decoder.");
		return false;
	}

	_avframe = av_frame_alloc();
	if (!_avframe)
	{
		simplelog("[Error] Failed to Load Decoder.");
		return false;
	}
#endif
	return true;
}

void CBaseDecoder::releaseDecoderFFmpeg()
{
#ifdef POR_SUPPORT_FFMPEG
	if (_avframe)
	{
		av_frame_free(&_avframe);
		_avframe = NULL;
	}
	if (_avcodec_context)
	{
		avcodec_free_context(&_avcodec_context);
		_avcodec_context = NULL;
		_avcodec = NULL;
	}
#endif
}

void CBaseDecoder::initializeGstreamer()
{
#ifdef POR_SUPPORT_GSTREAMER
	if (!_is_gst_inited)
	{
		gst_init(NULL, NULL);
		_is_gst_inited = true;
	}
#endif
}

bool CBaseDecoder::createDecoderGstreamer()
{
#ifdef POR_SUPPORT_GSTREAMER
	/* Initialize GStreamer. */
	initializeGstreamer();

	/* Create empty pipeline. */
	_pipeline = gst_pipeline_new("test-pipeline");
	if (!_pipeline)
	{
		simplelog("Failed to create test-pipeline element.");
		return false;
	}

	/* Create pipeline elements. Source, Encoder, Sink.*/
	_source = gst_element_factory_make("appsrc", "source");
	if (!_source)
	{
		simplelog("Failed to create appsrc element.");
		return false;
	}
	_decoder = gst_element_factory_make("jpegdec", "decoder");
	if (!_decoder)
	{
		simplelog("Failed to create jpegenc element.");
		return false;
	}
	_sink = gst_element_factory_make("appsink", "sink");
	if (!_sink)
	{
		simplelog("Failed to create appsink element.");
		return false;
	}


	/* Build the pipeline. */
	/* appsrc->jpegdec->appsink*/
	gst_bin_add_many(GST_BIN(_pipeline), _source, _decoder, _sink, NULL);
	if (!gst_element_link(_source, _decoder))
	{
		simplelog("Failed to link the elements between Source and Decoder.");
		return false;
	}
	if (!gst_element_link(_decoder, _sink))
	{
		simplelog("Failed to link the elements between Encoder and Sink.");
		return false;
	}

	/* Stop playing the pipeline. */
	gst_element_set_state(_pipeline, GST_STATE_PLAYING);

	/* Create buffer pool for decoding. */
	_buffer_pool = gst_buffer_pool_new();
#endif
	return true;
}

void CBaseDecoder::releaseDecoderGstreamer()
{
#ifdef POR_SUPPORT_GSTREAMER
	if (_pipeline)
	{
		/* Stop playing and release pipeline.
		* Note that _source, _encoder and _sink are released automatically
		* because child elements are released by parent element.
		*/
		gst_element_set_state(_pipeline, GST_STATE_NULL);
		gst_object_unref(_pipeline);
	}

	if (_buffer_pool)
	{
		gst_buffer_pool_set_active(_buffer_pool, FALSE);
		gst_object_unref(_buffer_pool);
	}
#endif
}

DecodedFrame* CBaseDecoder::decodeFFmpeg(u8*& encoded_buf, int& encoded_len)
{
#ifdef POR_SUPPORT_FFMPEG
	AVPacket pkt;
	int len = 0, got_frame = 0;

	if (!encoded_buf)
		return NULL;

	av_init_packet(&pkt);

	pkt.size = encoded_len;
	if (encoded_len == 0)
		return NULL;

	pkt.data = encoded_buf;

	len = avcodec_decode_video2(_avcodec_context, _avframe, &got_frame, &pkt);
	if (len < 0)
	{
		simplelog("[Error] Failed Decode packet.");
		return NULL;
	}

	if (got_frame)
	{
		DecodedFrame* decoded = new DecodedFrame();
		if (decoded)
		{
			decoded->setAllocator(m_pPipeline);

			decoded->timestamp = _avframe->pts;
			if (!decoded->setYuvData(_avframe->data, _avframe->linesize, _avframe->width, _avframe->height))
			{
				simplelog("[Error] Decoded Frame Alloc Error[1].");
				POSAFE_DELETE(decoded);
			}
		}
		else
		{
			simplelog("[Error] Decoded Frame Alloc Error[2].");
			POSAFE_DELETE(decoded);
		}

		encoded_buf += len;
		encoded_len -= len;

		return decoded;
	}
	encoded_buf += len;
	encoded_len -= len;
#endif
	return NULL;
}
/*
* decode
*	jpeg자료를 디코딩하여 화상자료를 얻는다.
* @parameters
*	[IN]	jpeg_data	: jpeg자료
*	[IN]	jpeg_size	: jpeg자료의 바이트수
*	[OUT]	raw_data	: 출력자료. 화상자료를 담은 기억공간.
						  null이 될수 없다.
*	[IN]	raw_size	: raw_data의 크기.
*	[OUT]	out_size	: decoding한 화상의 크기
						  raw_size는 최소한 out_size여야 한다.
						  만일 이보다 작은 경우는 실패한다.
*/
DecodedFrame* CBaseDecoder::decodeGstreamer(u8*& jpeg_data, int& jpeg_size)
{
#ifdef POR_SUPPORT_GSTREAMER
	/* input compressed image data. */
	if (!pushJpeg((const char*)jpeg_data, jpeg_size))
		return NULL;

	/* set jpeg_size to 0. for telling what jpeg_data was decoded. */
	jpeg_size = 0;

	DecodedFrame* decoded = new DecodedFrame();
	if (decoded)
	{
		decoded->setAllocator(m_pPipeline);

		/* get decompressed image data. */
		if (pullRaw(decoded, NULL))
			return decoded;

		POSAFE_DELETE(decoded);
	}
#endif
	return NULL;
}
/* _pushBuffer */
bool CBaseDecoder::pushJpeg(const char* data, int size)
{
#ifdef POR_SUPPORT_GSTREAMER
	GstBuffer *buffer = NULL;
	GstFlowReturn ret = GST_FLOW_OK;

	/* acquire buffer */
	ret = gst_buffer_pool_acquire_buffer(_buffer_pool, &buffer, NULL);
	if (ret != GST_FLOW_OK || !buffer)
	{
		simplelog("gst_buffer_pool_acquire_buffer error.");
		return false;
	}

	/* writable buffer map */
	GstMapInfo buffer_map_info;
	if (gst_buffer_map(buffer, &buffer_map_info, GST_MAP_WRITE) == FALSE)
	{
		simplelog("gst_buffer_map error.");

		gst_buffer_unref(buffer);	// gst_buffer_pool_release_buffer is automatically called.
		return false;
	}

	/* copy data */
	//if (gst_memory_lock(buffer_map_info.memory, GST_LOCK_FLAG_WRITE))
	{
		memcpy(buffer_map_info.data, data, std::min(size, (int)buffer_map_info.maxsize));
		//gst_memory_unlock(buffer_map_info.memory, GST_LOCK_FLAG_WRITE);
	}

	/* send jpeg data to gstreamer decoder pipeline */
	g_signal_emit_by_name(_source, "push-buffer", buffer, &ret);

	/* release buffer */
	gst_buffer_unmap(buffer, &buffer_map_info);
	gst_buffer_unref(buffer);	// gst_buffer_pool_release_buffer is automatically called.

	if (ret != GST_FLOW_OK)
	{
		simplelog("emit push-buffer signal error.");
		return false;
	}
#endif
	return true;
}

/*
* _pullRaw
*	gstreamer의 디코딩파이프라인에서 출구sink에서 디코딩자료를 얻어서 buf에 넣는다.
* @return
	디코딩자료가 없거나 buf_size가 raw_size보다 작은경우에 false를
	buf에 디코딩자료를 넣었으면 true를 되돌린다.
* @parameters
*	[out]	buf		: 출력자료를 저장할 기억공간
					  만일 buf_size가 -1이면 buf에 raw_size만한 기억공간을 할당한다.
					  아니면 복사만을 진행한다.
					  할당된 기억기는 사용후 해방해주어야 한다.
*	[in]	buf_size: buf의 크기이다.
					  만일 -1이면 buf는 할당되지 않은 기억기를 의미하기때문에
					  이 함수에서 기억공간을 할당한다.
*	[out]	raw_size: 디코딩자료의 크기를 되돌린다.
					  null이면 디코딩자료으 크기를 넣지 않는다.
*/
bool CBaseDecoder::pullRaw(DecodedFrame* decoded_frame, int* raw_size)
{
#ifdef POR_SUPPORT_GSTREAMER
	/* Validate input parameters */
	if (!decoded_frame)
	{
		simplelog("_pullRaw input parameter error: buf can't be null.");
		return false;
	}

	/* Will block until sample is ready. In our case "sample" is decoded raw image data. */
	GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(_sink));

	if (sample == NULL)
	{
		simplelog("gst_app_sink_pull_sample returned null");
		return false;
	}

	/* Check properties of decoded raw image data. */
	/* Check to verify image decoding. */
	if (_is_enabled_checking_caps)
	{
		GstCaps *decoded = NULL;
		GstCaps *expected = NULL;
		char expected_caps_str[100];
		sprintf_s(expected_caps_str, sizeof(expected_caps_str),
			"video/x-raw, width=%d, height=%d",
			_width, _height);

		decoded = gst_sample_get_caps(sample);
		expected = gst_caps_from_string(expected_caps_str);

		if (!gst_caps_is_always_compatible(decoded, expected))
		{
			gst_caps_unref(expected);
			gst_sample_unref(sample);
			return false;
		}
		gst_caps_unref(expected);
	}

	/* Actual decompressed image is stored inside GstSample. */
	/* So we can get raw data from guffer of sample. */
	GstBuffer* buffer = gst_sample_get_buffer(sample);
	GstMapInfo map_info;

	if (!buffer)
	{
		simplelog("gst_sample_get_buffer error.");
		gst_sample_unref(sample);
		return false;
	}
	gst_buffer_map(buffer, &map_info, GST_MAP_READ);
	/* set raw data size */
	if (raw_size)
	{
		*raw_size = map_info.size;
	}

	/* copy data */
	//if (gst_memory_lock(map_info.memory, GST_LOCK_FLAG_READ))
	{
		decoded_frame->setGrayData(map_info.data, map_info.size, _width, _height);
		//gst_memory_unlock(map_info.memory, GST_LOCK_FLAG_READ);
	}

	/* release */
	gst_buffer_unmap(buffer, &map_info);
	gst_sample_unref(sample);

#endif
	return true;
}

//////////////////////////////////////////////////////////////////////////
// RawFrame
//////////////////////////////////////////////////////////////////////////
RawFrame::RawFrame() : data(NULL), width(0), height(0), timestamp(0)
{
	
}

RawFrame::~RawFrame()
{
	if (!m_pAllocator.isNull())
	{
		m_pAllocator->freeRawFrame(data, false);
		data = NULL;
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);
	}
}

void RawFrame::setAllocator(PipelineRef pAllocator)
{
	m_pAllocator = pAllocator;
}

bool RawFrame::alloc(int len)
{
	if (!m_pAllocator.isNull())
	{
		m_pAllocator->freeRawFrame(data);

		data = m_pAllocator->allocRawFrame(len);
		length = len;
	}
	else
	{
		POSAFE_DELETE_ARRAY(data);

		data = new u8[len];
		length = len;
	}

	return data != NULL;
}

#include "base_encoder.h"
#include "base.h"

#if defined(POR_WITH_STREAM)

#ifdef POR_SUPPORT_FFMPEG
#if defined(POR_WINDOWS)
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "swscale.lib")
#endif

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include <libswscale/swscale.h>
}
#endif

#ifdef POR_SUPPORT_GSTREAMER
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#endif
#endif 

//////////////////////////////////////////////////////////////////////////
ImgQueue::ImgQueue()
{
	ERR_PREPARE(0);
	
	w = h = channel = 0;
	mw = mh = mchannel = 0;
	count = 0;
	read_pos = 0;
	write_pos = 0;
	is_inited = false;

	memset(time_stamp, 0, sizeof(i64)*STREAM_QUEUELEN);
	memset(is_calibed, 0, sizeof(bool)*STREAM_QUEUELEN);
	memset(pixel_per_mm, 0, sizeof(f32)*STREAM_QUEUELEN);
	memset(img_ptr, 0, sizeof(u8*)*STREAM_QUEUELEN);
}

ImgQueue::~ImgQueue()
{
	exitInstance();
}

bool ImgQueue::initInstance(i32 nw, i32 nh, i32 nchannel, i32 queue_count)
{
	exitInstance();

	if (CPOBase::checkRange(queue_count, STREAM_QUEUELEN))
	{
		singlelog_lv0("ImageQueue for CaptureStream InitInstance");

		w = nw; h = nh; channel = nchannel;
		mw = nw; mh = nh; mchannel = nchannel;
		count = queue_count;
		read_pos = 0;
		write_pos = 0;
		is_inited = true;
		initBuffer();
	}
	return true;
}

bool ImgQueue::exitInstance()
{
	singlelog_lv0("ImageQueue for CaptureStream ExitInstance");

	if (is_inited)
	{
		is_inited = false;
		for (i32 i = 0; i < STREAM_QUEUELEN; i++)
		{
			POSAFE_DELETE_ARRAY(img_ptr[i]);
		}
	}
	return true;
}

void ImgQueue::reset()
{
	read_pos = write_pos = 0;
}

void ImgQueue::initBuffer()
{
	if (w*h*channel <= 0)
	{
		return;
	}

	i32 i, buffer_size = w*h*channel;
	for (i = 0; i < count; i++)
	{
		POSAFE_DELETE_ARRAY(img_ptr[i]);
		img_ptr[i] = po_new u8[buffer_size];
		memset(img_ptr[i], 0x00, buffer_size);
	}
}

void ImgQueue::changeQueueInfo(i32 nw, i32 nh, i32 nchannel)
{
	read_pos = write_pos = 0;
	w = nw; h = nh; channel = nchannel;

	if (mw*mh*mchannel < nw*nh*nchannel)
	{
		initBuffer();
		mw = nw; mh = nh; mchannel = nchannel;
	}
}

void ImgQueue::setInputFrame(ImageData* img_data_ptr)
{
	if (!is_inited || !img_data_ptr || !img_data_ptr->img_ptr)
	{
		return;
	}

	i32 iw, ih, ichannel;
	i32 npos = (write_pos + 1) % count;
	img_data_ptr->copyToImage(img_ptr[npos], iw, ih, ichannel);
	
	time_stamp[npos] = img_data_ptr->time_stamp;
	is_calibed[npos] = img_data_ptr->is_calibed;
	pixel_per_mm[npos] = img_data_ptr->pixel_per_mm;
	write_pos = npos;

	if (write_pos != read_pos)
	{
		ERR_UNOCCUR(0);
	}
	else
	{
		ERR_OCCUR(0, debug_log(QString("The encoding is delayed than camera preview [%1]").arg(_err_rep0)));
	}
}

bool ImgQueue::getOutputFrame(ImageData* img_data_ptr)
{
	if (!is_inited || img_data_ptr == NULL || read_pos == write_pos)
	{
		return false;
	}

	img_data_ptr->w = w;
	img_data_ptr->h = h;
	img_data_ptr->channel = channel;
	read_pos = (read_pos + 1) % count;
	printlog_lvs4(QString("Got EncodeQueue: %1:%2").arg(write_pos).arg(read_pos), LOG_SCOPE_ENCODE);

	img_data_ptr->img_ptr = img_ptr[read_pos];
	img_data_ptr->time_stamp = time_stamp[read_pos];
	img_data_ptr->is_calibed = is_calibed[read_pos];
	img_data_ptr->pixel_per_mm = pixel_per_mm[read_pos];
	img_data_ptr->is_external_alloc = true;
	return true;
}

//////////////////////////////////////////////////////////////////////////
#if defined(POR_WITH_STREAM)
CBaseEncoder::CBaseEncoder()
{
	m_encoder_mode = kPOEncoderNone;
	m_width = 0;
	m_height = 0;
	m_channel = 0;
	m_bit_rate = 0;
	m_frame_rate = 0;
	m_video_id = 0;
	m_frame_id = 0;

#ifdef POR_SUPPORT_FFMPEG
	m_av_codec_ptr = NULL;
	m_av_frame_ptr = NULL;
	m_av_context_ptr = NULL;
	m_sws_context_ptr = NULL;
#endif

#ifdef POR_SUPPORT_GSTREAMER
	m_pipeline_ptr = NULL;
	m_source_ptr = NULL;
	m_encoder_ptr = NULL;
	m_sink_ptr = NULL;
	m_buffer_pool_ptr = NULL;
#endif

	m_is_inited = false;
	m_use_encoder = false;
	m_is_thread_cancel = false;
}

CBaseEncoder::~CBaseEncoder()
{
	exitInstance();
}

bool CBaseEncoder::initInstance(i32 nw, i32 nh, i32 nchannel)
{
	if (!m_is_inited)
	{
		singlelog_lv0("Capture Encoder InitInstance");

		m_is_inited = true;
		m_use_encoder = false;
		m_is_thread_cancel = false;
		return m_img_queue.initInstance(nw, nh, nchannel, STREAM_QUEUELEN);
	}
	return true;
}

bool CBaseEncoder::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv0("Capture Encoder ExitInstance");

		m_is_thread_cancel = true;
		QThreadStop();

		releaseEncoder();
		m_img_queue.exitInstance();
		m_is_inited = false;

#if defined(POR_SUPPORT_FFMPEG)
		exitEncoderFFMpegMJpeg();
#elif defined(POR_SUPPORT_GSTREAMER)
		exitEncoderGStreamerMJpeg();
#endif
	}
	return true;
}

bool CBaseEncoder::acquireEncoder(i32 encoder, i32 w, i32 h, i32 channel, i32 frate, i32 brate, i32 vid)
{
	singlelog_lvs2(QString("Acquire Encoder: encoder:%1, w:%2, h:%3, ch:%4, frate:%5, brate:%6, vid:%7")
					.arg(encoder).arg(w).arg(h).arg(channel).arg(frate).arg(brate).arg(vid), LOG_SCOPE_ENCODE);

	if (m_encoder_mode != encoder ||
		m_width != w || m_height != h || m_channel != channel ||
		m_bit_rate != brate || m_frame_rate != frate || m_video_id != vid)
	{
		QMutexLocker p(&m_codec_mutex);
		QMutexLocker q(&m_queue_mutex);

		m_encoder_mode = encoder;
		m_width = w;
		m_height = h;
		m_channel = channel;
		m_bit_rate = brate;
		m_frame_rate = frate;
		m_video_id = vid;

		m_img_queue.changeQueueInfo(m_width, m_height, m_channel);

		switch (encoder)
		{
			case kPOEncoderFFMpegMJpeg:
			case kPOEncoderFFMpegH264:
			{
#if defined(POR_SUPPORT_FFMPEG)
				if (!initEncoderFFMpegMJpeg(m_width, m_height, m_channel, m_frame_rate, m_bit_rate))
				{
					return false;
				}
#else
				return false;
#endif
				break;
			}
			case kPOEncoderGStreamerMJpeg:
			case kPOEncoderGStreamerH264:
			{
#if defined(POR_SUPPORT_GSTREAMER)
				if (!initEncoderGStreamerMJpeg(m_width, m_height, m_channel, m_frame_rate, m_bit_rate))
				{
					return false;
				}
#else
				return false;
#endif
				break;
			}
			case kPOEncoderNetworkRaw:
			case kPOEncoderIPCRaw:
			{
				break;
			}
			default:
			{
				return false;
			}
		}

		m_frame_id = 0;
		m_yuv_frame_size = (3 * w*h) >> 1;
		m_use_encoder = true;
		m_is_inited = true;
		QThreadStart();
	}
	return true;
}

void CBaseEncoder::releaseEncoder()
{
	if (!m_is_inited)
	{
		return;
	}

	printlog_lvs3("Release Encoder", LOG_SCOPE_ENCODE);

	{
		QMutexLocker l(&m_queue_mutex);
		m_encoder_mode = kPOEncoderNone;
		m_width = 0;
		m_height = 0;
		m_channel = 0;
		m_frame_rate = 0;
		m_bit_rate = 0;
		m_video_id = -1;
		m_use_encoder = false;
		m_img_queue.reset();
	}
}

bool CBaseEncoder::initEncoderFFMpegMJpeg(i32 w, i32 h, i32 channel, i32 frate, i32 brate)
{
	if (m_encoder_mode != kPOEncoderFFMpegMJpeg)
	{
		return true;
	}
	exitEncoderFFMpegMJpeg();

#ifdef POR_SUPPORT_FFMPEG
	if (m_av_codec_ptr || m_av_context_ptr || m_av_frame_ptr || m_sws_context_ptr)
	{
		return false;
	}

	AVCodecID codec_id = AV_CODEC_ID_MJPEG;
	m_av_codec_ptr = avcodec_find_encoder(codec_id);

	if (!m_av_codec_ptr)
	{
		printlog_lvs2(QString("AVCodecFind Failed, codec:%1").arg(codec_id), LOG_SCOPE_ENCODE);
		return false;
	}

	m_av_context_ptr = avcodec_alloc_context3(m_av_codec_ptr);
	if (!m_av_context_ptr)
	{
		printlog_lvs2(QString("AVCodecAllocateContext Failed, codec:%1").arg(codec_id), LOG_SCOPE_ENCODE);
		return false;
	}

	/* put sample parameters */
	m_av_context_ptr->bit_rate = brate;

	/* resolution must be a multiple of two */
	m_av_context_ptr->width = w;
	m_av_context_ptr->height = h;

	/* frames per second */
	m_av_context_ptr->time_base.num = 1;
	m_av_context_ptr->time_base.den = frate;

	/* emit one intra frame every ten frames
	* check frame pict_type before passing frame
	* to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
	* then gop_size is ignored and the output of encoder
	* will always be I frame irrespective to gop_size
	*/
	m_av_context_ptr->gop_size = 0;
	m_av_context_ptr->max_b_frames = 0;
	m_av_context_ptr->pix_fmt = AV_PIX_FMT_YUVJ420P;
	/*
	if (codec_id == AV_CODEC_ID_H264)
	{
		av_opt_set(m_pCodecContext->priv_data, "preset", "ultrafast", 0);
		av_opt_set(m_pCodecContext->priv_data, "pass", "1", 0);
		av_opt_set(m_pCodecContext, "tune", "zerolatency", 0);
		av_opt_set(m_pCodecContext, "tune", "fastdecode", 0);
		av_opt_set(m_pCodecContext, "threads", "1", 0);
	}
	*/
	/* open it */
	if (avcodec_open2(m_av_context_ptr, m_av_codec_ptr, NULL) < 0)
	{
		printlog_lvs2("The avcodec open is failed", LOG_SCOPE_ENCODE);
		return false;
	}

	m_av_frame_ptr = av_frame_alloc();
	if (!m_av_frame_ptr)
	{
		return false;
	}

	m_av_frame_ptr->width = m_av_context_ptr->width;
	m_av_frame_ptr->height = m_av_context_ptr->height;
	m_av_frame_ptr->format = m_av_context_ptr->pix_fmt;

	i32 ret = av_image_alloc(m_av_frame_ptr->data, m_av_frame_ptr->linesize, m_av_context_ptr->width, m_av_context_ptr->height, m_av_context_ptr->pix_fmt, 8);
	if (ret < 0)
	{
		return false;
	}

	i32 pix_mode = AV_PIX_FMT_YUV420P;
	m_sws_context_ptr = sws_getCachedContext(m_sws_context_ptr,
								w, h, (channel == kPOGrayChannels) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24,
								w, h, AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
	return (m_sws_context_ptr != NULL);
#else
	return false;
#endif
}

void CBaseEncoder::exitEncoderFFMpegMJpeg()
{
	if (m_encoder_mode != kPOEncoderFFMpegMJpeg)
	{
		return;
	}

#ifdef POR_SUPPORT_FFMPEG
	if (m_av_context_ptr)
	{
		avcodec_close(m_av_context_ptr);
		av_free(m_av_context_ptr);
		sws_freeContext(m_sws_context_ptr);
		m_av_codec_ptr = NULL;
		m_av_context_ptr = NULL;
		m_sws_context_ptr = NULL;
	}
	if (m_av_frame_ptr)
	{
		if (m_av_frame_ptr->data)
		{
			av_freep(&m_av_frame_ptr->data[0]);
		}
		av_frame_free(&m_av_frame_ptr);
		m_av_frame_ptr = NULL;
	}
#endif
}

bool CBaseEncoder::initEncoderGStreamerMJpeg(i32 w, i32 h, i32 channel, i32 frate, i32 bit_rate)
{
	if (m_encoder_mode != kPOEncoderGStreamerMJpeg)
	{
		return true;
	}
	exitEncoderGStreamerMJpeg();

#ifdef POR_SUPPORT_GSTREAMER
	/* Create empty pipeline. */
	m_pipeline_ptr = gst_pipeline_new("test-pipeline");
	if (!m_pipeline_ptr)
	{
		printlog_lvs2("Failed to create test-pipeline element.", LOG_SCOPE_ENCODE);
		return false;
	}

	/* Create pipeline elements. Source, Encoder, Sink.*/
	m_source_ptr = gst_element_factory_make("appsrc", "source");
	if (!m_source_ptr)
	{
		printlog_lvs2("Failed to create fakesrc element.", LOG_SCOPE_ENCODE);
		return false;
	}
	m_encoder_ptr = gst_element_factory_make("jpegenc", "encoder");
	if (!m_encoder_ptr)
	{
		printlog_lvs2("Failed to create jpegenc element.", LOG_SCOPE_ENCODE);
		return false;
	}
	m_sink_ptr = gst_element_factory_make("appsink", "sink");
	if (!m_sink_ptr)
	{
		printlog_lvs2("Failed to create sink element.", LOG_SCOPE_ENCODE);
		return false;
	}

	/* configure jpeg encoder params. */
	/* appsrc should be linked to jpegenc with these caps otherwise jpegenc does not know size of incoming buffer */
	GstCaps* jpeg_enc_caps = NULL;
	switch (channel)
	{
		case kPOGrayChannels:
		{
			jpeg_enc_caps = gst_caps_new_simple("video/x-raw",
									"format", G_TYPE_STRING, "GRAY8",
									"width", G_TYPE_INT, w,
									"height", G_TYPE_INT, h,
									"framerate", GST_TYPE_FRACTION, 0, 1,
									NULL);
			break;
		}
		case kPORGBChannels:
		{
			jpeg_enc_caps = gst_caps_new_simple("video/x-raw",
									"format", G_TYPE_STRING, "RGB",
									"width", G_TYPE_INT, w,
									"height", G_TYPE_INT, h,
									"framerate", GST_TYPE_FRACTION, 0, 1,
									NULL);
			break;
		}
	}
	if (!jpeg_enc_caps)
	{
		printlog_lvs2("Failed to create caps new sample.", LOG_SCOPE_ENCODE);
		return false;
	}

	/* blocksize is important for jpegenc to know how many data to expect from appsrc in a single frame, too */
	char block_size[64];
	po_sprintf(block_size, 64, "%d", w*h);
	g_object_set(m_source_ptr, "blocksize", block_size, NULL);

	/* jpeg encoding quality */
	g_object_set(G_OBJECT(m_encoder_ptr), "quality", bit_rate, NULL);

	/* Build the pipeline. */
	/* appsrc->jpegenc->appsink*/
	gst_bin_add_many(GST_BIN(m_pipeline_ptr), m_source_ptr, m_encoder_ptr, m_sink_ptr, NULL);
	if (!gst_element_link_filtered(m_source_ptr, m_encoder_ptr, jpeg_enc_caps))
	{
		printlog_lvs2("Failed to link the elements between Source and Encoder.", LOG_SCOPE_ENCODE);
		return false;
	}
	if (!gst_element_link(m_encoder_ptr, m_sink_ptr))
	{
		printlog_lvs2("Failed to link the elements between Encoder and Sink.", LOG_SCOPE_ENCODE);
		return false;
	}

	/* Stop playing the pipeline. */
	gst_element_set_state(m_pipeline_ptr, GST_STATE_PLAYING);

	/* Create buffer pool for encoding. */
	m_buffer_pool_ptr = gst_buffer_pool_new();
	guint size = 0;
	guint min_buffers = 0;
	guint max_buffers = 0;
	GstCaps *caps = NULL;
	GstStructure *pool_config = NULL;
	pool_config = gst_buffer_pool_get_config(m_buffer_pool_ptr);
	gst_buffer_pool_config_get_params(pool_config, &caps, &size, &min_buffers, &max_buffers);
	gst_buffer_pool_config_set_params(pool_config, caps, w*h * channel, STREAM_QUEUELEN, STREAM_QUEUELEN);

	/* must set config buffer_pool to use*/
	gst_buffer_pool_set_config(m_buffer_pool_ptr, pool_config);

	/* must set active after configured.
	if not, gst_buffer_pool_acquire will be failed. */
	gst_buffer_pool_set_active(m_buffer_pool_ptr, true);
#endif
	return true;
}

void CBaseEncoder::exitEncoderGStreamerMJpeg()
{
	if (m_encoder_mode != kPOEncoderGStreamerMJpeg)
	{
		return;
	}

#ifdef POR_SUPPORT_GSTREAMER
	if (m_pipeline_ptr)
	{
		/* Stop playing and release pipeline.
		* Note that _source, _encoder and _sink are released automatically
		* because child elements are released by parent element.
		*/
		gst_element_set_state(m_pipeline_ptr, GST_STATE_NULL);
		gst_object_unref(m_pipeline_ptr);
	}

	if (m_buffer_pool_ptr)
	{
		gst_object_unref(m_buffer_pool_ptr);
	}
#endif
}

void CBaseEncoder::run()
{
	singlelog_lv0("The BaseEncoder thread is");

	bool has_frame;
	ImageData tmp_img_data;
	ImageData img_data;

	while (!m_is_thread_cancel)
	{
		has_frame = false;
		{
			QMutexLocker l(&m_codec_mutex);
			{
				{
					QMutexLocker l(&m_queue_mutex);
					has_frame = m_img_queue.getOutputFrame(&img_data);
					if (has_frame)
					{
						printlog_lvs4(QString("Encode Frame(encoder:%1)").arg(m_encoder_mode), LOG_SCOPE_ENCODE);
						img_data.reserved = m_video_id;

						switch (m_encoder_mode)
						{
							case kPOEncoderFFMpegMJpeg:
							case kPOEncoderFFMpegH264:
							{
#if defined(POR_SUPPORT_FFMPEG)
								has_frame = setFrameDataFFMpeg(&img_data);
#else
								has_frame = false;
#endif
								break;
							}
							case kPOEncoderGStreamerMJpeg:
							case kPOEncoderGStreamerH264:
							{
#if defined(POR_SUPPORT_GSTREAMER)
								has_frame = setFrameDataGStreamer(&img_data);
#else
								has_frame = false;
#endif
								break;
							}
							case kPOEncoderIPCRaw:
							case kPOEncoderNetworkRaw:
							{
								has_frame = true;
								tmp_img_data.copyImage(img_data);
								tmp_img_data.reserved = m_video_id;
								break;
							}
							default:
							{
								has_frame = false;
								break;
							}
						}
					}
				}
			}

			if (has_frame)
			{
				switch (m_encoder_mode)
				{
					case kPOEncoderFFMpegMJpeg:
					case kPOEncoderFFMpegH264:
					{
#if defined(POR_SUPPORT_FFMPEG)
						encodeFrameFFMpeg(&img_data);
#endif
						break;
					}
					case kPOEncoderGStreamerMJpeg:
					case kPOEncoderGStreamerH264:
					{
#if defined(POR_SUPPORT_GSTREAMER)
						encodeFrameGStreamer(&img_data);
#endif
						break;
					}
					case kPOEncoderIPCRaw:
					case kPOEncoderNetworkRaw:
					{
						rawDataSend(&tmp_img_data);
						break;
					}
				}
			}
		}
		QThread::msleep(1);
	}
}

void CBaseEncoder::setImageToEncoder(ImageData* img_data_ptr)
{
	if (!m_is_inited || !m_use_encoder || !img_data_ptr || !img_data_ptr->isValid())
	{
		printlog_lvs3("SetImageToEncoder Failed.", LOG_SCOPE_ENCODE);
		return;
	}

	/* 비데오프레임의 크기가 달라지는 경우 엔코더를 재초기화한다. */
	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	i32 channel = img_data_ptr->channel;
	if (w != m_width || h != m_height || channel != m_channel)
	{
		printlog_lvs3(QString("Encoder Changed. prev(%1*%2*%3):cur(%4*%5*%6)")
						.arg(m_width).arg(m_height).arg(m_channel).arg(w).arg(h).arg(channel), LOG_SCOPE_ENCODE);

		acquireEncoder(m_encoder_mode, w, h, channel, m_frame_rate, m_bit_rate, m_video_id);
	}

	/* 프레임바퍼에 자료를 추가한다. */
	{
		QMutexLocker l(&m_queue_mutex);
		m_img_queue.setInputFrame(img_data_ptr);
	}
}

void CBaseEncoder::setImageToEncoder(ImageData* img_data_ptr, i32 cam_id)
{
	setImageToEncoder(img_data_ptr);
}

bool CBaseEncoder::setFrameDataFFMpeg(ImageData* img_data_ptr)
{
#ifdef POR_SUPPORT_FFMPEG
	if (!m_av_frame_ptr || !m_sws_context_ptr || !img_data_ptr || !img_data_ptr->isValid())
	{
		printlog_lvs3("FFMpeg SetFrameData Failed.", LOG_SCOPE_ENCODE);
		return false;
	}

	const i32 in_linesize[1] = { m_av_frame_ptr->width*img_data_ptr->channel };
	sws_scale(m_sws_context_ptr, (const uint8_t* const*)&(img_data_ptr->img_ptr),
			in_linesize, 0, m_av_frame_ptr->height, m_av_frame_ptr->data, m_av_frame_ptr->linesize);
	m_av_frame_ptr->pts = m_frame_id++;
#endif
	return true;
}

bool CBaseEncoder::encodeFrameFFMpeg(ImageData* img_data_ptr)
{
#ifdef POR_SUPPORT_FFMPEG
	AVPacket av_packet;
	av_init_packet(&av_packet);
	av_packet.data = NULL;
	av_packet.size = 0;

	i32 output = 0;
	i32 ret = avcodec_encode_video2(m_av_context_ptr, &av_packet, m_av_frame_ptr, &output);
	if (ret < 0 || output == 0 || av_packet.size <= 0)
	{
		if (ret < 0)
		{
			printlog_lvs4("FFMpeg EncodeFrame Process is failed.", LOG_SCOPE_ENCODE);
		}
		else
		{
			printlog_lvs4("FFMpeg EncodeFrame Process is invalid.", LOG_SCOPE_ENCODE);
		}
		return false;
	}

	void* send_data_ptr = onEncodedFrame(av_packet.data, av_packet.size, av_packet.pts, img_data_ptr);
	av_packet_unref(&av_packet);

	if (send_data_ptr)
	{
		onSendFrame(send_data_ptr);
	}
	else
	{
		printlog_lvs4("FFMpeg Encoded SendData is invalid.", LOG_SCOPE_ENCODE);
	}
#endif
	return true;
}

bool CBaseEncoder::setFrameDataGStreamer(ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return false;
	}

#ifdef POR_SUPPORT_GSTREAMER
	GstBuffer *buffer = NULL;
	GstFlowReturn ret = GST_FLOW_OK;

	gst_buffer_pool_acquire_buffer(m_buffer_pool_ptr, &buffer, NULL);

	GstMapInfo buffer_map_info;
	if (gst_buffer_map(buffer, &buffer_map_info, GST_MAP_WRITE) == FALSE)
	{
		printlog_lvs3("gst_buffer_map error.", LOG_SCOPE_ENCODE);
		gst_buffer_unref(buffer);
		return false;
	}

	//gst_memory_lock(buffer_map_info.memory, GST_LOCK_FLAG_WRITE);
	i32 src_stride = img_data_ptr->getImageStride();
	i32 dst_stride = CPOBase::round(src_stride, 4);
	i32 i, h = po::_min(img_data_ptr->h, (i32)buffer_map_info.maxsize/dst_stride);
	for (i = 0; i < h; i++)
	{
		u8* dst_buffer_ptr = buffer_map_info.data + dst_stride*i;
		u8* src_buffer_ptr = img_data_ptr->img_ptr + src_stride*i;
		memcpy(dst_buffer_ptr, src_buffer_ptr, src_stride);
	}
	//gst_memory_unlock(buffer_map_info.memory, GST_LOCK_FLAG_WRITE);

	gst_buffer_unmap(buffer, &buffer_map_info);

	g_signal_emit_by_name(m_source_ptr, "push-buffer", buffer, &ret);

	if (ret != GST_FLOW_OK)
	{
		printlog_lvs3("emit push-buffer signal error.", LOG_SCOPE_ENCODE);
		gst_buffer_unref(buffer);
		return false;
	}
	gst_buffer_unref(buffer);
#endif

	return true;
}

bool CBaseEncoder::encodeFrameGStreamer(ImageData* img_data_ptr)
{
#ifdef POR_SUPPORT_GSTREAMER
	// Will block until sample is ready. In our case "sample" is encoded picture.
	GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(m_sink_ptr));

	if (sample == NULL)
	{
		printlog_lvs3("gst_app_sink_pull_sample returned null", LOG_SCOPE_ENCODE);
		return false;
	}

	// Actual compressed image is stored inside GstSample.
	GstBuffer* gst_buffer_ptr = gst_sample_get_buffer(sample);
	GstMapInfo map_info;
	gst_buffer_map(gst_buffer_ptr, &map_info, GST_MAP_READ);

	void* send_data_ptr = onEncodedFrame(map_info.data, map_info.size, 0, img_data_ptr);

	gst_buffer_unmap(gst_buffer_ptr, &map_info);
	gst_sample_unref(sample);

	if (send_data_ptr)
	{
		onSendFrame(send_data_ptr);
	}
#endif
	return true;
}

void CBaseEncoder::rawDataSend(ImageData* img_data_ptr)
{
	if (!img_data_ptr || !img_data_ptr->isValid())
	{
		return;
	}

	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	i32 channel = img_data_ptr->channel;
	u8* img_ptr = img_data_ptr->img_ptr;
	void* send_data_ptr = onEncodedFrame(img_ptr, w*h*channel, img_data_ptr->time_stamp, img_data_ptr);
	if (send_data_ptr)
	{
		onSendFrame(send_data_ptr);
	}
}

void CBaseEncoder::registerAllCodec()
{
#if defined(POR_SUPPORT_FFMPEG)
	{
		avcodec_register_all();
	}
#elif defined(POR_SUPPORT_GSTREAMER)
	{
		gst_init(NULL, NULL);
	}
#endif
}
#endif

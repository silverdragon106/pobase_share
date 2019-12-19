#pragma once

#include "struct.h"
#include "logger/logger.h"

#include <QThread>
#include <QElapsedTimer>
#include <QMutex>

#define STREAM_FPS_LOW			7
#define STREAM_FPS_NORMAL		15
#define STREAM_FPS_FULL			25

#define STREAM_BITRATE_LOW		(20*1024*1024)
#define STREAM_BITRATE_MIDEUM	(40*1024*1024)
#define STREAM_BITRATE_HIGH		(60*1024*1024)
#define STREAM_BITRATE_SUPER	(80*1024*1024)

#define STREAM_QUALITY_LOW		50
#define STREAM_QUALITY_MEDIUM	75
#define STREAM_QUALITY_HIGH		85
#define STREAM_QUALITY_SUPER	97

#define STREAM_QUEUELEN			5

struct ImgQueue
{
	ERR_DEFINE(0)

	i32						w;
	i32						h;
	i32						channel;
	i32						mw;
	i32						mh;
	i32						mchannel;
	i32						count;
	std::atomic<i32>		read_pos;
	std::atomic<i32>		write_pos;
	i64						time_stamp[STREAM_QUEUELEN];
	bool					is_calibed[STREAM_QUEUELEN];
	f32						pixel_per_mm[STREAM_QUEUELEN];
	u8*						img_ptr[STREAM_QUEUELEN];

	bool					is_inited;

public:
	ImgQueue();
	~ImgQueue();

	bool					initInstance(i32 nw, i32 nh, i32 nchannel, i32 count);
	bool					exitInstance();
	void					initBuffer();
	void					changeQueueInfo(i32 nw, i32 nh, i32 nchannel);
	void					reset();

	void					setInputFrame(ImageData* img_data_ptr);
	bool					getOutputFrame(ImageData* img_data_ptr);
};

#ifdef POR_SUPPORT_FFMPEG
struct AVCodec;
struct AVFrame;
struct AVCodecContext;
struct SwsContext;
#endif

#if defined(POR_WITH_STREAM) && defined(POR_SUPPORT_GSTREAMER)
#include <gst/gst.h>
#endif

class CBaseEncoder : public QThread, public CVirtualEncoder
{
	Q_OBJECT

#if defined(POR_WITH_STREAM)
public:
	CBaseEncoder();
	virtual ~CBaseEncoder();

public:
	bool					initInstance(i32 nw = 0, i32 nh = 0, i32 nchannel = 0);
	bool					exitInstance();

	bool					initEncoderFFMpegMJpeg(i32 w, i32 h, i32 channel, i32 frate, i32 brate);
	void					exitEncoderFFMpegMJpeg();
	bool					setFrameDataFFMpeg(ImageData* img_data_ptr);
	bool					encodeFrameFFMpeg(ImageData* img_data_ptr);

	bool					initEncoderGStreamerMJpeg(i32 w, i32 h, i32 channel, i32 frate, i32 brate);
	void					exitEncoderGStreamerMJpeg();
	bool					setFrameDataGStreamer(ImageData* img_data_ptr);
	bool					encodeFrameGStreamer(ImageData* img_data_ptr);
	void					rawDataSend(ImageData* img_data_ptr);

	virtual bool			acquireEncoder(i32 encoder, i32 w, i32 h, i32 channel, i32 frate, i32 brate, i32 vid);
	virtual void			releaseEncoder();
	virtual void			setImageToEncoder(ImageData* img_data_ptr);
	virtual void			setImageToEncoder(ImageData* img_data_ptr, i32 cam_id);

	virtual void*			onEncodedFrame(u8* buffer_ptr, i32 size, i64 pts, ImageData* img_data_ptr) = 0;
	virtual void			onSendFrame(void* send_void_ptr) = 0;

	static void				registerAllCodec();

protected:
	void					run() Q_DECL_OVERRIDE;

public:
	i32						m_encoder_mode;
	i32						m_width;
	i32						m_height;
	i32						m_channel;
	i32						m_frame_rate;
	i32						m_bit_rate;
	i32						m_yuv_frame_size;
	i32						m_video_id;
	i64						m_frame_id;

	bool					m_is_inited;
	bool					m_use_encoder;
	std::atomic<bool>		m_is_thread_cancel;
	ImgQueue				m_img_queue;

#ifdef POR_SUPPORT_FFMPEG
	AVCodec*				m_av_codec_ptr;
	AVFrame*				m_av_frame_ptr;
	AVCodecContext*			m_av_context_ptr;
	SwsContext*				m_sws_context_ptr;
#endif
#ifdef POR_SUPPORT_GSTREAMER
	GstElement*				m_pipeline_ptr;			/* encoding pipeline. test-pipeline */
	GstElement*				m_source_ptr;			/* pipeline source element. appsrc */
	GstElement*				m_encoder_ptr;			/* encoder element. jpegenc */
	GstElement*				m_sink_ptr;				/* pipeline sink element. appsink */
	GstBufferPool*			m_buffer_pool_ptr;		/* buffer pool */
#endif

	QMutex					m_queue_mutex;
	QMutex					m_codec_mutex;
#endif
};

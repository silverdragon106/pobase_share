#pragma once

#include <QtCore>
#include <QThread>
#include "pipeline.h"
#include "types.h"
/*
* RawDataFormat
* @brief
*	화상자료 형식을 정의한다.
*/
enum RawDataFormat
{
	kRawDataGray8,
	kRawDataRGB
};

struct RawFrame
{
	RawFrame();
	~RawFrame();

	void		setAllocator(PipelineRef pAllocator);
	bool		alloc(int len);

	int			width;
	int			height;
	i64			timestamp;
	int			length;
	int			bpp;
	u8*			data;

	PipelineRef	m_pAllocator;
};

#ifdef POR_SUPPORT_FFMPEG
struct AVCodec;
struct AVFrame;
struct AVCodecContext;
#endif

#ifdef POR_SUPPORT_GSTREAMER
#include <gst/gst.h>
#endif

//////////////////////////////////////////////////////////////////////////
// Encoded Frame
//////////////////////////////////////////////////////////////////////////
struct EncodedFrame
{
	EncodedFrame();
	~EncodedFrame();

	bool		alloc(int len);
	void		setAllocator(PipelineRef pPipeline);

	int			width;
	int			height;
	int			length;
	int			bpp;
	i64			timestamp;
	u8*			data;

	PipelineRef	m_pAllocator;
};
//////////////////////////////////////////////////////////////////////////
// DecodedFrame
//////////////////////////////////////////////////////////////////////////
#define DECODED_CHANNEL_COUNT 3
struct DecodedFrame
{
	DecodedFrame();
	~DecodedFrame();

	bool		setYuvData(u8* decoded[DECODED_CHANNEL_COUNT], int linesize[DECODED_CHANNEL_COUNT], int width, int height);
	bool		setGrayData(u8* decoded, int decoded_len, int width, int height);
	void		setAllocator(PipelineRef pAllocator);

	int			width;
	int			height;
	//int			stride;		// not used on this version.
	int			length;
	int			bpp;
	i64			timestamp;
	u8*			data;

private:
	PipelineRef	m_pAllocator;
};

//////////////////////////////////////////////////////////////////////////
// CBaseDecoder
// *ffmpeg base decoder.
//////////////////////////////////////////////////////////////////////////
class CBaseDecoder: public QThread
{
	static const int kMinBufferPoolCount = 1;
	static const int kMaxBufferPoolCount = 5;
	Q_OBJECT
public:
	CBaseDecoder(int decoder_index, int decoder_type);
	virtual ~CBaseDecoder();

	bool			initInstance();
	void			exitInstance();
	static void		registerCodecs();

	bool			isRawMode();
	void			start();
	void			stop();

	PipelineRef		getPipeline();
	/* update decoder parameters. if changed param, then setParams. if not, then nothing to do.*/
	bool			updateParams(int width, int height, RawDataFormat format);
	/* set decoder parameters */
	bool			setParams(int width, int height, RawDataFormat format);
	/* push encoded data to decode thread in smart camera. */
	bool			pushEncodedFrameDataSC(u8* pData, int nSize);
	/* push encoded data to decode thread in mold vision. */
	bool			pushEncodedFrameDataMV(u8* pData, int nSize);

	bool			pushEncodedFrame(u8* pbuffer, int datalen, int width, int height, int bpp, i64 timestamp);
	bool			pushRawFrame(u8* pbuffer, int datalen, int width, int height, int bpp, i64 timestamp);
	/* decoded call back */
#ifndef USE_STREAMER
	/* theStreamer->animate에서 프레임 notify를 진행하기 때문에 아래의 함수는 필요가 없다. */
	virtual void	onDecodedFrame(DecodedFrame* pDecoded) = 0;
#endif
protected:
	bool			createDecoder();
	void			releaseDecoder();
	bool			createPipeline();
	void			releasePipeline();
	DecodedFrame*	decode(u8*& encoded_buf, int& encoded_len);

	bool			createDecoderFFmpeg();
	void			releaseDecoderFFmpeg();

	bool			createDecoderGstreamer();
	void			releaseDecoderGstreamer();

	DecodedFrame*	decodeFFmpeg(u8*& encoded_buf, int& encoded_len);
	DecodedFrame*	decodeGstreamer(u8*& encoded_buf, int& encoded_len);

	/* used functions in gstreamer */
	bool			pushJpeg(const char* data, int size);
	bool			pullRaw(DecodedFrame* decoded_frame, int* raw_size);
	static void		initializeGstreamer();
protected:
	void			run() Q_DECL_OVERRIDE;
private:
	PipelineRef		m_pPipeline;
	int				_decoder_index;
	int				_decoder_type;

	int				_width;
	int				_height;
	RawDataFormat	_format;
	bool			m_bAbort;
#ifdef POR_SUPPORT_FFMPEG
	AVCodec*		_avcodec;
	AVFrame*		_avframe;
	AVCodecContext*	_avcodec_context;
#endif

#ifdef POR_SUPPORT_GSTREAMER
	GstElement*		_pipeline;			/* decoding pipeline. test-pipeline */
	GstElement*		_source;			/* pipeline source element. appsrc */
	GstElement*		_decoder;			/* decoder element. jpegdec */
	GstElement*		_sink;				/* pipeline sink element. appsink */
	GstBufferPool*	_buffer_pool;		/* buffer pool */
	bool			_is_enabled_checking_caps;	/* flag whether checking to validate decoded frame data with decoder configuration. */
	static bool		_is_gst_inited;
#endif

	
};

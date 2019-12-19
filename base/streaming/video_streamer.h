#pragma once

/*
	HeartBeat udp packet receiver.
	This class receives heartbeat packet from heartbeat server every 3 seconds.
	And unpack heartbeat udp packet to ServerInfo.
*/
#include <QObject>
#include <QString>
#include "types.h"
#include "base_decoder.h"
struct Packet;
/**
* VideoStreamer
* @brief
*
* @code
* @endcode
**/
class VideoStreamer : public QObject
{
	Q_OBJECT
public:
	VideoStreamer();
	~VideoStreamer();

	/* Create pipelines. */
	bool			prepare(int nCameras, int decoder_type);
	/* start pipelines */
	void			start();
	/* stop pipelines */
	void			stop();

	/* synchronize frame timestamp */
	void			animate(i64 timestamp);

	/* get pipeline */
	CBaseDecoder*	getDecoder(int vid);

	/* process tcp Image Packet */
	void			onReceive(Packet* pak);
	/* process ipc image packet */
	void			onReceiveIPC(Packet* pak);

signals:
	/* thread-unsafe signal */
	void			newFrameReady(int /*vid*/, bool /*is_raw_mode?*/, void* /*frame_data*/);			/* new frame is ready. */
	void			timestampFrameReady(int /*vid*/, bool /*is_raw_mode?*/, void* /*frame_data*/);		/* used for ng image. */

private:
	QList<CBaseDecoder*>		m_decoders;
	bool						m_bStarted;
};

extern VideoStreamer* g_stream_mgr;

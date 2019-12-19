#pragma once
#include "struct.h"

struct Packet;

enum EmulatorTypes
{
	kEmulatorNone = 0,
	kEmulatorOne,
	kEmulatorLocal,

	kEmulatorTypeCount
};

class CEmuSamples
{
public:
	CEmuSamples();
	virtual ~CEmuSamples();

	virtual i32 			memSizeThumb();
	virtual i32				memWriteThumb(u8*& buffer_ptr, i32& buffer_size);

	virtual i32				getEmuSampleCams();

	virtual	i32				getEmuTriggerMode();
	virtual i32				getEmuTriggerInterval();
	virtual void			getEmuMaxSampleSize(i32& max_width, i32& max_height);

	virtual bool			getEmuSmapleGrayImage(i32 cam_id, i32& index, bool force_snap,
									u8* gray_buffer_ptr, i32& capture_width, i32& capture_height);
	virtual bool			getEmuSmapleImage(i32 cam_id, i32& index, bool force_snap,
									u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel);
};

class COneEmuSamples : public CEmuSamples, public CLockGuard
{
public:
	COneEmuSamples(i32 max_width, i32 max_height);
	virtual ~COneEmuSamples();

	virtual i32				getEmuSampleCams();

	virtual	i32				getEmuTriggerMode();
	virtual i32				getEmuTriggerInterval();
	virtual void			getEmuMaxSampleSize(i32& max_width, i32& max_height);

	virtual bool			getEmuSmapleGrayImage(i32 cam_id, i32& index, bool force_snap,
									u8* gray_buffer_ptr, i32& capture_width, i32& capture_height);
	virtual bool			getEmuSmapleImage(i32 cam_id, i32& index, bool force_snap,
									u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel);
	
	void					freeEmuSample();
	bool					setEmuSample(ImageData& img_data);

public:
	i32						m_max_width;
	i32						m_max_height;
	i32						m_base_width;
	i32						m_base_height;

	bool					m_emu_changed;
	ImageData				m_emu_image;
};

class CLocalEmuSamples : public CEmuSamples
{
public:
	CLocalEmuSamples();
	virtual ~CLocalEmuSamples();

	void					freeSamples();
	void					freeSampleThumbs();

	virtual i32				memSizeThumb();
	virtual i32				memWriteThumb(u8*& buffer_ptr, i32& buffer_size);
	
	virtual i32				getEmuSampleCams();
	virtual	i32				getEmuTriggerMode();
	virtual i32				getEmuTriggerInterval();
	virtual void			getEmuMaxSampleSize(i32& max_width, i32& max_heght);

	virtual bool			getEmuSmapleGrayImage(i32 cam, i32& index, bool force_snap,
									u8* gray_buffer_ptr, i32& capture_width, i32& capture_height);
	virtual bool			getEmuSmapleImage(i32 cam, i32& index, bool force_snap,
									u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel);

	bool					setSamples(Packet* packet_ptr, bool is_thumb_generated = false);
	bool					loadSamples(const QString& data_path, bool is_thumb_generated = false);
	bool					writeSamples(const QString& data_path);

	bool					setEmuTriggerInterval(i32 interval);
	bool					setEmuSelected(i32 cam_index, i32 selected, bool need_stop = true);
		
	i32						getEmuSampleCount(i32 cam_index);
	i32						getEmuSelected(i32 cam_index);
	void					getEmuSampleInfo(i32 cam, i32& w, i32& h);

	void					play();
	void					stop();
	void					nextFrame(i32 cam_index);

private:
	void					buildThumbSamples(ImageData** sample_ptr, i32 sample_cam_count, i32* sample_num_ptr,
									ImageData** sample_thumb_ptr);
	bool					loadSamples(const QString& source_path, const QString& sample_pattern,
									ImageData** sample_ptr, i32* sample_num_ptr, i32* sample_index_ptr, i32& sample_cam);
	void					freeSamples(ImageData** sample_ptr, ImageData** sample_thumb_ptr,
									i32* sample_num_ptr, i32* sample_index_ptr, i32& sample_cam);
	void					freeSampleThumbs(ImageData** sample_thumb_ptr, i32* sample_num_ptr, i32 sample_cam);

public:
	i32						m_sample_max_width;
	i32						m_sample_max_height;
	std::atomic<i32>		m_sample_interval;

	i32						m_sample_cam_count;
	i32						m_sample_count[PO_CAM_COUNT];
	i32						m_sample_index[PO_CAM_COUNT];
	ImageData*				m_sample_ptr[PO_CAM_COUNT];
	ImageData*				m_sample_thumb_ptr[PO_CAM_COUNT];

	std::atomic<bool>		m_is_inited;
	std::atomic<bool>		m_is_played;
	bool					m_is_thumb_generated;
};

#pragma once

#include "define.h"
#include "struct.h"
#include "struct/camera_setting.h"

struct BaseFrameInfo
{
	u8						reg_rotation;
	bool					is_flip_x;
	bool					is_flip_y;
	bool					is_invert;

	bool					has_frame;
	POMutex*				snap_mutex_ptr;

public:
	BaseFrameInfo()
	{
		memset(this, 0, sizeof(BaseFrameInfo));
	}
};

class CameraSetting;
class CameraTrigger;
class CBaseCamera
{
public:
	CBaseCamera() {};
	virtual ~CBaseCamera() {};

	virtual i32					initCamera() = 0;
	virtual i32					exitCamera() = 0;

	virtual i32					play() = 0;
	virtual i32					stop() = 0;

	virtual i32					snapToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info) = 0;
	virtual i32					snapToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info) = 0;

	virtual i32					snapManualToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info) = 0;
	virtual i32					snapManualToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info) = 0;

	virtual i32					snapSoftwareTrigger() = 0;

	virtual i32					setColorMode() = 0;
	virtual i32					setWhiteBalanceMode() = 0;
	virtual i32					setColorAWBOnce() = 0;
	virtual i32					setColorGain() = 0;

	virtual i32					setShutterMode() = 0;
	virtual i32					setShutterJitterTime() = 0;

	virtual i32					setCaptureInvert() = 0;
	virtual	i32					setCaptureFlip() = 0;
	virtual i32					setCaptureRotation() = 0;
	virtual i32					setCaptureRange() = 0;

	virtual i32					setGain() = 0;
	virtual i32					setAeGain() = 0;
	virtual i32					setExposureTimeMs() = 0;
	virtual i32					setAeExposureTimeMs() = 0;
	virtual i32					setAeBrightness() = 0;
	virtual i32					setAeWindow() = 0;
	virtual i32					setAeState(const i32 autoexp_mode) = 0;

	virtual i32					setCorrectionGamma() = 0;
	virtual i32					setCorrectionContrast() = 0;
	virtual i32					setCorrectionSaturation() = 0;
	virtual i32					setCorrectionSharpness() = 0;

	virtual i32					setTriggerMode(CameraTrigger& cam_trigger) = 0;
	virtual	i32					setStrobeEnabled() = 0;
	virtual	i32					setStrobeControl() = 0;
	virtual i32					setLightForTrigger(bool use_strobe) = 0;

	virtual i32					setNoiseReduction() = 0;
	virtual i32					setColorTemperature() = 0;

	virtual	i32					setEmuSampler(void* emu_sample_ptr) = 0;

	virtual bool				needTriggerScan() = 0;

	virtual	f32					getGain() = 0;
	virtual	f32					getExposureTimeMs() = 0;
	virtual i32					getCameraState(bool& autoexp_mode, f32& gain, f32& expsoure_time_ms) = 0;
	virtual i32					getCameraColorState(f32& rgain, f32& ggain, f32& bgain) = 0;
	virtual i32					getTriggerInterval() = 0;

	virtual i32					saveCameraParamToFile(const postring& cam_param_file) = 0;
	virtual i32					loadCameraParamToFile(const postring& cam_param_file) = 0;
};

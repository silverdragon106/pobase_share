#pragma once

#include "base_camera.h"

#if defined(POR_WITH_CAMERA)
#if defined(POR_SUPPORT_BASLER)

#include <pylon/PylonIncludes.h>

enum POBaslerCameraKind
{
	kPOCamBaslerNone = 0x0000,
	kPOCamBaslerGigE = 0x0001,
	kPOCamBaslerUsb = 0x0002
};

struct BaslerIPMatch;
class CBaslerPylonCamera : public CBaseCamera
{
public:
	CBaslerPylonCamera(CameraSetting* cam_param_ptr);
	virtual ~CBaslerPylonCamera();

	static i32					initSDK();
	static i32					exitSDK();
	static i32					getAvailableCamera(CameraDevice* cam_device_ptr, i32& cam_count);

	virtual	i32					initCamera();
	virtual	i32					exitCamera();
	i32							initGigECamera();
	i32							initUsbCamera();

	virtual	i32					play();
	virtual i32					stop();

	virtual i32					snapToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info);
	virtual i32					snapToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info);

	virtual i32					snapManualToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info);
	virtual i32					snapManualToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info);

	virtual i32					snapSoftwareTrigger();

	virtual i32					setColorMode();
	virtual i32					setWhiteBalanceMode();
	virtual i32					setColorAWBOnce();
	virtual i32					setColorGain();

	virtual i32					setShutterMode();
	virtual i32					setShutterJitterTime();

	virtual i32					setCaptureInvert();
	virtual	i32					setCaptureFlip();
	virtual i32					setCaptureRotation();
	virtual i32					setCaptureRange();
	virtual i32					setCaptureRangeInternal();

	virtual i32					setGain();
	virtual i32					setAeGain();
	virtual i32					setExposureTimeMs();
	virtual i32					setAeExposureTimeMs();
	virtual i32					setAeBrightness();
	virtual i32					setAeWindow();
	virtual i32					setAeState(const i32 autoexp_mode);

	virtual i32					setCorrectionGamma();
	virtual i32					setCorrectionContrast();
	virtual i32					setCorrectionSaturation();
	virtual i32					setCorrectionSharpness();

	virtual i32					setTriggerMode(CameraTrigger& cam_trigger);
	virtual	i32					setStrobeEnabled();
	virtual	i32					setStrobeControl();
	virtual i32					setLightForTrigger(bool use_strobe);

	virtual i32					setNoiseReduction();
	virtual i32					setColorTemperature();

	virtual	i32					setEmuSampler(void* emu_sample_ptr);

	virtual bool				needTriggerScan();

	virtual	f32					getGain();
	virtual	f32					getExposureTimeMs();
	virtual i32					getCameraState(bool& autoexp_mode, f32& gain, f32& expsoure_time_ms);
	virtual i32					getCameraColorState(f32& rgain, f32& ggain, f32& bgain);
	virtual i32					getTriggerInterval();

	virtual i32					saveCameraParamToFile(const postring& cam_param_file);
	virtual i32					loadCameraParamToFile(const postring& cam_param_file);

private:
	bool						changeBaslerTriggerMode(i32 trigger_mode, i32& prev_trigger_mode);
	bool						setBaslerContinuousTriggerMode(Pylon::CInstantCamera* device_ptr);
	bool						setBaslerSoftwareTriggerMode(Pylon::CInstantCamera* device_ptr);
	bool						setBaslerCameraTriggerMode(Pylon::CInstantCamera* device_ptr, CameraTrigger* cam_trigger_ptr);

	bool						adjust_value_i64(i64 cur_value, i64 min_value, i64 max_value, i64 inc_value, i64& new_value);
	bool						adjust_value_f64(f64 cur_value, f64 min_value, f64 max_value, f64& new_value);

public:
	CameraTrigger				m_dev_trigger;
	CameraSetting*				m_cam_param_ptr;
};

#endif
#endif

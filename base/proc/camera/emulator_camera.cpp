#include "emulator_camera.h"
#include "base.h"
#include "proc/image_proc.h"
#include "proc/emulator/emu_samples.h"

CEmulatorCamera::CEmulatorCamera(CameraSetting* cam_param_ptr)
{
	m_emu_sample_ptr = NULL;
	m_cam_param_ptr = cam_param_ptr;
	m_dev_trigger.init();
}

CEmulatorCamera::~CEmulatorCamera()
{
	m_emu_sample_ptr = NULL;
}

i32 CEmulatorCamera::initSDK()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::exitSDK()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::getAvailableCamera(CameraDevice* cam_device_ptr, i32& cam_count)
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::initCamera()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::exitCamera()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::play()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::stop()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::snapToBufferInternal(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
								bool force_snap, i32& capture_width, i32& capture_height, bool& has_frame)
{
	has_frame = false;
	capture_width = 0;
	capture_height = 0;

	if (!m_cam_param_ptr || !m_emu_sample_ptr)
	{
		return kPODevErrException;
	}

	i32 index = -1;
	i32 cam_id = m_cam_param_ptr->getCamID();

	if (gray_buffer_ptr)
	{
		if (m_emu_sample_ptr->getEmuSmapleGrayImage(cam_id, index, force_snap,
										gray_buffer_ptr, capture_width, capture_height))
		{
			has_frame = true;
			if (m_cam_param_ptr->m_cam_range.m_is_invert)
			{
				CImageProc::invertImage(gray_buffer_ptr, capture_width, capture_height);
			}
		}
	}
	else
	{
		i32 capture_channel;
		ImageData* img_data_ptr = (ImageData*)rgb_buffer_ptr;
		if (m_emu_sample_ptr->getEmuSmapleImage(cam_id, index, force_snap,
										img_data_ptr->img_ptr, capture_width, capture_height, capture_channel))
		{
			has_frame = true;
			img_data_ptr->w = capture_width;
			img_data_ptr->h = capture_height;
			img_data_ptr->channel = capture_channel;
			if (m_cam_param_ptr->m_cam_range.m_is_invert)
			{
				CImageProc::invertImage(img_data_ptr->img_ptr, capture_width, capture_height, capture_channel);
			}
		}
	}
	return kPODevErrNone;
}

i32 CEmulatorCamera::snapToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
								i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	return snapToBufferInternal(gray_buffer_ptr, rgb_buffer_ptr,
							false, capture_width, capture_height, frame_info.has_frame);
}

i32 CEmulatorCamera::snapToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	i32 capture_width;
	i32 capture_height;
	return snapToBufferInternal(NULL, (u8*)(&img_data),
							false, capture_width, capture_height, frame_info.has_frame);
}

i32 CEmulatorCamera::snapManualToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
								i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	return snapToBufferInternal(gray_buffer_ptr, rgb_buffer_ptr,
							true, capture_width, capture_height, frame_info.has_frame);
}

i32 CEmulatorCamera::snapManualToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	i32 capture_width;
	i32 capture_height;
	return snapToBufferInternal(NULL, (u8*)(&img_data),
							true, capture_width, capture_height, frame_info.has_frame);
}

i32 CEmulatorCamera::setGain()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setExposureTimeMs()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setAeGain()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setAeExposureTimeMs()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setAeBrightness()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setAeWindow()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setAeState(const i32 autoexp_mode)
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setTriggerMode(CameraTrigger& cam_trigger)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	CPOBase::memCopy(&m_dev_trigger, &cam_trigger);
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCaptureInvert()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCaptureFlip()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCaptureRotation()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCaptureRange()
{
	return kPODevErrNone;
}

bool CEmulatorCamera::needTriggerScan()
{
	return true;
}

i32 CEmulatorCamera::getTriggerInterval()
{
	if (m_emu_sample_ptr)
	{
		return m_emu_sample_ptr->getEmuTriggerInterval();
	}
	return 0;
}

f32 CEmulatorCamera::getGain()
{
	return 0;
}

f32 CEmulatorCamera::getExposureTimeMs()
{
	return 0;
}

i32 CEmulatorCamera::getCameraState(bool& autoexp_mode, f32& gain, f32& expsoure_time_ms)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	gain = 0;
	expsoure_time_ms = 0;
	autoexp_mode = false;
	return kPODevErrNone;
}

i32 CEmulatorCamera::getCameraColorState(f32& rgain, f32& ggain, f32& bgain)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	rgain = 0;
	ggain = 0;
	bgain = false;
	return kPODevErrNone;
}

i32 CEmulatorCamera::setEmuSampler(void* emu_sample_ptr)
{
	m_emu_sample_ptr = (CEmuSamples*)emu_sample_ptr;
	return kPODevErrNone;
}

i32 CEmulatorCamera::saveCameraParamToFile(const postring& cam_param_file)
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::loadCameraParamToFile(const postring& cam_param_file)
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::snapSoftwareTrigger()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setShutterJitterTime()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setColorMode()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setWhiteBalanceMode()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setColorAWBOnce()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setColorGain()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setShutterMode()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setStrobeEnabled()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setStrobeControl()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setLightForTrigger(bool use_strobe)
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCorrectionGamma()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCorrectionContrast()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCorrectionSaturation()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setCorrectionSharpness()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setNoiseReduction()
{
	return kPODevErrNone;
}

i32 CEmulatorCamera::setColorTemperature()
{
	return kPODevErrNone;
}
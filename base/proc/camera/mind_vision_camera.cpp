#include "mind_vision_camera.h"
#include "base.h"
#include "proc/image_proc.h"

#if defined(POR_WITH_CAMERA)
#if defined(POR_SUPPORT_MINDVISION)

#if defined(POR_WINDOWS)
#include <windows.h>
#if defined(POR_IA64)
#pragma comment(lib, "MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "MVCAMSDK.lib")
#endif
#endif

#include "CameraApi.h"

enum MVSdkSnapMode
{
	kMVTriggerModeNone = -1,
	kMVTriggerModeContinue = 0,
	kMVTriggerModeManual,
	kMVTriggerModeExternal
};

enum MVExtTrigSignal
{
	kMVExtTriggerNone = -1,
	kMVExtTriggerLeadingEdge = 0,
	kMVExtTriggerFallingEdge,
	kMVExtTriggerHighLevel,
	kMVExtTriggerLowLevel,
	kMVExtTriggerDoubleEdge,
};

//////////////////////////////////////////////////////////////////////////
i32 CMindVisionCamera::initSDK()
{
	if (CameraSdkInit(0) != CAMERA_STATUS_SUCCESS) //0: init for English ,else init for Chinese
	{
		printlog_lv1("MindVision SDKInit is Failed.");
		return kPODevErrConnect;
	}

	char mindvision_sdk_name[PO_MAXPATH];
	if (CameraSdkGetVersionString(mindvision_sdk_name) != CAMERA_STATUS_SUCCESS)
	{
		printlog_lv1("MindVision GetSDKVersion is Failed.");
		return kPODevErrConnect;
	}

	printlog_lv1(QString("MindVision CameraSDK is %1").arg(mindvision_sdk_name));
	return kPODevErrNone;
}

i32 CMindVisionCamera::exitSDK()
{
	return kPODevErrNone;
}

i32 CMindVisionCamera::getAvailableCamera(CameraDevice* cam_device_ptr, i32& cam_count)
{
	i32 i, count = PO_MAX_CAM_COUNT;
	i32 code = CAMERA_STATUS_SUCCESS;
	tSdkCameraDevInfo camera_devices[PO_MAX_CAM_COUNT];
	CameraDevice* tmp_device_ptr;

	code = CameraEnumerateDevice(camera_devices, &count);
	switch (code)
	{
		case CAMERA_STATUS_SUCCESS:
		case CAMERA_STATUS_NO_DEVICE_FOUND:
		case CAMERA_STATUS_NO_LOGIC_DEVICE_FOUND:
		{
			break;
		}
		default:
		{
			return kPODevErrException;
		}
	}

	if (count <= 0 || cam_count >= PO_MAX_CAM_COUNT)
	{
		return kPODevErrNone;
	}

	for (i = 0; i < count; i++)
	{
		tmp_device_ptr = cam_device_ptr + cam_count;
		tmp_device_ptr->m_cam_type = kPOCamMindVision;
		tmp_device_ptr->m_cam_name = getCameraPortName(camera_devices[i].acFriendlyName);
		tmp_device_ptr->m_cam_blob.initBuffer((u8*)(camera_devices + i), sizeof(tSdkCameraDevInfo));

		cam_count++;
		if (cam_count >= PO_MAX_CAM_COUNT)
		{
			break;
		}
	}
	return kPODevErrNone;
}

//////////////////////////////////////////////////////////////////////////
CMindVisionCamera::CMindVisionCamera(CameraSetting* cam_param_ptr)
{
	m_last_error_code = kPODevErrNone;
	m_cam_param_ptr = cam_param_ptr;
	m_dev_trigger.init();
}

CMindVisionCamera::~CMindVisionCamera()
{

}

postring CMindVisionCamera::getCameraPortName(const postring& cam_friendly_name)
{
	postring cam_name = cam_friendly_name.substr(cam_friendly_name.find('#') + 1);
	CPOBase::toLower(cam_name);
	return cam_name;
}

i32 CMindVisionCamera::initCamera()
{
	m_dev_trigger.init();
	m_last_error_code = kPODevErrNone;

	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	tSdkCameraCapbility	dev_capability;
	tSdkImageResolution resolution;

	i32 cam_handle = -1;
	i32 cam_id = m_cam_param_ptr->getCamID();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraState* cam_state_ptr = m_cam_param_ptr->getCameraState();

	tSdkCameraDevInfo* sdk_cam_info_ptr = (tSdkCameraDevInfo*)cam_info_ptr->m_cam_blob.getBuffer();
	if (!checkError(CameraInit(sdk_cam_info_ptr, -1, -1, &cam_handle)))
	{
		printlog_lvs2(QString("Camera%1 can't be Initialize").arg(cam_id), LOG_SCOPE_CAM);
		return getLastError();
	}
	m_cam_param_ptr->m_cam_handle = cam_handle;

	// read default some information of camera such as camera capability & exptime step and so on...
	if (!checkError(CameraGetCapability(cam_handle, &dev_capability)))
	{
		return getLastError();
	}

	// check current camera setting validation
	f64 dexp_stp;
	if (!checkError(CameraGetExposureLineTime(cam_handle, &dexp_stp)))
	{
		return false;
	}

	// set image size for preview and capture mode
	if (!checkError(CameraSetImageResolution(cam_handle, dev_capability.pImageSizeDesc)) ||
		!checkError(CameraGetImageResolution(cam_handle, &resolution)) ||
		!checkError(CameraSetResolutionForSnap(cam_handle, dev_capability.pImageSizeDesc)))
	{
		return false;
	}

	i32 width = resolution.iWidth;
	i32 height = resolution.iHeight;

	u32 cam_capability = kCamSupportGray | kCamSupportRange | 
					kCamSupportAutoStrobe | kCamSupportManualStrobe |
					kCamSupportGamma | kCamSupportContrast | kCamSupportSharpness | kCamSupportNoiseReduction;
	cam_capability |= dev_capability.sIspCapacity.bMonoSensor ? kCamSupportGray : kCamSupportColor;
	cam_capability |= dev_capability.sIspCapacity.bAutoWb ? kCamSupportAutoWb : kCamSupportNone;
	cam_capability |= dev_capability.sIspCapacity.bWbOnce ? kCamSupportAutoWbOnce : kCamSupportNone;
	cam_capability |= dev_capability.sIspCapacity.bAutoExposure ? kCamSupportAutoExp : kCamSupportNone;
	cam_capability |= dev_capability.sIspCapacity.bManualExposure ? kCamSupportManualExp : kCamSupportNone;

	if (CPOBase::bitCheck(cam_capability, kCamSupportColor))
	{
		cam_capability |= (kCamSupportManualWb | 
						kCamSupportRedGain | kCamSupportGreenGain | kCamSupportBlueGain);
	}

	u32 cap_trig_mask = 0;
	if (checkError(CameraGetExtTrigCapability(cam_handle, &cap_trig_mask)))
	{
#if defined(POR_WINDOWS)
		if (CPOBase::bitCheck(cap_trig_mask, EXT_TRIG_MASK_GRR_SHUTTER))
		{
			cam_capability |= kCamSupportShutter;
		}
#endif
	}

	if (CPOBase::bitCheck(cam_capability, kCamSupportAutoExp))
	{
		cam_capability |= kCamSupportAEWindow;
	}

	//set frame speed to maximum
	checkError(CameraSetFrameSpeed(cam_handle, dev_capability.iFrameSpeedDesc-1));

	//set strobe line
	if (CPOBase::bitCheck(cam_capability, kCamSupportAutoStrobe) ||
		CPOBase::bitCheck(cam_capability, kCamSupportManualStrobe))
	{
		checkError(CameraSetOutPutIOMode(cam_handle, 0, IOMODE_GP_OUTPUT));
	}
	
	//read camera infomation
	{
		anlock_guard_ptr(cam_info_ptr);
		cam_info_ptr->m_cam_name = sdk_cam_info_ptr->acFriendlyName;
		cam_info_ptr->m_cam_capability = cam_capability;
		cam_info_ptr->m_gain_min = dev_capability.sExposeDesc.uiAnalogGainMin;
		cam_info_ptr->m_gain_max = dev_capability.sExposeDesc.uiAnalogGainMax;
		cam_info_ptr->m_exposure_min = (f32)dev_capability.sExposeDesc.uiExposeTimeMin * dexp_stp / 1000;
		cam_info_ptr->m_exposure_max = (f32)dev_capability.sExposeDesc.uiExposeTimeMax * dexp_stp / 1000;
		cam_info_ptr->m_brightness_min = dev_capability.sExposeDesc.uiTargetMin;
		cam_info_ptr->m_brightness_max = dev_capability.sExposeDesc.uiTargetMax;

		cam_info_ptr->m_red_gain_min = (f32)dev_capability.sRgbGainRange.iRGainMin / 100;
		cam_info_ptr->m_red_gain_max = (f32)dev_capability.sRgbGainRange.iRGainMax / 100;
		cam_info_ptr->m_green_gain_min = (f32)dev_capability.sRgbGainRange.iGGainMin / 100;
		cam_info_ptr->m_green_gain_max = (f32)dev_capability.sRgbGainRange.iGGainMax / 100;
		cam_info_ptr->m_blue_gain_min = (f32)dev_capability.sRgbGainRange.iBGainMin / 100;
		cam_info_ptr->m_blue_gain_max = (f32)dev_capability.sRgbGainRange.iBGainMax / 100;

		cam_info_ptr->m_gamma_min = (f32)dev_capability.sGammaRange.iMin / 100.0f;
		cam_info_ptr->m_gamma_max = (f32)dev_capability.sGammaRange.iMax / 100.0f;
		cam_info_ptr->m_contrast_min = (f32)dev_capability.sContrastRange.iMin;
		cam_info_ptr->m_contrast_max = (f32)dev_capability.sContrastRange.iMax;
		cam_info_ptr->m_saturation_min = (f32)dev_capability.sSaturationRange.iMin;
		cam_info_ptr->m_saturation_max = (f32)dev_capability.sSaturationRange.iMax;
		cam_info_ptr->m_sharpness_min = (f32)dev_capability.sSharpnessRange.iMin;
		cam_info_ptr->m_sharpness_max = (f32)dev_capability.sSharpnessRange.iMax;

		cam_info_ptr->m_max_width = width;
		cam_info_ptr->m_max_height = height;
	}
	{
		anlock_guard_ptr(cam_state_ptr);
		cam_state_ptr->m_capture_width = width;
		cam_state_ptr->m_capture_height = height;
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::exitCamera()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	if (!checkError(CameraStop(cam_handle)))
	{
		return getLastError();
	}
	if (!checkError(CameraUnInit(cam_handle)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::play()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	if (!checkError(CameraPlay(cam_handle)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::stop()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	if (!checkError(CameraStop(cam_handle)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::snapToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
									i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	bool& is_flip_y = frame_info.is_flip_y;
	has_frame = false;

	capture_width = 0;
	capture_height = 0;
	u8* raw_buffer_ptr = NULL;

	if (!m_cam_param_ptr)
	{
		return kPODevErrNone;
	}

	tSdkFrameHead sdk_frame_info;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

#if defined(POR_WINDOWS)
	i32 error_code = checkErrorEx(CameraGetImageBufferPriority(cam_handle, &sdk_frame_info,
												&raw_buffer_ptr, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST));
#elif defined(POR_LINUX)
	i32 error_code = checkErrorEx(CameraGetImageBuffer(cam_handle, &sdk_frame_info, &raw_buffer_ptr, 1000));
#endif

	if (error_code == kPODevErrNone)
	{
		capture_width = sdk_frame_info.iWidth;
		capture_height = sdk_frame_info.iHeight;
		checkError(CameraImageProcess(cam_handle, raw_buffer_ptr, rgb_buffer_ptr, &sdk_frame_info));

		if (sdk_frame_info.uiMediaType == CAMERA_MEDIA_TYPE_MONO8)
		{
			has_frame = true;
			memcpy(gray_buffer_ptr, rgb_buffer_ptr, capture_width*capture_height);
#if defined(POR_WINDOWS)
			is_flip_y = true;
#endif
		}
		else
		{
		}
		checkError(CameraReleaseImageBuffer(cam_handle, raw_buffer_ptr));
	}
	else if (error_code == kPODevErrException)
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::snapToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	u8* raw_buffer_ptr = NULL;
	bool& has_frame = frame_info.has_frame;
	bool& is_flip_y = frame_info.is_flip_y;
	has_frame = false;
	
	if (!m_cam_param_ptr)
	{
		return kPODevErrNone;
	}

	tSdkFrameHead sdk_frame_info;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

#if defined(POR_WINDOWS)
	i32 error_code = checkErrorEx(CameraGetImageBufferPriority(cam_handle, &sdk_frame_info,
												&raw_buffer_ptr, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST));
#elif defined(POR_LINUX)
	i32 error_code = checkErrorEx(CameraGetImageBuffer(cam_handle, &sdk_frame_info, &raw_buffer_ptr, 1000));
#endif

	if (error_code == kPODevErrNone)
	{
		checkError(CameraImageProcess(cam_handle, raw_buffer_ptr, rgb_buffer_ptr, &sdk_frame_info));

		i32 capture_width = sdk_frame_info.iWidth;
		i32 capture_height = sdk_frame_info.iHeight;
		ImageData raw_img_data;

		switch (sdk_frame_info.uiMediaType)
		{
			case CAMERA_MEDIA_TYPE_MONO8:
			{
				has_frame = true;
				raw_img_data.setImageData(rgb_buffer_ptr, capture_width, capture_height, 1);
				break;
			}
			case CAMERA_MEDIA_TYPE_RGB8:
			{
				has_frame = true;
				raw_img_data.setImageData(rgb_buffer_ptr, capture_width, capture_height, 3);
				break;
			}
		}

		if (has_frame)
		{
			POMutex* mutex_ptr = frame_info.snap_mutex_ptr;
			if (mutex_ptr)
			{
				mutex_ptr->lock();
			}
			img_data.copyImage(raw_img_data);
			if (mutex_ptr)
			{
				mutex_ptr->unlock();
			}
#if defined(POR_WINDOWS)
			is_flip_y = true;
#endif
		}
		checkError(CameraReleaseImageBuffer(cam_handle, raw_buffer_ptr));
	}
	else if (error_code == kPODevErrException)
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::snapManualToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	bool& is_flip_y = frame_info.is_flip_y;
	has_frame = false;

	capture_width = 0;
	capture_height = 0;
	u8* raw_buffer_ptr = NULL;

	if (!m_cam_param_ptr || !gray_buffer_ptr || !rgb_buffer_ptr)
	{
		return kPODevErrNone;
	}

	tSdkFrameHead sdk_frame_info;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	i32 prev_trigger_mode = kMVTriggerModeContinue;
	if (!changeMVTriggerMode(kMVTriggerModeContinue, prev_trigger_mode))
	{
		return getLastError();
	}

	i32 result = kPODevErrNone;
#if defined(POR_WINDOWS)
	i32 error_code = checkErrorEx(CameraGetImageBufferPriority(cam_handle, &sdk_frame_info,
											&raw_buffer_ptr, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST));
#elif defined(POR_LINUX)
    i32 error_code = checkErrorEx(CameraGetImageBuffer(cam_handle, &sdk_frame_info, &raw_buffer_ptr, 1000));
#endif

	if (error_code == kPODevErrNone)
	{
		capture_width = sdk_frame_info.iWidth;
		capture_height = sdk_frame_info.iHeight;
		checkError(CameraImageProcess(cam_handle, raw_buffer_ptr, rgb_buffer_ptr, &sdk_frame_info));

		if (sdk_frame_info.uiMediaType == CAMERA_MEDIA_TYPE_MONO8)
		{
			has_frame = true;
			memcpy(gray_buffer_ptr, rgb_buffer_ptr, capture_width*capture_height);
#if defined(POR_WINDOWS)
			is_flip_y = true;
#endif
		}
		else
		{
			result = kPODevErrException;
		}
		checkError(CameraReleaseImageBuffer(cam_handle, raw_buffer_ptr));
	}
	else if (error_code == kPODevErrException)
	{
		result = getLastError();
	}

	//restore trigger mode
	if (CPOBase::checkRange(prev_trigger_mode, kMVTriggerModeContinue, kMVTriggerModeExternal))
	{
		i32 store_trigger_mode;
		changeMVTriggerMode(prev_trigger_mode, store_trigger_mode);
	}
	return result;
}

i32 CMindVisionCamera::snapManualToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	bool& is_flip_y = frame_info.is_flip_y;
	has_frame = false;

	u8* raw_buffer_ptr = NULL;
	if (!m_cam_param_ptr || !rgb_buffer_ptr)
	{
		return kPODevErrNone;
	}

	tSdkFrameHead sdk_frame_info;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	i32 prev_trigger_mode = kMVTriggerModeContinue;
	if (!changeMVTriggerMode(kMVTriggerModeContinue, prev_trigger_mode))
	{
		return getLastError();
	}

	i32 result = kPODevErrNone;
#if defined(POR_WINDOWS)
	i32 error_code = checkErrorEx(CameraGetImageBufferPriority(cam_handle, &sdk_frame_info,
												&raw_buffer_ptr, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST));
#elif defined(POR_LINUX)
	i32 error_code = checkErrorEx(CameraGetImageBuffer(cam_handle, &sdk_frame_info, &raw_buffer_ptr, 1000));
#endif

	if (error_code == kPODevErrNone)
	{
		checkError(CameraImageProcess(cam_handle, raw_buffer_ptr, rgb_buffer_ptr, &sdk_frame_info));

		i32 capture_width = sdk_frame_info.iWidth;
		i32 capture_height = sdk_frame_info.iHeight;
		ImageData raw_img_data;

		switch (sdk_frame_info.uiMediaType)
		{
			case CAMERA_MEDIA_TYPE_MONO8:
			{
				has_frame = true;
				raw_img_data.setImageData(rgb_buffer_ptr, capture_width, capture_height, 1);
				break;
			}
			case CAMERA_MEDIA_TYPE_RGB8:
			{
				has_frame = true;
				raw_img_data.setImageData(rgb_buffer_ptr, capture_width, capture_height, 3);
				break;
			}
		}

		if (has_frame)
		{
			POMutex* mutex_ptr = frame_info.snap_mutex_ptr;
			if (mutex_ptr)
			{
				mutex_ptr->lock();
			}
			img_data.copyImage(raw_img_data);
			if (mutex_ptr)
			{
				mutex_ptr->unlock();
			}
#if defined(POR_WINDOWS)
			is_flip_y = true;
#endif
		}
		checkError(CameraReleaseImageBuffer(cam_handle, raw_buffer_ptr));
	}
	else if (error_code == kPODevErrException)
	{
		result = getLastError();
	}

	//restore trigger mode
	if (CPOBase::checkRange(prev_trigger_mode, kMVTriggerModeContinue, kMVTriggerModeExternal))
	{
		i32 store_trigger_mode;
		changeMVTriggerMode(prev_trigger_mode, store_trigger_mode);
	}
	return result;
}

i32 CMindVisionCamera::snapSoftwareTrigger()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrNone;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	if (!checkError(CameraSoftTrigger(cam_handle)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

bool CMindVisionCamera::changeMVTriggerMode(i32 trigger_mode, i32& prev_trigger_mode)
{
	if (!m_cam_param_ptr ||
		!CPOBase::checkRange(trigger_mode, kMVTriggerModeContinue, kMVTriggerModeExternal))
	{
		return false;
	}

	i32 mode;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	//get current camera trigger mode
	if (!checkError(CameraGetTriggerMode(cam_handle, &mode)))
	{
		return false;
	}

	//check current trigger mode and input mode
	if (mode == trigger_mode)
	{
		prev_trigger_mode = mode;
		return true;
	}

	//set triggermode and store prevmode
	if (!checkError(CameraSetTriggerMode(cam_handle, trigger_mode)))
	{
		return false;
	}
	prev_trigger_mode = mode;
	return true;
}

i32 CMindVisionCamera::setGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();

	i32 gain = 0;
	{
		anlock_guard_ptr(cam_info_ptr);
		anlock_guard_ptr(cam_exposure_ptr);

		gain = cam_exposure_ptr->m_gain;
		gain = po::_min(gain, cam_info_ptr->m_gain_max);
		gain = po::_max(gain, cam_info_ptr->m_gain_min);
	}

	if (!checkError(CameraSetAnalogGain(cam_handle, gain)))
	{
		return getLastError();
	}

	return kPODevErrNone;
}

i32 CMindVisionCamera::setExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();

	f64 exposure = 0;
	{
		anlock_guard_ptr(cam_info_ptr);
		anlock_guard_ptr(cam_exposure_ptr);
		exposure = cam_exposure_ptr->m_exposure;
		exposure = po::_min(exposure, cam_info_ptr->m_exposure_max);
		exposure = po::_max(exposure, cam_info_ptr->m_exposure_min);
	}

	if (!checkError(CameraSetExposureTime(cam_handle, exposure * 1000)))
	{
		return getLastError();
	}

	return kPODevErrNone;
}

i32 CMindVisionCamera::setColorMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	// set mediatype and image datatype
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();

	switch (cam_color_ptr->getColorMode())
	{
		case kCamColorGray:
		{
			if (!m_cam_param_ptr->supportFunc(kCamSupportGray))
			{
				return kPODevErrException;
			}
			if (!checkError(CameraSetIspOutFormat(cam_handle, CAMERA_MEDIA_TYPE_MONO8)))
			{
				return getLastError();
			}

			//update color mode
			cam_color_ptr->setColorMode(kCamColorGray);
			break;
		}
		case kCamColorRGB8:
		case kCamColorAny:
		{
			if (!m_cam_param_ptr->supportFunc(kCamSupportColor))
			{
				cam_color_ptr->setColorMode(kCamColorGray);
				return setColorMode();
			}
			if (!checkError(CameraSetIspOutFormat(cam_handle, CAMERA_MEDIA_TYPE_RGB8)))
			{
				return getLastError();
			}

			//update color mode
			cam_color_ptr->setColorMode(kCamColorRGB8);
			break;
		}
		case kCamColorYUV422:
		{
			if (!m_cam_param_ptr->supportFunc(kCamSupportColor))
			{
				cam_color_ptr->setColorMode(kCamColorGray);
				return setColorMode();
			}
			if (!checkError(CameraSetIspOutFormat(cam_handle, CAMERA_MEDIA_TYPE_YUV422_8)))
			{
				return getLastError();
			}

			//update color mode
			cam_color_ptr->setColorMode(kCamColorYUV422);
			break;
		}
		default:
		{
			return kPODevErrException;
		}
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setWhiteBalanceMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	// set mediatype and image datatype
	bool wb_mode;
	i32 color_mode;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();
	{
		anlock_guard_ptr(cam_color_ptr);
		wb_mode = cam_color_ptr->m_wb_mode;
		color_mode = cam_color_ptr->m_color_mode;
	}
	if (color_mode <= kCamColorGray || !m_cam_param_ptr->supportFunc(kCamSupportColor))
	{
		return kPODevErrUnsupport;
	}

	if (!checkError(CameraSetWbMode(cam_handle, (BOOL)wb_mode)))
	{
		return getLastError();
	}
	if (!wb_mode && !checkError(CameraSetOnceWB(cam_handle)))
	{
		return getLastError();
	}

	//update color gain
	f32 rgain, ggain, bgain;
	i32 ret_code = getCameraColorState(rgain, ggain, bgain);
	if (ret_code != kPODevErrNone)
	{
		return ret_code;
	}
	{
		anlock_guard_ptr(cam_color_ptr);
		cam_color_ptr->m_red_gain = rgain;
		cam_color_ptr->m_green_gain = ggain;
		cam_color_ptr->m_blue_gain = bgain;
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setColorAWBOnce()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	// set mediatype and image datatype
	i32 color_mode;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();
	{
		anlock_guard_ptr(cam_color_ptr);
		color_mode = cam_color_ptr->m_color_mode;
	}
	if (color_mode <= kCamColorGray)
	{
		return kPODevErrUnsupport;
	}

	if (!checkError(CameraSetOnceWB(cam_handle)))
	{
		return getLastError();
	}

	//update color gain
	f32 rgain, ggain, bgain;
	i32 ret_code = getCameraColorState(rgain, ggain, bgain);
	if (ret_code != kPODevErrNone)
	{
		return ret_code;
	}
	{
		anlock_guard_ptr(cam_color_ptr);
		cam_color_ptr->m_red_gain = rgain;
		cam_color_ptr->m_green_gain = ggain;
		cam_color_ptr->m_blue_gain = bgain;
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setColorGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	// set mediatype and image datatype
	i32 color_mode, rgain, ggain, bgain;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();
	{
		anlock_guard_ptr(cam_color_ptr);
		color_mode = cam_color_ptr->m_color_mode;
		rgain = (i32)(cam_color_ptr->m_red_gain * 100);
		ggain = (i32)(cam_color_ptr->m_green_gain * 100);
		bgain = (i32)(cam_color_ptr->m_blue_gain * 100);
	}
	if (color_mode <= kCamColorGray || !m_cam_param_ptr->supportFunc(kCamSupportManualWb | 
		kCamSupportRedGain | kCamSupportGreenGain | kCamSupportBlueGain))
	{
		return kPODevErrUnsupport;
	}

	if (!checkError(CameraSetGain(cam_handle, rgain, ggain, bgain)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCorrectionGamma()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportGamma))
	{
		return kPODevErrUnsupport;
	}

	// set mediatype and image datatype
	i32 gamma;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	{
		anlock_guard_ptr(cam_corr_ptr);
		gamma = (i32)(cam_corr_ptr->m_gamma * 100);
	}

	if (!checkError(CameraSetGamma(cam_handle, gamma)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCorrectionContrast()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportContrast))
	{
		return kPODevErrUnsupport;
	}

	// set mediatype and image datatype
	i32 contrast;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	{
		anlock_guard_ptr(cam_corr_ptr);
		contrast = (i32)(cam_corr_ptr->m_contrast);
	}

	if (!checkError(CameraSetContrast(cam_handle, contrast)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCorrectionSaturation()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportSaturation) || !m_cam_param_ptr->isColorMode())
	{
		return kPODevErrUnsupport;
	}

	// set mediatype and image datatype
	i32 saturation;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	{
		anlock_guard_ptr(cam_corr_ptr);
		saturation = (i32)(cam_corr_ptr->m_saturation);
	}

	if (!checkError(CameraSetSaturation(cam_handle, saturation)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCorrectionSharpness()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportSharpness))
	{
		return kPODevErrUnsupport;
	}

	// set mediatype and image datatype
	i32 sharpness;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	{
		anlock_guard_ptr(cam_corr_ptr);
		sharpness = (i32)(cam_corr_ptr->m_sharpness);
	}

	if (!checkError(CameraSetSharpness(cam_handle, sharpness)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setAeGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();

	if (!m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
	{
		return kPODevErrException;
	}

	i32 autogain_min = 0;
	i32 autogain_max = 0;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		autogain_min = cam_exposure_ptr->m_autogain_min;
		autogain_max = cam_exposure_ptr->m_autogain_max;
	}

	if (!checkError(CameraSetAeAnalogGainRange(cam_handle, autogain_min, autogain_max)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setAeExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();

	if (!m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
	{
		return kPODevErrException;
	}

	f64 autoexp_min = 0;
	f64 autoexp_max = 0;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		autoexp_min = cam_exposure_ptr->m_autoexp_min * 1000;
		autoexp_max = cam_exposure_ptr->m_autoexp_max * 1000;
	}
	if (!checkError(CameraSetAeExposureRange(cam_handle, autoexp_min, autoexp_max)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setAeBrightness()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();

	if (!m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
	{
		return kPODevErrException;
	}

	i32 auto_brightness = 0;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		auto_brightness = cam_exposure_ptr->m_auto_brightness;
	}

	if (!checkError(CameraSetAeTarget(cam_handle, auto_brightness)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setAeWindow()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();

	i32 x, y, w, h;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		x = cam_exposure_ptr->m_autoexp_window.x1;
		y = cam_exposure_ptr->m_autoexp_window.y1;
		w = cam_exposure_ptr->m_autoexp_window.getWidth();
		h = cam_exposure_ptr->m_autoexp_window.getHeight();
	}
	if (w == 0 || h == 0)
	{
		anlock_guard_ptr(cam_range_ptr);
		x = cam_range_ptr->m_range.x1;
		y = cam_range_ptr->m_range.y1;
		w = cam_range_ptr->m_range.getWidth();
		h = cam_range_ptr->m_range.getHeight();
	}

	if (!m_cam_param_ptr->supportFunc(kCamSupportAEWindow))
	{
		return kPODevErrException;
	}
	if (!checkError(CameraSetAeWindow(cam_handle, x, y, w, h)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setAeState(const i32 autoexp_mode)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();

	switch (autoexp_mode)
	{
		case kCamAEModeContinuous:
		{
			if (!m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
			{
				return kPODevErrException;
			}
			if (!checkError(CameraSetAeState(cam_handle, TRUE)))
			{
				return getLastError();
			}
			{
				anlock_guard_ptr(cam_spec_ptr);
				checkError(CameraSetAntiFlick(cam_handle, (BOOL)cam_spec_ptr->m_anti_flick));
				checkError(CameraSetLightFrequency(cam_handle,
					cam_spec_ptr->m_ambient_freq == kCamEnvFrequency50Hz ? LIGHT_FREQUENCY_50HZ : LIGHT_FREQUENCY_60HZ));
			}
			break;
		}
		case kCamAEModeOff:
		{
			if (!m_cam_param_ptr->supportFunc(kCamSupportManualExp))
			{
				return kPODevErrException;
			}
			if (!checkError(CameraSetAeState(cam_handle, FALSE)))
			{
				return getLastError();
			}

			i32 gain, exposure;
			{
				anlock_guard_ptr(cam_exposure_ptr);
				gain = cam_exposure_ptr->m_gain;
				exposure = cam_exposure_ptr->m_exposure * 1000;
			}

			//set manual gain and exposure
			if (!checkError(CameraSetAnalogGain(cam_handle, gain)))
			{
				return getLastError();
			}
			if (!checkError(CameraSetExposureTime(cam_handle, exposure)))
			{
				return getLastError();
			}
			checkError(CameraSetAntiFlick(cam_handle, (BOOL)FALSE));
			break;
		}
		default:
		{
			return kPODevErrException;
		}
	}

	return kPODevErrNone;
}

i32 CMindVisionCamera::setTriggerMode(CameraTrigger& cam_trigger)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
		
	anlock_guard(m_dev_trigger);
	anlock_guard(cam_trigger);

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	i32 trigger_mode = cam_trigger.m_trigger_mode;

	/* 촉발방식에 따라 트리거스캔시간을 결정한다. */
	switch (trigger_mode)
	{
		case kCamTriggerContinuous:
		{
			m_dev_trigger.m_trigger_interval = cam_trigger.m_trigger_interval;
			break;
		}
		default:
		{
			m_dev_trigger.m_trigger_interval = 1; //ms
			break;
		}
	}

	//switch camera trigger mode for each camera mode
	i32 dev_trigger_mode = kMVTriggerModeNone;
	checkError(CameraGetTriggerMode(cam_handle, &dev_trigger_mode));

	switch (trigger_mode)
	{
		//set free-trigger mode
		case kCamTriggerContinuous:
		{
			if (dev_trigger_mode != kMVTriggerModeContinue || 
				m_dev_trigger.m_trigger_mode != kMVTriggerModeContinue)
			{
				if (!checkError(CameraSetTriggerMode(cam_handle, kMVTriggerModeContinue)))
				{
					return getLastError();
				}
				m_dev_trigger.m_trigger_mode = kMVTriggerModeContinue;
			}
			break;
		}

		//set manual-trigger mode
		case kCamTriggerManual:
		case kCamTriggerIO:
		case kCamTriggerRS:
		case kCamTriggerNetwork:
		{
			//set trigger mode
			if (dev_trigger_mode != kMVTriggerModeManual || 
				m_dev_trigger.m_trigger_mode != kMVTriggerModeManual)
			{
				if (!checkError(CameraSetTriggerMode(cam_handle, kMVTriggerModeManual)))
				{
					return getLastError();
				}
				checkError(CameraSetTriggerCount(cam_handle, 1));
				m_dev_trigger.m_trigger_mode = kMVTriggerModeManual;
			}
			break;
		}

		//set external-trigger mode
		case kCamTriggerCamera:
		{
			//set trigger mode
			if (dev_trigger_mode != kMVTriggerModeExternal ||
				m_dev_trigger.m_trigger_mode != kMVTriggerModeExternal)
			{
				if (!checkError(CameraSetTriggerMode(cam_handle, kMVTriggerModeExternal)))
				{
					return getLastError();
				}
				m_dev_trigger.m_trigger_mode = kMVTriggerModeExternal;
			}

			//set trigger signal
			i32 dev_signal = cam_trigger.m_trigger_signal;
			switch (dev_signal)
			{
				case kCamTriggerRisingEdge:
				{
					dev_signal = kMVExtTriggerLeadingEdge;
					break;
				}
				case kCamTriggerFallingEdge:
				{
					dev_signal = kMVExtTriggerFallingEdge;
					break;
				}
				default:
				{
					return kPODevErrException;
				}
			}

			/* set trigger signal */
			i32 cur_signal = kCamTriggerSignalNone;
			checkError(CameraGetExtTrigSignalType(cam_handle, &cur_signal));

			if (m_dev_trigger.m_trigger_signal != dev_signal || cur_signal != dev_signal)
			{
				checkError(CameraSetExtTrigSignalType(cam_handle, dev_signal));
				m_dev_trigger.m_trigger_signal = dev_signal;
			}

			/* set trigger delay */
			u32 cur_delayus;
			u32 trigger_delayus = cam_trigger.m_trigger_delay * 1000;
			checkError(CameraGetTriggerDelayTime(cam_handle, &cur_delayus));

			if (m_dev_trigger.m_trigger_delay != trigger_delayus || cur_delayus != trigger_delayus)
			{
				checkError(CameraSetTriggerDelayTime(cam_handle, trigger_delayus));
				m_dev_trigger.m_trigger_delay = trigger_delayus;
			}

			//after camera setting in camera trigger mode
			setShutterMode();
			setShutterJitterTime();
			break;
		}
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCaptureInvert()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCaptureFlip()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();
	if (!cam_range_ptr)
	{
		return kPODevErrException;
	}

	BOOL is_h_flip, is_v_flip;
	{
		anlock_guard_ptr(cam_range_ptr);
		is_h_flip = cam_range_ptr->m_is_flip_x ? TRUE : FALSE;
		is_v_flip = cam_range_ptr->m_is_flip_y ? TRUE : FALSE;
	}

	if (!checkError(CameraSetMirror(cam_handle, MIRROR_DIRECTION_HORIZONTAL, is_h_flip)) ||
		!checkError(CameraSetMirror(cam_handle, MIRROR_DIRECTION_VERTICAL, is_v_flip)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCaptureRotation()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();
	if (!cam_range_ptr)
	{
		return kPODevErrException;
	}

#if defined(POR_WINDOWS)
	i32 rot = ROTATE_DIRECTION_0;
	switch (cam_range_ptr->getRotation())
	{
		case kPORotation90:
		{
			rot = ROTATE_DIRECTION_90;
			break;
		}
		case kPORotation180:
		{
			rot = ROTATE_DIRECTION_180;
			break;
		}
		case kPORotation270:
		{
			rot = ROTATE_DIRECTION_270;
			break;
		}
		default:
		{
			rot = ROTATE_DIRECTION_0;
			break;
		}
	}
	if (!checkError(CameraSetRotate(cam_handle, rot)))
	{
		return getLastError();
	}
#endif
	return kPODevErrNone;
}

i32 CMindVisionCamera::setShutterJitterTime()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (m_dev_trigger.getTriggerMode() != kMVTriggerModeExternal)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	if (!checkError(CameraSetExtTrigJitterTime(cam_handle, cam_spec_ptr->getJitterTimeus())))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setShutterMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportShutter))
	{
		return kPODevErrUnsupport;
	}
	if (m_dev_trigger.getTriggerMode() != kMVTriggerModeExternal)
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	i32 shutter_mode = (cam_spec_ptr->getShutterMode() == kShutterRolling) ? EXT_TRIG_EXP_STANDARD : EXT_TRIG_EXP_GRR;
	if (!checkError(CameraSetExtTrigShutterType(cam_handle, shutter_mode)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setNoiseReduction()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	bool noise_reduction = false;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	{
		anlock_guard_ptr(cam_spec_ptr);
		noise_reduction = cam_spec_ptr->m_noise_reduce;
	}
	if (!checkError(CameraSetNoiseFilter(cam_handle, noise_reduction)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setColorTemperature()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 color_temp = false;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	{
		anlock_guard_ptr(cam_spec_ptr);
		color_temp = cam_spec_ptr->m_color_temperature;
	}
	if (!m_cam_param_ptr->isColorMode())
	{
		return kPODevErrException;
	}

	i32 mv_color_temp = (color_temp == kCamColorTempAuto) ? CT_MODE_AUTO : CT_MODE_PRESET;
	if (!checkError(CameraSetClrTempMode(cam_handle, mv_color_temp)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setStrobeEnabled()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportAutoStrobe | kCamSupportManualStrobe))
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraStrobe* cam_strobe_ptr = m_cam_param_ptr->getCameraStrobe();
	if (cam_strobe_ptr->isEnabled())
	{
		checkError(CameraSetOutPutIOMode(cam_handle, 0, IOMODE_STROBE_OUTPUT));
	}
	else
	{
		checkError(CameraSetOutPutIOMode(cam_handle, 0, IOMODE_GP_OUTPUT));
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setStrobeControl()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportAutoStrobe | kCamSupportManualStrobe))
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraStrobe* cam_strobe_ptr = m_cam_param_ptr->getCameraStrobe();
	if (cam_strobe_ptr->isEnabled())
	{
		if (cam_strobe_ptr->isAutoStrobeMode())
		{
			checkError(CameraSetStrobeMode(cam_handle, STROBE_SYNC_WITH_TRIG_AUTO));
		}
		else
		{
			checkError(CameraSetStrobeMode(cam_handle, STROBE_SYNC_WITH_TRIG_MANUAL));
			checkError(CameraSetStrobeDelayTime(cam_handle, cam_strobe_ptr->getStrobePWMDelay()));
			checkError(CameraSetStrobePulseWidth(cam_handle, cam_strobe_ptr->getStrobePWMWidth()));
			checkError(CameraSetStrobePolarity(cam_handle, cam_strobe_ptr->getStrobeLevel()));
		}
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setLightForTrigger(bool use_strobe)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportManualStrobe))
	{
		return kPODevErrException;
	}

	CameraStrobe* cam_strobe_ptr = m_cam_param_ptr->getCameraStrobe();
	if (!cam_strobe_ptr->isAutoStrobeMode())
	{
		i32 cam_strobe_level = cam_strobe_ptr->getStrobeLevel();
		i32 cam_handle = m_cam_param_ptr->getCamHandle();
		i32 strobe_mode = (cam_strobe_level + (use_strobe ? 1 : 0)) % kStrobeLevelCount;
		if (!checkError(CameraSetStrobePolarity(cam_handle, strobe_mode)))
		{
			return getLastError();
		}
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setCaptureRange()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportRange))
	{
		return kPODevErrException;
	}

	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();
	if (!cam_range_ptr)
	{
		return kPODevErrException;
	}

	//[MVSDK API Documentation Examples/roi.cpp]
	// Set to 0xff for custom resolution, set to 0 to N for select preset resolution
	//offsetx, offsety, width, height: 
	//offset width and height are all chosen to be 16 times the best compatibility 
	//(different cameras have different requirements for this, some need only a multiple of 2,
	//some may require a multiple of 16)
	i32 NX = 16;
	i32 NY = 4; //자체추가함
	tSdkImageResolution resolution;
	memset(&resolution, 0, sizeof(tSdkImageResolution));
	checkError(CameraGetImageResolution(cam_handle, &resolution));
	resolution.iIndex = 0xFF; //.iIndex == 0XFF, that is a custom resolution Switch to ROI resolution

	i32 max_width;
	i32 max_height;
	{
		CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
		anlock_guard_ptr(cam_info_ptr);
		max_width = cam_info_ptr->m_max_width;
		max_height = cam_info_ptr->m_max_height;
	}

	i32 x, y, w, h;
	Recti range = cam_range_ptr->getRange();
	x = range.x1;
	y = range.y1;
	w = range.getWidth();
	h = range.getHeight();
	if (w <= 0)
	{
		x = 0; w = max_width;
	}
	if (h <= 0)
	{
		y = 0; h = max_height;
	}
	//if is not full resolution
	if (x != 0 || y != 0 || w != max_width || h != max_height)
	{
		w = CPOBase::round(w, NX);
		h = CPOBase::round(h, NY);
		if (x + w > max_width)
		{
			w = range.getWidth() / NX* NX;
		}
		if (y + h > max_height)
		{
			h = range.getHeight() / NY* NY;
		}
		if (x < 0 || y < 0 || w <= 0 || h <= 0)
		{
			return kPODevErrException;
		}
	}

	//if updated, update camera range
	if (x != range.x1 || y != range.y1 || w != range.getWidth() || h != range.getHeight())
	{
		anlock_guard_ptr(cam_range_ptr);
		cam_range_ptr->m_range = Recti(x, y, x + w, y + h);
	}

	resolution.iHOffsetFOV = x;
	resolution.iVOffsetFOV = y;
	resolution.iWidth = resolution.iWidthFOV = w;
	resolution.iHeight = resolution.iHeightFOV = h;

	// ISP software zoom width and height, all 0 means not zoom
	resolution.iWidthZoomSw = 0;
	resolution.iHeightZoomSw = 0;

	// BIN SKIP mode setting (requires camera hardware support)
	resolution.uBinAverageMode = 0;
	resolution.uBinSumMode = 0;
	resolution.uResampleMask = 0;
	resolution.uSkipMode = 0;

	if (!checkError(CameraSetImageResolution(cam_handle, &resolution)) ||
		!checkError(CameraSetResolutionForSnap(cam_handle, &resolution)))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::setEmuSampler(void* emu_sample_ptr)
{
	return kPODevErrNone;
}

bool CMindVisionCamera::needTriggerScan()
{
	return (m_dev_trigger.getTriggerMode() == kMVTriggerModeContinue);
}

i32 CMindVisionCamera::getTriggerInterval()
{
	if (!needTriggerScan())
	{
		return 0;
	}
	return m_dev_trigger.getTriggerInterval();
}

f32 CMindVisionCamera::getGain()
{
	if (!m_cam_param_ptr)
	{
		return 0;
	}

	i32 tmp_gain;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	if (!checkError(CameraGetAnalogGain(cam_handle, &tmp_gain)))
	{
		return 0;
	}
	return tmp_gain;
}

f32 CMindVisionCamera::getExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return 0;
	}

	f64 tmp_expsoure_time_us;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	if (!checkError(CameraGetExposureTime(cam_handle, &tmp_expsoure_time_us)))
	{
		return getLastError();
	}

	return tmp_expsoure_time_us / 1000;
}

i32 CMindVisionCamera::getCameraState(bool& autoexp_mode, f32& gain, f32& expsoure_time_ms)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	BOOL tmp_autoexp_mode;
	i32 tmp_gain;
	f64 tmp_expsoure_time_us;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	if (!checkError(CameraGetAeState(cam_handle, &tmp_autoexp_mode)))
	{
		return getLastError();
	}

	if (!checkError(CameraGetAnalogGain(cam_handle, &tmp_gain)))
	{
		return getLastError();
	}

	if (!checkError(CameraGetExposureTime(cam_handle, &tmp_expsoure_time_us)))
	{
		return getLastError();
	}

	autoexp_mode = tmp_autoexp_mode;
	expsoure_time_ms = tmp_expsoure_time_us / 1000;
	gain = tmp_gain;
	return kPODevErrNone;
}

i32 CMindVisionCamera::getCameraColorState(f32& rgain, f32& ggain, f32& bgain)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	rgain = 0; ggain = 0; bgain = 0;

	i32 irgain, iggain, ibgain;
	i32 cam_handle = m_cam_param_ptr->getCamHandle();
	if (!checkError(CameraGetGain(cam_handle, &irgain, &iggain, &ibgain)))
	{
		return getLastError();
	}
	rgain = (f32)irgain / 100.0f;
	ggain = (f32)iggain / 100.0f;
	bgain = (f32)ibgain / 100.0f;
	return kPODevErrNone;
}

bool CMindVisionCamera::checkError(const i32 error_code)
{
	if (error_code != CAMERA_STATUS_SUCCESS)
	{
		m_last_error_code = kPODevErrException;
		switch (error_code)
		{
			/* 카메라촉발방식에서는 카메라의 시간초과오유는 장치오유로 인식하지 않는다.
			또한 장치련결해제, 통신선로오유만 련결오유로 인식한다. 나머지 오유들은 기타오유로 인식한다. */
			case CAMERA_STATUS_TIME_OUT:
			{
				if (m_dev_trigger.getTriggerMode() != kMVTriggerModeContinue)
				{
					return true;
				}
			}
			case CAMERA_STATUS_DEVICE_LOST:
			case CAMERA_STATUS_COMM_ERROR:
			case CAMERA_STATUS_BUS_ERROR:
			{
				m_last_error_code = kPODevErrConnect;
				break;
			}
		}

		i32 cam_id = m_cam_param_ptr->getCamID();
		printlog_lvs2(QString("Camera%1 Error: %2").arg(cam_id).arg(CameraGetErrorString(error_code)), LOG_SCOPE_CAM);
		return false;
	}
	return true;
}

i32 CMindVisionCamera::checkErrorEx(const i32 error_code)
{
	if (error_code == CAMERA_STATUS_SUCCESS)
	{
		/* 오유없음 */
		return kPODevErrNone;
	}

	m_last_error_code = kPODevErrException;
	switch (error_code)
	{
		/* 카메라촉발방식에서는 카메라의 시간초과오유는 장치오유로 인식하지 않는다.
		또한 장치련결해제, 통신선로오유만 련결오유로 인식한다. 나머지 오유들은 기타오유로 인식한다. */
		case CAMERA_STATUS_TIME_OUT:
		{
			if (m_dev_trigger.getTriggerMode() != kMVTriggerModeContinue)
			{
				/* 장치에서 일정한 시간동안 응답없음, 하지만 카메라촉발방식이므로 오유로 웃준위모듈에 통보하지는 않는다. */
				return kPODevErrConnect;
			}
		}
		case CAMERA_STATUS_DEVICE_LOST:
		case CAMERA_STATUS_COMM_ERROR:
		case CAMERA_STATUS_BUS_ERROR:
		{
			m_last_error_code = kPODevErrConnect;
			break;
		}
	}

	i32 cam_id = m_cam_param_ptr->getCamID();
	printlog_lvs2(QString("Camera%1 Error: %2").arg(cam_id).arg(CameraGetErrorString(error_code)), LOG_SCOPE_CAM);

	/* 오유경우 */
	return kPODevErrException;
}

i32 CMindVisionCamera::saveCameraParamToFile(const postring& cam_param_file)
{
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	if (!checkError(CameraSaveParameterToFile(cam_handle, const_cast<char*>(cam_param_file.c_str()))))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

i32 CMindVisionCamera::loadCameraParamToFile(const postring& cam_param_file)
{
	i32 cam_handle = m_cam_param_ptr->getCamHandle();

	if (!checkError(CameraReadParameterFromFile(cam_handle, const_cast<char*>(cam_param_file.c_str()))))
	{
		return getLastError();
	}
	return kPODevErrNone;
}

#endif
#endif

#include "basler_pylon_camera.h"
#include "base.h"
#include "proc/image_proc.h"

#if defined(POR_WITH_CAMERA)
#if defined(POR_SUPPORT_BASLER)

#include <pylon/gige/BaslerGigEInstantCamera.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>

//////////////////////////////////////////////////////////////////////////
i32 CBaslerPylonCamera::initSDK()
{
	try
	{
		Pylon::PylonInitialize();
		const char* basler_pylon_version = Pylon::GetPylonVersionString();
		printlog_lv1(QString("Basler Pylons SDKversion: %1").arg(basler_pylon_version));
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
        printlog_lv1(QString("BaslerPylon init sdk is failed, %1").arg(e.GetDescription()));
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::exitSDK()
{
	try
	{
		Pylon::PylonTerminate();
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		printlog_lv1(QString("BaslerPylon exit sdk is failed, %1").arg(e.GetDescription()));
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::getAvailableCamera(CameraDevice* cam_device_ptr, i32& cam_count)
{
	try
	{
		// Get the transport layer factory.
		Pylon::CTlFactory& tlFactory = Pylon::CTlFactory::GetInstance();

		// Get all attached devices and exit application if no device is found.
		Pylon::DeviceInfoList_t device_list;
		if (tlFactory.EnumerateDevices(device_list) == 0 || cam_count >= PO_MAX_CAM_COUNT)
		{
			return kPODevErrNone;
		}

		CameraDevice* tmp_device_ptr;
		Pylon::String_t dev_class, dev_full_name;
		i32 i, dev_count = (i32)device_list.size();

		for (i = 0; i < dev_count; i++)
		{
			Pylon::CDeviceInfo& dev_info = device_list[i];
			dev_class = dev_info.GetDeviceClass();
			tmp_device_ptr = NULL;

			if (dev_class == Pylon::BaslerGigEDeviceClass)
			{
				const Pylon::CBaslerGigEDeviceInfo& gige_info = static_cast<const Pylon::CBaslerGigEDeviceInfo&>(dev_info);
				if (!gige_info.IsInterfaceAvailable())
				{
					continue;
				}

				tmp_device_ptr = cam_device_ptr + cam_count;
				tmp_device_ptr->m_cam_type = kPOCamBaslerPylon;
				tmp_device_ptr->m_cam_name = gige_info.GetSerialNumber().c_str();
				tmp_device_ptr->m_cam_reserved[0] = kPOCamBaslerGigE;
			}
			else if (dev_class == Pylon::BaslerUsbDeviceClass)
			{
				const Pylon::CBaslerUsbDeviceInfo& usb_info = static_cast<const Pylon::CBaslerUsbDeviceInfo&>(dev_info);
				if (!usb_info.IsDeviceGUIDAvailable())
				{
					continue;
				}

				tmp_device_ptr = cam_device_ptr + cam_count;
				tmp_device_ptr->m_cam_type = kPOCamBaslerPylon;
				tmp_device_ptr->m_cam_name = usb_info.GetSerialNumber().c_str();
				tmp_device_ptr->m_cam_reserved[0] = kPOCamBaslerUsb;
			}
			else
			{
				continue;
			}

			dev_full_name = dev_info.GetFullName();
			tmp_device_ptr->m_cam_blob.initBuffer((const u8*)dev_full_name.c_str(), (i32)dev_full_name.length() + 1);

			cam_count++;
			if (cam_count >= PO_MAX_CAM_COUNT)
			{
				break;
			}
		}

	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		printlog_lvs2(QString("Basler pylon available camera search is failed, %1")
						.arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

//////////////////////////////////////////////////////////////////////////
CBaslerPylonCamera::CBaslerPylonCamera(CameraSetting* cam_param_ptr)
{
	m_cam_param_ptr = cam_param_ptr;

	/* 트리거모드의 초기값이 0 즉 kCamTriggerContinuous과 같다. 
	따라서 초기트리거모드설정을 위하여 초기값을 -1로 설정한다. */ 
	m_dev_trigger.init();
	m_dev_trigger.m_trigger_mode = kCamTriggerModeCount; //thread_safe
}

CBaslerPylonCamera::~CBaslerPylonCamera()
{
}

i32 CBaslerPylonCamera::initCamera()
{
	/* 트리거모드의 초기값이 0 즉 kCamTriggerContinuous과 같다.
	따라서 초기트리거모드설정을 위하여 초기값을 -1로 설정한다. */
	m_dev_trigger.init();
	m_dev_trigger.m_trigger_mode = kCamTriggerModeCount; //thread_safe

	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	i32 cam_id = m_cam_param_ptr->getCamID();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();

	switch (cam_info_ptr->getReserved(0))
	{
		case kPOCamBaslerGigE:
		{
			return initGigECamera();
		}
		case kPOCamBaslerUsb:
		{
			return initUsbCamera();
		}
		default:
		{
			return kPODevErrException;
		}
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::initGigECamera()
{
	Pylon::CBaslerGigEInstantCamera* cam_device_ptr = NULL;

	try
	{
		CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
		CameraState* cam_state_ptr = m_cam_param_ptr->getCameraState();
		anlock_guard_ptr(cam_info_ptr);

		// Get the transport layer factory.
		Pylon::CTlFactory& tlFactory = Pylon::CTlFactory::GetInstance();

		Pylon::String_t cam_dev_name((char*)cam_info_ptr->m_cam_blob.getBuffer());
		cam_device_ptr = po_new Pylon::CBaslerGigEInstantCamera(tlFactory.CreateDevice(cam_dev_name));
		m_cam_param_ptr->m_cam_handle = (i64)cam_device_ptr;

		i32 width = 0, height = 0;
		u32 cam_capability = kCamSupportNone;

		// Open the camera for accessing the parameters.
		cam_device_ptr->Open();

		/* 장치를 Open한후에 설정제한을 해제한다. */
		if (GenApi::IsAvailable(cam_device_ptr->RemoveLimits))
		{
			cam_device_ptr->RemoveLimits.SetValue(true);
		}

		width = cam_device_ptr->WidthMax.GetValue(true);
		height = cam_device_ptr->HeightMax.GetValue(true);

		if (GenApi::IsAvailable(cam_device_ptr->PixelFormat.GetEntry(Basler_GigECamera::PixelFormat_Mono8)))
		{
			cam_capability |= kCamSupportGray;
		}
		if (GenApi::IsAvailable(cam_device_ptr->PixelFormat.GetEntry(Basler_GigECamera::PixelFormat_RGB8Packed)))
		{
			cam_capability |= kCamSupportColor;
		}

		if (GenApi::IsAvailable(cam_device_ptr->Width) && GenApi::IsAvailable(cam_device_ptr->Height) &&
			GenApi::IsAvailable(cam_device_ptr->OffsetX) && GenApi::IsAvailable(cam_device_ptr->OffsetY))
		{
			cam_capability |= kCamSupportRange;
		}

		if (GenApi::IsAvailable(cam_device_ptr->GainAuto.GetEntry(Basler_GigECamera::GainAuto_Off)) &&
			GenApi::IsAvailable(cam_device_ptr->ExposureAuto.GetEntry(Basler_GigECamera::ExposureAuto_Off)) &&
			GenApi::IsAvailable(cam_device_ptr->GainRaw) &&
			GenApi::IsAvailable(cam_device_ptr->ExposureTimeRaw))
		{
			cam_capability |= kCamSupportManualExp;
		}

		if (GenApi::IsAvailable(cam_device_ptr->GainAuto.GetEntry(Basler_GigECamera::GainAuto_Continuous)) &&
			GenApi::IsAvailable(cam_device_ptr->ExposureAuto.GetEntry(Basler_GigECamera::ExposureAuto_Continuous)) &&
			GenApi::IsAvailable(cam_device_ptr->AutoGainRawLowerLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoGainRawUpperLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoExposureTimeAbsLowerLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoExposureTimeAbsUpperLimit))
		{
			cam_capability |= kCamSupportAutoExp;

			if (GenApi::IsAvailable(cam_device_ptr->AutoTargetValue))
			{
				cam_capability |= kCamSupportAEBrightness;
			}

			if (GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOIOffsetX) &&
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOIOffsetY) &&
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOIWidth) &&
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOIHeight) &&
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOISelector))
			{
				cam_device_ptr->AutoFunctionAOISelector.SetValue(Basler_GigECamera::AutoFunctionAOISelector_AOI2);
				cam_capability |= kCamSupportAEWindow;

				if (GenApi::IsWritable(cam_device_ptr->AutoFunctionAOIUsageIntensity))
				{
					cam_device_ptr->AutoFunctionAOIUsageIntensity.SetValue(true);
				}
			}
		}

		if (GenApi::IsAvailable(cam_device_ptr->AutoFunctionAOIUsageWhiteBalance))
		{
			cam_capability |= kCamSupportAutoWb;
		}
		if (GenApi::IsAvailable(cam_device_ptr->ShutterMode))
		{
			cam_capability |= kCamSupportShutter;
		}
		if (GenApi::IsAvailable(cam_device_ptr->LineSelector) && GenApi::IsAvailable(cam_device_ptr->LineMode) &&
			GenApi::IsAvailable(cam_device_ptr->LineSource))
		{
			//camera trigger : Line1
			if (cam_device_ptr->LineSelector.GetEntry(Basler_GigECamera::LineSelector_Line1))
			{
				cam_device_ptr->LineSelector.SetValue(Basler_GigECamera::LineSelector_Line1);
				if (cam_device_ptr->LineMode.GetEntry(Basler_GigECamera::LineMode_Input))
				{
					cam_device_ptr->LineMode.SetValue(Basler_GigECamera::LineMode_Input);
				}
			}

			//strobe: Out1
			if (cam_device_ptr->LineSelector.GetEntry(Basler_GigECamera::LineSelector_Out1))
			{
				cam_device_ptr->LineSelector.SetValue(Basler_GigECamera::LineSelector_Out1);
				if (cam_device_ptr->LineMode.GetEntry(Basler_GigECamera::LineMode_Output))
				{
					cam_device_ptr->LineMode.SetValue(Basler_GigECamera::LineMode_Output);
					if (cam_device_ptr->LineSource.GetEntry(Basler_GigECamera::LineSource_FlashWindow) &&
						cam_device_ptr->LineSource.GetEntry(Basler_GigECamera::LineSource_UserOutput))
					{
						cam_device_ptr->LineSource.SetValue(Basler_GigECamera::LineSource_UserOutput);
						cam_capability |= kCamSupportAutoStrobe;
					}
				}
			}
		}
		if (CPOBase::bitCheck(cam_capability, kCamSupportColor))
		{
			//TODO:Not Impelemented

			//kCamSupportManualWb
			//kCamSupportRedGamma
			//kCamSupportGreenGamma
			//kCamSupportBlueGamma
		}
		if (GenApi::IsAvailable(cam_device_ptr->GammaEnable) &&
			GenApi::IsAvailable(cam_device_ptr->GammaSelector) &&
			GenApi::IsAvailable(cam_device_ptr->Gamma))
		{
			if (GenApi::IsWritable(cam_device_ptr->GammaEnable))
			{
				cam_device_ptr->GammaEnable.SetValue(true);
			}
			if (GenApi::IsWritable(cam_device_ptr->GammaSelector))
			{
				cam_device_ptr->GammaSelector.SetValue(Basler_GigECamera::GammaSelector_User);
			}
			cam_capability |= kCamSupportGamma;
		}
		if (GenApi::IsAvailable(cam_device_ptr->BslContrast))
		{
			if (GenApi::IsWritable(cam_device_ptr->BslContrastMode))
			{
				cam_device_ptr->BslContrastMode.SetValue(Basler_GigECamera::BslContrastMode_Linear);
			}
			cam_capability |= kCamSupportContrast;
		}
		if (GenApi::IsAvailable(cam_device_ptr->PgiMode))
		{
			cam_capability |= kCamSupportNoiseReduction;
			if (GenApi::IsAvailable(cam_device_ptr->SharpnessEnhancementRaw))
			{
				cam_capability |= kCamSupportSharpness;
			}
		}

		//digital shift default value
		if (GenApi::IsAvailable(cam_device_ptr->DigitalShift) && GenApi::IsWritable(cam_device_ptr->DigitalShift))
		{
			cam_device_ptr->DigitalShift.SetValue(0);
		}

		//restore width and height
		if (GenApi::IsAvailable(cam_device_ptr->Width))
		{
			cam_device_ptr->Width.SetValue(width);
		}
		if (GenApi::IsAvailable(cam_device_ptr->Height))
		{
			cam_device_ptr->Height.SetValue(height);
		}
		if (GenApi::IsWritable(cam_device_ptr->GammaEnable))
		{
			cam_device_ptr->GammaEnable.SetValue(true);
			cam_device_ptr->GammaSelector.SetValue(Basler_GigECamera::GammaSelector_User);
		}

		{
			anlock_guard_ptr(cam_info_ptr);
			cam_info_ptr->m_cam_name = (char*)cam_info_ptr->m_cam_blob.getBuffer();
			cam_info_ptr->m_cam_capability = cam_capability;
			cam_info_ptr->m_gain_min = cam_device_ptr->GainRaw.GetMin();
			cam_info_ptr->m_gain_max = cam_device_ptr->GainRaw.GetMax();
			cam_info_ptr->m_exposure_min = (f32)cam_device_ptr->ExposureTimeRaw.GetMin() / 1000;
			cam_info_ptr->m_exposure_max = (f32)cam_device_ptr->ExposureTimeRaw.GetMax() / 1000;
			cam_info_ptr->m_brightness_min = cam_device_ptr->AutoTargetValue.GetMin();
			cam_info_ptr->m_brightness_max = cam_device_ptr->AutoTargetValue.GetMax();
			cam_info_ptr->m_max_width = width;
			cam_info_ptr->m_max_height = height;
		}
		cam_state_ptr->m_capture_width = width;
        cam_state_ptr->m_capture_height = height;
    }
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		if (cam_device_ptr && cam_device_ptr->IsOpen())
		{
			cam_device_ptr->Close();
		}
		printlog_lvs2(QString("BaslerGigE camera init failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::initUsbCamera()
{
	Pylon::CBaslerUsbInstantCamera* cam_device_ptr = NULL;

	try
    {
        CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
		CameraState* cam_state_ptr = m_cam_param_ptr->getCameraState();
		anlock_guard_ptr(cam_info_ptr);

		// Get the transport layer factory.
		Pylon::CTlFactory& tlFactory = Pylon::CTlFactory::GetInstance();

        Pylon::String_t cam_dev_name((char*)cam_info_ptr->m_cam_blob.getBuffer());
        cam_device_ptr = po_new Pylon::CBaslerUsbInstantCamera(tlFactory.CreateDevice(cam_dev_name));
        m_cam_param_ptr->m_cam_handle = (i64)cam_device_ptr;

        i32 width = 0, height = 0;
		u32 cam_capability = kCamSupportNone;

		// Open the camera for accessing the parameters.
        cam_device_ptr->Open();

        /* 장치를 Open한후에 설정제한을 해제한다. */
		if (GenApi::IsAvailable(cam_device_ptr->RemoveParameterLimit))
		{
			cam_device_ptr->RemoveParameterLimit.SetValue(true);
		}

		width = cam_device_ptr->WidthMax.GetValue(true);
		height = cam_device_ptr->HeightMax.GetValue(true);

		if (GenApi::IsAvailable(cam_device_ptr->PixelFormat.GetEntry(Basler_UsbCameraParams::PixelFormat_Mono8)))
		{
			cam_capability |= kCamSupportGray;
		}
		if (GenApi::IsAvailable(cam_device_ptr->PixelFormat.GetEntry(Basler_UsbCameraParams::PixelFormat_RGB8)))
		{
			cam_capability |= kCamSupportColor;
		}
		if (GenApi::IsAvailable(cam_device_ptr->Width) && GenApi::IsAvailable(cam_device_ptr->Height) &&
			GenApi::IsAvailable(cam_device_ptr->OffsetX) && GenApi::IsAvailable(cam_device_ptr->OffsetY))
		{
			cam_capability |= kCamSupportRange;
		}

		if (GenApi::IsAvailable(cam_device_ptr->GainAuto.GetEntry(Basler_UsbCameraParams::GainAuto_Off)) && 
			GenApi::IsAvailable(cam_device_ptr->ExposureAuto.GetEntry(Basler_UsbCameraParams::ExposureAuto_Off)) && 
			GenApi::IsAvailable(cam_device_ptr->Gain) && 
			GenApi::IsAvailable(cam_device_ptr->ExposureTime))
		{
			cam_capability |= kCamSupportManualExp;
		}

		if (GenApi::IsAvailable(cam_device_ptr->GainAuto.GetEntry(Basler_UsbCameraParams::GainAuto_Continuous)) && 
			GenApi::IsAvailable(cam_device_ptr->ExposureAuto.GetEntry(Basler_UsbCameraParams::ExposureAuto_Continuous)) &&
			GenApi::IsAvailable(cam_device_ptr->AutoGainLowerLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoGainUpperLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoExposureTimeLowerLimit) &&
			GenApi::IsAvailable(cam_device_ptr->AutoExposureTimeUpperLimit))
		{
			cam_capability |= kCamSupportAutoExp;

			if (GenApi::IsAvailable(cam_device_ptr->AutoTargetBrightness))
			{
				cam_capability |= kCamSupportAEBrightness;
			}

			if (GenApi::IsAvailable(cam_device_ptr->AutoFunctionROIOffsetX) && 
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionROIOffsetY) && 
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionROIWidth) && 
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionROIHeight) && 
				GenApi::IsAvailable(cam_device_ptr->AutoFunctionROISelector))
			{
				cam_device_ptr->AutoFunctionROISelector.SetValue(Basler_UsbCameraParams::AutoFunctionROISelector_ROI1);
				cam_capability |= kCamSupportAEWindow;

				if (GenApi::IsWritable(cam_device_ptr->AutoFunctionROIUseBrightness))
				{
					cam_device_ptr->AutoFunctionROIUseBrightness.SetValue(true);
				}
			}
		}

		if (GenApi::IsAvailable(cam_device_ptr->AutoFunctionROIUseWhiteBalance))
		{
			cam_capability |= kCamSupportAutoWb;
		}
		if (GenApi::IsAvailable(cam_device_ptr->SensorShutterMode))
		{
			cam_capability |= kCamSupportShutter;
		}
		if (GenApi::IsAvailable(cam_device_ptr->LineSelector) && GenApi::IsAvailable(cam_device_ptr->LineMode) &&
			GenApi::IsAvailable(cam_device_ptr->LineSource))
		{
			//camera trigger : Line1
			if (cam_device_ptr->LineSelector.GetEntry(Basler_UsbCameraParams::LineSelector_Line1))
			{
				cam_device_ptr->LineSelector.SetValue(Basler_UsbCameraParams::LineSelector_Line1);
				if (cam_device_ptr->LineMode.GetEntry(Basler_UsbCameraParams::LineMode_Input))
				{
					cam_device_ptr->LineMode.SetValue(Basler_UsbCameraParams::LineMode_Input);
				}
			}

			//strobe: Line2
			if (cam_device_ptr->LineSelector.GetEntry(Basler_UsbCameraParams::LineSelector_Line2))
			{
				cam_device_ptr->LineSelector.SetValue(Basler_UsbCameraParams::LineSelector_Line2);
				if (cam_device_ptr->LineMode.GetEntry(Basler_UsbCameraParams::LineMode_Output))
				{
					cam_device_ptr->LineMode.SetValue(Basler_UsbCameraParams::LineMode_Output);
					if (cam_device_ptr->LineSource.GetEntry(Basler_UsbCameraParams::LineSource_FlashWindow) &&
						cam_device_ptr->LineSource.GetEntry(Basler_UsbCameraParams::LineSource_UserOutput1))
					{
						cam_device_ptr->LineSource.SetValue(Basler_UsbCameraParams::LineSource_UserOutput1);
						cam_capability |= kCamSupportAutoStrobe;
					}
				}
			}
		}
		if (CPOBase::bitCheck(cam_capability, kCamSupportColor))
		{
			//TODO:Not Impelemented

			//kCamSupportManualWb
			//kCamSupportRedGamma
			//kCamSupportGreenGamma
			//kCamSupportBlueGamma
		}
		if (GenApi::IsAvailable(cam_device_ptr->Gamma))
		{
			cam_capability |= kCamSupportGamma;
		}
		if (GenApi::IsAvailable(cam_device_ptr->BslContrast))
		{
			if (GenApi::IsWritable(cam_device_ptr->BslContrastMode))
			{
				cam_device_ptr->BslContrastMode.SetValue(Basler_UsbCameraParams::BslContrastMode_Linear);
			}
			cam_capability |= kCamSupportContrast;
		}
		if (GenApi::IsAvailable(cam_device_ptr->PgiMode))
		{
			cam_capability |= kCamSupportNoiseReduction;
			if (GenApi::IsAvailable(cam_device_ptr->SharpnessEnhancement))
			{
				cam_capability |= kCamSupportSharpness;
			}
		}

		//digital shift default value
		if (GenApi::IsAvailable(cam_device_ptr->DigitalShift) && GenApi::IsWritable(cam_device_ptr->DigitalShift))
		{
			cam_device_ptr->DigitalShift.SetValue(1);
		}
		
		//restore width and height
		if (GenApi::IsAvailable(cam_device_ptr->Width))
		{
			cam_device_ptr->Width.SetValue(width);
		}
		if (GenApi::IsAvailable(cam_device_ptr->Height))
		{
			cam_device_ptr->Height.SetValue(height);
		}

		{
			anlock_guard_ptr(cam_info_ptr);
			cam_info_ptr->m_cam_name = (char*)cam_info_ptr->m_cam_blob.getBuffer();
			cam_info_ptr->m_cam_capability = cam_capability;
			if (IsAvailable(cam_device_ptr->Gain))
			{
				cam_info_ptr->m_gain_min = cam_device_ptr->Gain.GetMin();
				cam_info_ptr->m_gain_max = cam_device_ptr->Gain.GetMax();
			}
			if (IsAvailable(cam_device_ptr->ExposureTime))
			{
				cam_info_ptr->m_exposure_min = (f32)cam_device_ptr->ExposureTime.GetMin() / 1000;
				cam_info_ptr->m_exposure_max = (f32)cam_device_ptr->ExposureTime.GetMax() / 1000;
			}
			if (IsAvailable(cam_device_ptr->AutoTargetBrightness))
			{
				cam_info_ptr->m_brightness_min = cam_device_ptr->AutoTargetBrightness.GetMin();
				cam_info_ptr->m_brightness_max = cam_device_ptr->AutoTargetBrightness.GetMax();
			}

			if (IsAvailable(cam_device_ptr->Gamma))
			{
				cam_info_ptr->m_gamma_min = (f32)cam_device_ptr->Gamma.GetMin();
				cam_info_ptr->m_gamma_max = (f32)cam_device_ptr->Gamma.GetMax();
			}
			if (IsAvailable(cam_device_ptr->BslContrast))
			{
				cam_info_ptr->m_contrast_min = (f32)cam_device_ptr->BslContrast.GetMin();
				cam_info_ptr->m_contrast_max = (f32)cam_device_ptr->BslContrast.GetMax();
			}
			if (IsAvailable(cam_device_ptr->BslSaturation))
			{
				cam_info_ptr->m_saturation_min = (f32)cam_device_ptr->BslSaturation.GetMin();
				cam_info_ptr->m_saturation_max = (f32)cam_device_ptr->BslSaturation.GetMax();
			}
			if (IsAvailable(cam_device_ptr->SharpnessEnhancement))
			{
				cam_info_ptr->m_saturation_min = (f32)cam_device_ptr->SharpnessEnhancement.GetMin();
				cam_info_ptr->m_saturation_max = (f32)cam_device_ptr->SharpnessEnhancement.GetMax();
			}

			cam_info_ptr->m_max_width = width;
			cam_info_ptr->m_max_height = height;
		}
		cam_state_ptr->m_capture_width = width;
		cam_state_ptr->m_capture_height = height;
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		if (cam_device_ptr && cam_device_ptr->IsOpen())
		{
			cam_device_ptr->Close();
		}
		printlog_lvs2(QString("BaslerUsb camera init failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::exitCamera()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		if (device_ptr->IsOpen())
		{
			device_ptr->StopGrabbing();
			device_ptr->Close();
		}

		m_cam_param_ptr->m_cam_handle = -1;
		POSAFE_DELETE(device_ptr);
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		printlog_lvs2(QString("Basler pylon exit camera is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}

	return kPODevErrNone;
}

i32 CBaslerPylonCamera::play()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	try
    {
        device_ptr->StartGrabbing(Pylon::GrabStrategy_LatestImages);
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		printlog_lvs2(QString("Basler startGrabbing failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::stop()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		device_ptr->StopGrabbing();
	}
	catch (const Pylon::GenericException &e)
	{
		// Error handling
		printlog_lvs2(QString("Basler stopGrabbing failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::snapToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
									i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	has_frame = false;

	capture_width = 0;
	capture_height = 0;

	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		Pylon::CGrabResultPtr basler_grab_ptr;
		if (device_ptr->RetrieveResult(1000, basler_grab_ptr, Pylon::TimeoutHandling_Return))
		{
			if (basler_grab_ptr->GrabSucceeded())
			{
				has_frame = true;
				capture_width = basler_grab_ptr->GetWidth();
				capture_height = basler_grab_ptr->GetHeight();
				memcpy(gray_buffer_ptr, (u8*)basler_grab_ptr->GetBuffer(), capture_width* capture_height);
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler snapToBuffer failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrConnect;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::snapToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	has_frame = false;

	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		Pylon::CGrabResultPtr basler_grab_ptr;
		if (device_ptr->RetrieveResult(1000, basler_grab_ptr, Pylon::TimeoutHandling_Return))
		{
			if (basler_grab_ptr->GrabSucceeded())
			{
				i32 capture_width = basler_grab_ptr->GetWidth();
				i32 capture_height = basler_grab_ptr->GetHeight();
				ImageData raw_img_data;
				
				switch (basler_grab_ptr->GetPixelType())
				{
					case Pylon::PixelType_Mono8:
					{
						has_frame = true;
						raw_img_data.setImageData((u8*)basler_grab_ptr->GetBuffer(), capture_width, capture_height, 1);
						break;
					}
					case Pylon::PixelType_RGB8packed:
					{
						has_frame = true;
						raw_img_data.setImageData((u8*)basler_grab_ptr->GetBuffer(), capture_width, capture_height, 3);
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
				}
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler snapToBuffer failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrConnect;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::snapManualToBuffer(u8* gray_buffer_ptr, u8* rgb_buffer_ptr,
										i32& capture_width, i32& capture_height, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	has_frame = false;

	capture_width = 0;
	capture_height = 0;

	if (!m_cam_param_ptr || !gray_buffer_ptr || !rgb_buffer_ptr)
	{
		return kPODevErrException;
	}

	i32 result = kPODevErrNone;
	i32 prev_trigger_mode = kCamTriggerContinuous;
	Pylon::CInstantCamera* cam_dev_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!cam_dev_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		if (!changeBaslerTriggerMode(kCamTriggerContinuous, prev_trigger_mode))
		{
			return kPODevErrException;
		}

		Pylon::CGrabResultPtr basler_grab_ptr;
		if (cam_dev_ptr->RetrieveResult(1000, basler_grab_ptr, Pylon::TimeoutHandling_ThrowException))
		{
			if (basler_grab_ptr->GrabSucceeded())
			{
				has_frame = true;
				capture_width = basler_grab_ptr->GetWidth();
				capture_height = basler_grab_ptr->GetHeight();
				memcpy(gray_buffer_ptr, (u8*)basler_grab_ptr->GetBuffer(), capture_width*capture_height);
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		result = kPODevErrConnect;
		printlog_lvs2(QString("Basler snapManualToBuffer failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
	}

	i32 trigger_mode;
	changeBaslerTriggerMode(prev_trigger_mode, trigger_mode);
	return result;
}

i32 CBaslerPylonCamera::snapManualToBuffer(ImageData& img_data, u8* rgb_buffer_ptr, BaseFrameInfo& frame_info)
{
	bool& has_frame = frame_info.has_frame;
	has_frame = false;

	if (!m_cam_param_ptr || !rgb_buffer_ptr)
	{
		return kPODevErrException;
	}

	i32 result = kPODevErrNone;
	i32 prev_trigger_mode = kCamTriggerContinuous;
	Pylon::CInstantCamera* cam_dev_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!cam_dev_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		if (!changeBaslerTriggerMode(kCamTriggerContinuous, prev_trigger_mode))
		{
			return kPODevErrException;
		}

		Pylon::CGrabResultPtr basler_grab_ptr;
		if (cam_dev_ptr->RetrieveResult(1000, basler_grab_ptr, Pylon::TimeoutHandling_ThrowException))
		{
			if (basler_grab_ptr->GrabSucceeded())
			{
				has_frame = true;
				i32 capture_width = basler_grab_ptr->GetWidth();
				i32 capture_height = basler_grab_ptr->GetHeight();
				ImageData raw_img_data;

				switch (basler_grab_ptr->GetPixelType())
				{
					case Pylon::PixelType_Mono8:
					{
						has_frame = true;
						raw_img_data.setImageData((u8*)basler_grab_ptr->GetBuffer(), capture_width, capture_height, 1);
						break;
					}
					case Pylon::PixelType_RGB8packed:
					{
						has_frame = true;
						raw_img_data.setImageData((u8*)basler_grab_ptr->GetBuffer(), capture_width, capture_height, 3);
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
				}
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		result = kPODevErrConnect;
		printlog_lvs2(QString("Basler snapManualToBuffer failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
	}

	i32 trigger_mode;
	changeBaslerTriggerMode(prev_trigger_mode, trigger_mode);
	return result;
}

bool CBaslerPylonCamera::changeBaslerTriggerMode(i32 chk_trigger_mode, i32& prev_trigger_mode)
{
	if (!m_cam_param_ptr)
	{
		return false;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return false;
	}

	try
	{
		anlock_guard(m_dev_trigger);

		//check current trigger mode and input mode
		if (m_dev_trigger.m_trigger_mode == chk_trigger_mode)
		{
			prev_trigger_mode = chk_trigger_mode;
			return true;
		}

		//set triggermode and store prevmode
		switch (chk_trigger_mode)
		{
			case kCamTriggerContinuous:
			{
				if (!setBaslerContinuousTriggerMode(device_ptr))
				{
					printlog_lvs2("Can't change to continuous mode.", LOG_SCOPE_CAM);
					return false;
				}
				break;
			}
			case kCamTriggerManual:
			case kCamTriggerIO:
			case kCamTriggerRS:
			case kCamTriggerNetwork:
			{
				if (!setBaslerSoftwareTriggerMode(device_ptr))
				{
					printlog_lvs2("Can't change to software mode.", LOG_SCOPE_CAM);
					return false;
				}
				break;
			}
			case kCamTriggerCamera:
			{
				if (!setBaslerCameraTriggerMode(device_ptr, NULL))
				{
					printlog_lvs2("Can't change to camera trigger mode.", LOG_SCOPE_CAM);
					return false;
				}
				break;
			}
		}

		prev_trigger_mode = m_dev_trigger.m_trigger_mode;
		m_dev_trigger.m_trigger_mode = chk_trigger_mode;
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler trigger mode change is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return false;
	}
	return true;
}

bool CBaslerPylonCamera::setBaslerContinuousTriggerMode(Pylon::CInstantCamera* device_ptr)
{
	if (!device_ptr)
	{
		return false;
	}

	try
	{
		//Get required enumerations.
		GenApi::INodeMap& node_map = device_ptr->GetNodeMap();
		GenApi::CEnumerationPtr trigger_select(node_map.GetNode("TriggerSelector"));
		GenApi::CEnumerationPtr trigger_mode(node_map.GetNode("TriggerMode"));

		if (GenApi::IsAvailable(trigger_select))
		{
			//Get all enumeration entries of Trigger Selector.
			GenApi::NodeList_t trigger_select_entries;
			trigger_select->GetEntries(trigger_select_entries);

			//Turn Trigger Mode off For all Trigger Selector entries.
			GenApi::NodeList_t::iterator iter;
			for (iter = trigger_select_entries.begin(); iter != trigger_select_entries.end(); ++iter)
			{
				//Set Trigger Mode to off if the trigger is available.
				GenApi::CEnumEntryPtr entry_ptr(*iter);
				if (GenApi::IsAvailable(entry_ptr))
				{
					trigger_select->FromString(entry_ptr->GetSymbolic());
					trigger_mode->FromString("Off");
				}
			}
		}

		//Set acquisition mode.
		GenApi::CEnumerationPtr(node_map.GetNode("AcquisitionMode"))->FromString("Continuous");
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setContinuousTriggerMode failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return false;
	}
	return true;
}

bool CBaslerPylonCamera::setBaslerSoftwareTriggerMode(Pylon::CInstantCamera* device_ptr)
{
	if (!device_ptr)
	{
		return false;
	}

 	try
 	{
 		// Get required enumerations.
 		GenApi::INodeMap& node_map = device_ptr->GetNodeMap();
 		GenApi::CEnumerationPtr trigger_select(node_map.GetNode("TriggerSelector"));
 		GenApi::CEnumerationPtr trigger_mode(node_map.GetNode("TriggerMode"));
 
 		// Check the available camera trigger mode(s) to select the appropriate one: acquisition start trigger mode
 		// (used by older cameras, i.e. for cameras supporting only the legacy image acquisition control mode;
 		// do not confuse with acquisition start command) or frame start trigger mode
 		// (used by newer cameras, i.e. for cameras using the standard image acquisition control mode;
 		// equivalent to the acquisition start trigger mode in the legacy image acquisition control mode).
 		Pylon::String_t trigger_name("FrameStart");
 		if (!IsAvailable(trigger_select->GetEntryByName(trigger_name)))
 		{
 			trigger_name = "AcquisitionStart";
 			if (!IsAvailable(trigger_select->GetEntryByName(trigger_name)))
 			{
 				printlog_lvs2("Can't select trigger. Neither FrameStart nor AcquisitionStart is available.", LOG_SCOPE_CAM);
 				return false;
 			}
 		}
 
 		// Get all enumeration entries of trigger selector.
 		GenApi::NodeList_t trigger_select_entries;
 		trigger_select->GetEntries(trigger_select_entries);
 
 		// Turn trigger mode off for all trigger selector entries except for the frame trigger given by triggerName.
 		for (GenApi::NodeList_t::iterator it = trigger_select_entries.begin(); it != trigger_select_entries.end(); ++it)
 		{
 			// Set trigger mode to off if the trigger is available.
 			GenApi::CEnumEntryPtr entry_ptr(*it);
 			if (IsAvailable(entry_ptr))
 			{
 				Pylon::String_t trigger_name_entry(entry_ptr->GetSymbolic());
 				trigger_select->FromString(trigger_name_entry);
 				if (trigger_name == trigger_name_entry)
 				{
 					// Activate trigger.
 					trigger_mode->FromString("On");
					GenApi::CEnumerationPtr(node_map.GetNode("TriggerSource"))->FromString("Software");
 				}
 				else
 				{
 					trigger_mode->FromString("Off");
 				}
 			}
 		}
 		// Finally select the frame trigger type (resp. acquisition start type
 		// for older cameras). Issuing a software trigger will now trigger
 		// the acquisition of a frame.
 		trigger_select->FromString(trigger_name);
 
 		//Set acquisition mode.
 		GenApi::CEnumerationPtr(node_map.GetNode("AcquisitionMode"))->FromString("Continuous");
 	}
 	catch (const Pylon::GenericException &e)
 	{
 		printlog_lvs2(QString("Basler setBaslerSoftwareTriggerMode failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
 		return false;
 	}
	return true;
}

bool CBaslerPylonCamera::setBaslerCameraTriggerMode(Pylon::CInstantCamera* device_ptr, CameraTrigger* cam_trigger_ptr)
{
	if (!device_ptr)
	{
		return false;
	}

	try
	{
		// Get required enumerations.
		GenApi::INodeMap& node_map = device_ptr->GetNodeMap();
		GenApi::CEnumerationPtr trigger_select(node_map.GetNode("TriggerSelector"));
		GenApi::CEnumerationPtr trigger_mode(node_map.GetNode("TriggerMode"));

		// Check the available camera trigger mode(s) to select the appropriate one: acquisition start trigger mode
		// (used by older cameras, i.e. for cameras supporting only the legacy image acquisition control mode;
		// do not confuse with acquisition start command) or frame start trigger mode
		// (used by newer cameras, i.e. for cameras using the standard image acquisition control mode;
		// equivalent to the acquisition start trigger mode in the legacy image acquisition control mode).
		Pylon::String_t trigger_name("FrameStart");
		if (!IsAvailable(trigger_select->GetEntryByName(trigger_name)))
		{
			trigger_name = "AcquisitionStart";
			if (!IsAvailable(trigger_select->GetEntryByName(trigger_name)))
			{
				printlog_lvs2("Can't select trigger. Neither FrameStart nor AcquisitionStart is available.", LOG_SCOPE_CAM);
				return false;
			}
		}

		// Get all enumeration entries of trigger selector.
		GenApi::NodeList_t trigger_select_entries;
		trigger_select->GetEntries(trigger_select_entries);

		// Turn trigger mode off for all trigger selector entries except for the frame trigger given by triggerName.
		for (GenApi::NodeList_t::iterator it = trigger_select_entries.begin(); it != trigger_select_entries.end(); ++it)
		{
			// Set trigger mode to off if the trigger is available.
			GenApi::CEnumEntryPtr entry_ptr(*it);
			if (IsAvailable(entry_ptr))
			{
				Pylon::String_t trigger_name_entry(entry_ptr->GetSymbolic());
				trigger_select->FromString(trigger_name_entry);
				if (trigger_name == trigger_name_entry)
				{
					// Activate trigger.
					trigger_mode->FromString("On");

					// Alternative hardware trigger configuration:
					// This configuration can be copied and modified to create a hardware trigger configuration.
					// Remove setting the 'TriggerSource' to 'Software' (see above) and
					// use the commented lines as a starting point.
					// The camera user's manual contains more information about available configurations.
					// The Basler pylon Viewer tool can be used to test the selected settings first.

					// The trigger source must be set to the trigger input, e.g. 'Line1'.
					GenApi::CEnumerationPtr(node_map.GetNode("TriggerSource"))->FromString("Line1");

					if (cam_trigger_ptr)
					{
						/* 트리거형태를 설정한다. */
						switch (cam_trigger_ptr->m_trigger_signal)
						{
							case kCamTriggerRisingEdge:
							{
								GenApi::CEnumerationPtr(node_map.GetNode("TriggerActivation"))->FromString("RisingEdge");
								break;
							}
							case kCamTriggerFallingEdge:
							{
								GenApi::CEnumerationPtr(node_map.GetNode("TriggerActivation"))->FromString("FallingEdge");
								break;
							}
							default:
							{
								printlog_lvs2("trigger source signal is not compatibiliity", LOG_SCOPE_CAM);
								break;
							}
						}
						m_dev_trigger.m_trigger_signal = cam_trigger_ptr->m_trigger_signal;

						/* 트리거delay를 설정한다. */
						GenApi::CFloatPtr trigger_delay_abs_ptr = node_map.GetNode("TriggerDelayAbs");
						if (GenApi::IsAvailable(trigger_delay_abs_ptr))
						{
							f64 new_value = 0;
							f64 min_value = trigger_delay_abs_ptr->GetMin();
							f64 max_value = trigger_delay_abs_ptr->GetMax();
							adjust_value_f64(cam_trigger_ptr->m_trigger_delay * 1000, min_value, max_value, new_value);

							cam_trigger_ptr->m_trigger_delay = new_value / 1000;
							trigger_delay_abs_ptr->SetValue(new_value);
						}
					}
				}
				else
				{
					trigger_mode->FromString("Off");
				}
			}
		}
		// Finally select the frame trigger type (resp. acquisition start type
		// for older cameras). Issuing a software trigger will now trigger
		// the acquisition of a frame.
		trigger_select->FromString(trigger_name);

		//Set acquisition mode.
		GenApi::CEnumerationPtr(node_map.GetNode("AcquisitionMode"))->FromString("Continuous");
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler pylon can't set camera-trigger mode, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return false;
	}
	return true;
}

// Adjust value to make it comply with range and increment passed.
//
// The parameter's minimum and maximum are always considered as valid values.
// If the increment is larger than one, the returned value will be: min + (n * inc).
// If the value doesn't meet these criteria, it will be rounded down to ensure compliance.
bool CBaslerPylonCamera::adjust_value_i64(i64 cur_value, i64 min_value, i64 max_value, i64 inc_value, i64& new_value)
{
	new_value = 0;

	// Check the input parameters.
	if (inc_value <= 0)
	{
		// Negative increments are invalid.
		printlog_lvs2(QString("Unexpected increment %1").arg(inc_value), LOG_SCOPE_CAM);
		return false;
	}
	if (min_value > max_value)
	{
		// Minimum must not be bigger than or equal to the maximum.
		printlog_lvs2("minimum bigger than maximum.", LOG_SCOPE_CAM);
		return false;
	}

	if (cur_value < min_value)  // Check the lower bound.
	{
		new_value = min_value;
	}
	else if (cur_value > max_value)  // Check the upper bound.
	{
		new_value = max_value;
	}
	else
	{
		// Check the increment.
		if (inc_value == 1)
		{
			// Special case: all values are valid.
			new_value = cur_value;
		}
		else
		{
			// The value must be min + (n * inc).
			// Due to the integer division, the value will be rounded down.
			new_value = min_value + (((cur_value - min_value) / inc_value) * inc_value);
		}
	}
	return true;
}

bool CBaslerPylonCamera::adjust_value_f64(f64 cur_value, f64 min_value, f64 max_value, f64& new_value)
{
	new_value = 0;

	if (min_value > max_value)
	{
		// Minimum must not be bigger than or equal to the maximum.
		printlog_lvs2("minimum bigger than maximum.", LOG_SCOPE_CAM);
		return false;
	}

	if (cur_value < min_value)  // Check the lower bound.
	{
		new_value = min_value;
	}
	else if (cur_value > max_value)  // Check the upper bound.
	{
		new_value = max_value;
	}
	else
	{
		new_value = cur_value;
	}
	return true;
}

i32 CBaslerPylonCamera::setGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportManualExp))
	{
		return kPODevErrException;
	}

	i32 mode = 0;
	f32 gain_val = 0;
	{
		anlock_guard_ptr(cam_info_ptr);
		anlock_guard_ptr(cam_exposure_ptr);
		mode = cam_info_ptr->m_cam_reserved[0];
		gain_val = po::_max(cam_exposure_ptr->m_gain, cam_info_ptr->m_gain_min);
		gain_val = po::_min(gain_val, cam_info_ptr->m_gain_max);
	}
	try
	{
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				i64 new_value = 0;
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& gain = gige_cam_ptr->GainRaw;

				if (GenApi::IsWritable(gain) &&
					adjust_value_i64((i64)gain_val, gain.GetMin(), gain.GetMax(), gain.GetInc(), new_value))
				{
					gain.SetValue(new_value);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				f64 new_value = 0;
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IFloat& gain = usb_cam_ptr->Gain;

				if (GenApi::IsWritable(gain) &&
					adjust_value_f64(gain_val, gain.GetMin(), gain.GetMax(), new_value))
				{
					gain.SetValue(new_value);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setGain").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler pylon set gain is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportManualExp))
	{
		return kPODevErrException;
	}

	i32 mode;
	f32 exposure_val = 0;
	{
		anlock_guard_ptr(cam_info_ptr);
		anlock_guard_ptr(cam_exposure_ptr);
		mode = cam_info_ptr->m_cam_reserved[0];
		exposure_val = po::_max(cam_exposure_ptr->m_exposure, cam_info_ptr->m_exposure_min);
		exposure_val = po::_min(exposure_val, cam_info_ptr->m_exposure_max);
		exposure_val *= 1000;
	}
	try
	{
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				i64 new_value = 0;
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& exposure = gige_cam_ptr->ExposureTimeRaw;

				if (GenApi::IsWritable(exposure) &&
					adjust_value_i64((i64)exposure_val, exposure.GetMin(), exposure.GetMax(), exposure.GetInc(), new_value))
				{
					exposure.SetValue(new_value);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				f64 new_value = 0;
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IFloat& exposure = usb_cam_ptr->ExposureTime;

				if (GenApi::IsWritable(exposure) &&
					adjust_value_f64(exposure_val, exposure.GetMin(), exposure.GetMax(), new_value))
				{
					exposure.SetValue(new_value);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setExposureTimeMs").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler pylon set exposure is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setAeGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
	{
		return kPODevErrException;
	}

	f32 autogain_min, autogain_max;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		autogain_min = cam_exposure_ptr->m_autogain_min;
		autogain_max = cam_exposure_ptr->m_autogain_max;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				i64 new_value = 0;
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& gain_lower = gige_cam_ptr->AutoGainRawLowerLimit;
				GenApi::IInteger& gain_upper = gige_cam_ptr->AutoGainRawUpperLimit;

				/* Gain자동설정을 위한 최소값을 설정한다. */
				if (GenApi::IsWritable(gain_lower) && 
					adjust_value_i64((i64)autogain_min, gain_lower.GetMin(), gain_lower.GetMax(), gain_lower.GetInc(), new_value))
				{
					gain_lower.SetValue(new_value);
				}

				/* Gain자동설정을 위한 최대값을 설정한다. */
				if (GenApi::IsWritable(gain_upper) &&
					adjust_value_i64((i64)autogain_max, gain_upper.GetMin(), gain_upper.GetMax(), gain_upper.GetInc(), new_value))
				{
					gain_upper.SetValue(new_value);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				f64 new_value = 0;
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IFloat& gain_lower = usb_cam_ptr->AutoGainLowerLimit;
				GenApi::IFloat& gain_upper = usb_cam_ptr->AutoGainUpperLimit;

				/* Gain자동설정을 위한 최소값을 설정한다. */
				if (GenApi::IsWritable(gain_lower) && 
					adjust_value_f64(autogain_min, gain_lower.GetMin(), gain_lower.GetMax(), new_value))
				{
					gain_lower.SetValue(new_value);
				}

				/* Gain자동설정을 위한 최대값을 설정한다. */
				if (GenApi::IsWritable(gain_upper) &&
					adjust_value_f64(autogain_max, gain_upper.GetMin(), gain_upper.GetMax(), new_value))
				{
					gain_upper.SetValue(new_value);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setColorMode").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler set autogain is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setAeExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
	{
		return kPODevErrException;
	}

	f32 autoexp_min, autoexp_max;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		autoexp_min = cam_exposure_ptr->m_autoexp_min * 1000;
		autoexp_max = cam_exposure_ptr->m_autoexp_max * 1000;
	}

	try
	{
		f64 new_value = 0;
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IFloat& exposure_lower = gige_cam_ptr->AutoExposureTimeAbsLowerLimit;
				GenApi::IFloat& exposure_upper = gige_cam_ptr->AutoExposureTimeAbsUpperLimit;

				/* 폭광시간자동설정을 위한 최소값을 설정한다. */
				if (GenApi::IsWritable(exposure_lower) &&
					adjust_value_f64(autoexp_min, exposure_lower.GetMin(), exposure_lower.GetMax(), new_value))
				{
					exposure_lower.SetValue(new_value);
				}

				/* 폭광시간자동설정을 위한 최대값을 설정한다. */
				if (GenApi::IsWritable(exposure_upper) &&
					adjust_value_f64(autoexp_max, exposure_upper.GetMin(), exposure_upper.GetMax(), new_value))
				{
					exposure_upper.SetValue(new_value);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IFloat& exposure_lower = usb_cam_ptr->AutoExposureTimeLowerLimit;
				GenApi::IFloat& exposure_upper = usb_cam_ptr->AutoExposureTimeUpperLimit;

				/* 폭광시간자동설정을 위한 최소값을 설정한다. */
				if (GenApi::IsWritable(exposure_lower) &&
					adjust_value_f64(autoexp_min, exposure_lower.GetMin(), exposure_lower.GetMax(), new_value))
				{
					exposure_lower.SetValue(new_value);
				}

				/* 폭광시간자동설정을 위한 최대값을 설정한다. */
				if (GenApi::IsWritable(exposure_upper) &&
					adjust_value_f64(autoexp_max, exposure_upper.GetMin(), exposure_upper.GetMax(), new_value))
				{
					exposure_upper.SetValue(new_value);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setAeExposureTimeMs").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler set autoexposure is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setAeBrightness()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportAEBrightness))
	{
		return kPODevErrException;
	}

	f32 auto_brightness_val;
	{
		anlock_guard_ptr(cam_exposure_ptr);
		auto_brightness_val = cam_exposure_ptr->m_auto_brightness;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				i64 new_value = 0;
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& auto_brightness = gige_cam_ptr->AutoTargetValue;

				/* 자동폭광설정을 위한 밝기최소값을 설정한다. */
				if (GenApi::IsWritable(auto_brightness) &&
					adjust_value_i64((i64)auto_brightness_val, auto_brightness.GetMin(), auto_brightness.GetMax(),
								auto_brightness.GetInc(), new_value))
				{
					auto_brightness.SetValue(new_value);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				f64 new_value = 0;
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IFloat& auto_brightness = usb_cam_ptr->AutoTargetBrightness;

				/* 자동폭광설정을 위한 밝기최소값을 설정한다. */
				if (GenApi::IsWritable(auto_brightness) &&
					adjust_value_f64(auto_brightness_val, auto_brightness.GetMin(), auto_brightness.GetMax(), new_value))
				{
					auto_brightness.SetValue(new_value);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setAeBrightness").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler set auto-brightness is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setAeWindow()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();
	CameraExposure* cam_exposure_ptr = m_cam_param_ptr->getCameraExposure();
	if (!cam_device_ptr || !m_cam_param_ptr->supportFunc(kCamSupportAEWindow))
	{
		return kPODevErrException;
	}

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
		Recti range = cam_range_ptr->getRange();
		x = range.x1;
		y = range.y1;
		w = range.getWidth();
		h = range.getHeight();
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				i64 new_left = 0;
				i64 new_top = 0;
				i64 new_width = 0;
				i64 new_height = 0;

				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& aoi_offset_x = gige_cam_ptr->AutoFunctionAOIOffsetX;
				GenApi::IInteger& aoi_offset_y = gige_cam_ptr->AutoFunctionAOIOffsetY;
				GenApi::IInteger& aoi_width = gige_cam_ptr->AutoFunctionAOIWidth;
				GenApi::IInteger& aoi_height = gige_cam_ptr->AutoFunctionAOIHeight;

				if (!GenApi::IsWritable(aoi_offset_x) || !GenApi::IsWritable(aoi_offset_y) || 
					!GenApi::IsWritable(aoi_width) || !GenApi::IsWritable(aoi_height))
				{
					return kPODevErrException;
				}
				if (!adjust_value_i64(x, aoi_offset_x.GetMin(), aoi_offset_x.GetMax(), aoi_offset_x.GetInc(), new_left) || 
					!adjust_value_i64(y, aoi_offset_y.GetMin(), aoi_offset_y.GetMax(), aoi_offset_y.GetInc(), new_top) || 
					!adjust_value_i64(w, aoi_width.GetMin(), aoi_width.GetMax(), aoi_width.GetInc(), new_width) || 
					!adjust_value_i64(h, aoi_height.GetMin(), aoi_height.GetMax(), aoi_height.GetInc(), new_height))
				{
					return kPODevErrException;
				}

				aoi_width.SetValue(new_width);
				aoi_height.SetValue(new_height);
				aoi_offset_x.SetValue(new_left);
				aoi_offset_y.SetValue(new_top);
				break;
			}
			case kPOCamBaslerUsb:
			{
				i64 new_left = 0;
				i64 new_top = 0;
				i64 new_width = 0;
				i64 new_height = 0;

				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IInteger& roi_offset_x = usb_cam_ptr->AutoFunctionROIOffsetX;
				GenApi::IInteger& roi_offset_y = usb_cam_ptr->AutoFunctionROIOffsetY;
				GenApi::IInteger& roi_width = usb_cam_ptr->AutoFunctionROIWidth;
				GenApi::IInteger& roi_height = usb_cam_ptr->AutoFunctionROIHeight;

				if (!GenApi::IsWritable(roi_offset_x) || !GenApi::IsWritable(roi_offset_y) ||
					!GenApi::IsWritable(roi_width) || !GenApi::IsWritable(roi_height))
				{
					return kPODevErrException;
				}
				if (!adjust_value_i64(x, roi_offset_x.GetMin(), roi_offset_x.GetMax(), roi_offset_x.GetInc(), new_left) || 
					!adjust_value_i64(y, roi_offset_y.GetMin(), roi_offset_y.GetMax(), roi_offset_y.GetInc(), new_top) || 
					!adjust_value_i64(w, roi_width.GetMin(), roi_width.GetMax(), roi_width.GetInc(), new_width) || 
					!adjust_value_i64(h, roi_height.GetMin(), roi_height.GetMax(), roi_height.GetInc(), new_height))
				{
					return kPODevErrException;
				}

				roi_width.SetValue(new_width);
				roi_height.SetValue(new_height);
				roi_offset_x.SetValue(new_left);
				roi_offset_y.SetValue(new_top);
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setAeWindow").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler set auto-window is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setAeState(const i32 autoexp_mode)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);

		switch (autoexp_mode)
		{
			case kCamAEModeContinuous:
			{
				if (!m_cam_param_ptr->supportFunc(kCamSupportAutoExp))
				{
					return kPODevErrException;
				}

				switch (mode)
				{
					case kPOCamBaslerGigE:
					{
						Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
						if (!GenApi::IsWritable(gige_cam_ptr->GainAuto) || !GenApi::IsWritable(gige_cam_ptr->ExposureAuto))
						{
							return kPODevErrException;
						}
						gige_cam_ptr->GainAuto.SetValue(Basler_GigECamera::GainAuto_Continuous);
						gige_cam_ptr->ExposureAuto.SetValue(Basler_GigECamera::ExposureAuto_Continuous);
						break;
					}
					case kPOCamBaslerUsb:
					{
						Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
						if (!GenApi::IsWritable(usb_cam_ptr->GainAuto) || !GenApi::IsWritable(usb_cam_ptr->ExposureAuto))
						{
							return kPODevErrException;
						}
						usb_cam_ptr->GainAuto.SetValue(Basler_UsbCameraParams::GainAuto_Continuous);
						usb_cam_ptr->ExposureAuto.SetValue(Basler_UsbCameraParams::ExposureAuto_Continuous);
						break;
					}
					default:
					{
						printlog_lvs2(QString("Unknown BaslerType[%1] in setAeState").arg(mode), LOG_SCOPE_CAM);
						break;
					}
				}
				break;
			}
			case kCamAEModeOff:
			{
				if (!m_cam_param_ptr->supportFunc(kCamSupportManualExp))
				{
					return kPODevErrException;
				}

				switch (mode)
				{
					case kPOCamBaslerGigE:
					{
						Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
						if (!GenApi::IsWritable(gige_cam_ptr->GainAuto) || !GenApi::IsWritable(gige_cam_ptr->ExposureAuto))
						{
							return kPODevErrException;
						}
						gige_cam_ptr->GainAuto.SetValue(Basler_GigECamera::GainAuto_Off);
						gige_cam_ptr->ExposureAuto.SetValue(Basler_GigECamera::ExposureAuto_Off);
						break;
					}
					case kPOCamBaslerUsb:
					{
						Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
						if (!GenApi::IsWritable(usb_cam_ptr->GainAuto) || !GenApi::IsWritable(usb_cam_ptr->ExposureAuto))
						{
							return kPODevErrException;
						}
						usb_cam_ptr->GainAuto.SetValue(Basler_UsbCameraParams::GainAuto_Off);
						usb_cam_ptr->ExposureAuto.SetValue(Basler_UsbCameraParams::ExposureAuto_Off);
						break;
					}
					default:
					{
						printlog_lvs2(QString("Unknown BaslerType[%1] in setAeState").arg(mode), LOG_SCOPE_CAM);
						break;
					}
				}
				
				if (setGain() != kPODevErrNone || setExposureTimeMs() != kPODevErrNone)
				{
					return kPODevErrException;
				}
				break;
			}
			default:
			{
				return kPODevErrException;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler set ae-mode is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setTriggerMode(CameraTrigger& cam_trigger)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	anlock_guard(cam_trigger);
	anlock_guard(m_dev_trigger);
	i32 trigger_mode = cam_trigger.m_trigger_mode;

	/* 촉발방식에 따라 트리거스캔시간을 결정한다. */
	switch (trigger_mode)
	{
		case kCamTriggerContinuous:
		{
			m_dev_trigger.m_trigger_interval = cam_trigger.m_trigger_interval;
			break;
		}
		case kCamTriggerCamera:
		default:
		{
			m_dev_trigger.m_trigger_interval = 1; //ms
			break;
		}
	}

	/* 트리거정보를 설정한다. */
	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}

	switch (trigger_mode)
	{
		//set free-trigger mode
		case kCamTriggerContinuous:
		{
			if (m_dev_trigger.m_trigger_mode != kCamTriggerContinuous)
			{
				setBaslerContinuousTriggerMode(device_ptr);
				m_dev_trigger.m_trigger_mode = kCamTriggerContinuous;
			}
			break;
		}

		//set software-trigger mode
		case kCamTriggerManual:
		case kCamTriggerIO:
		case kCamTriggerRS:
		case kCamTriggerNetwork:
		{
			if (m_dev_trigger.m_trigger_mode != kCamTriggerContinuous)
			{
				setBaslerSoftwareTriggerMode(device_ptr);
				m_dev_trigger.m_trigger_mode = kCamTriggerManual;
			}
			break;
		}

		//set external-trigger mode
		case kCamTriggerCamera:
		{
			if (m_dev_trigger.m_trigger_mode != kCamTriggerCamera)
			{
				setBaslerCameraTriggerMode(device_ptr, &cam_trigger);
				m_dev_trigger.m_trigger_mode = kCamTriggerCamera;
			}
			break;
		}
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCaptureInvert()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}
	//TODO: Not Implemented
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCaptureFlip()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}
	//TODO: Not Implemented
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCaptureRotation()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* device_ptr = (Pylon::CInstantCamera*)(m_cam_param_ptr->getCamDevice());
	if (!device_ptr)
	{
		return kPODevErrException;
	}
	//TODO: Not Implemented
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCaptureRange()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportRange))
	{
		return kPODevErrUnsupport;
	}

	stop();
	i32 code = setCaptureRangeInternal();
	play();
	return code;
}

i32 CBaslerPylonCamera::setCaptureRangeInternal()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportRange))
	{
		return kPODevErrUnsupport;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraRange* cam_range_ptr = m_cam_param_ptr->getCameraRange();

	i32 x, y, w, h;
	i32 max_width, max_height, mode;
	Recti range = cam_range_ptr->getRange();
	{
		anlock_guard_ptr(cam_info_ptr);
		max_width = cam_info_ptr->m_max_width;
		max_height = cam_info_ptr->m_max_height;
		mode = cam_info_ptr->m_cam_reserved[0];
	}

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
	i32 NX = 8;
	i32 NY = 4; //자체추가함
	if (x != 0 || y != 0 || w != max_width || h != max_height)
	{
		w = CPOBase::round(w, NX);
		h = CPOBase::round(h, NY);
		if (x + w > max_width)
		{
			w = range.getWidth() / NX * NX;
		}
		if (y + h > max_height)
		{
			h = range.getHeight() / NY * NY;
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
	
	try
	{
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				GenApi::IInteger& offset_x = gige_cam_ptr->OffsetX;
				GenApi::IInteger& offset_y = gige_cam_ptr->OffsetY;
				GenApi::IInteger& width = gige_cam_ptr->Width;
				GenApi::IInteger& height = gige_cam_ptr->Height;

				if (!GenApi::IsWritable(offset_x) || !GenApi::IsWritable(offset_y) ||
					!GenApi::IsWritable(width) || !GenApi::IsWritable(height))
				{
					return kPODevErrException;
				}

				width.SetValue(w);
				height.SetValue(h);
				offset_x.SetValue(x);
				offset_y.SetValue(y);
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				GenApi::IInteger& offset_x = usb_cam_ptr->OffsetX;
				GenApi::IInteger& offset_y = usb_cam_ptr->OffsetY;
				GenApi::IInteger& width = usb_cam_ptr->Width;
				GenApi::IInteger& height = usb_cam_ptr->Height;

				width.SetValue(w);
				height.SetValue(h);
				offset_x.SetValue(x);
				offset_y.SetValue(y);
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setCaptureRange").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setCaptureRange failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setEmuSampler(void* emu_sample_ptr)
{
	return kPODevErrNone;
}

bool CBaslerPylonCamera::needTriggerScan()
{
	return (m_dev_trigger.getTriggerMode() == kCamTriggerContinuous);
}

i32 CBaslerPylonCamera::getTriggerInterval()
{
	if (!needTriggerScan())
	{
		return 0;
	}
	return m_dev_trigger.getTriggerInterval();
}

f32 CBaslerPylonCamera::getGain()
{
	if (!m_cam_param_ptr)
	{
		return 0;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr)
	{
		return 0;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(gige_cam_ptr->GainRaw))
				{
					return 0;
				}
				return (f32)gige_cam_ptr->GainRaw.GetValue();
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(usb_cam_ptr->Gain))
				{
					return 0;
				}
				return (f32)usb_cam_ptr->Gain.GetValue();
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in getGain").arg(mode), LOG_SCOPE_CAM);
				return 0;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
        printlog_lvs2(QString("Basler getGain is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return 0;
	}
	return 0;
}

f32 CBaslerPylonCamera::getExposureTimeMs()
{
	if (!m_cam_param_ptr)
	{
		return 0;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr)
	{
		return 0;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(gige_cam_ptr->ExposureTimeRaw))
				{
					return 0;
				}
				return (f32)gige_cam_ptr->ExposureTimeRaw.GetValue() / 1000;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(usb_cam_ptr->ExposureTime))
				{
					return 0;
				}
				return (f32)usb_cam_ptr->ExposureTime.GetValue() / 1000;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in getExposureTimeMs").arg(mode), LOG_SCOPE_CAM);
				return 0;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
        printlog_lvs2(QString("Basler getExposureTimeMs is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return 0;
	}
	return 0;
}

i32 CBaslerPylonCamera::getCameraState(bool& autoexp_mode, f32& gain, f32& expsoure_time_ms)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(gige_cam_ptr->GainAuto) ||
					!GenApi::IsReadable(gige_cam_ptr->ExposureAuto) || 
					!GenApi::IsReadable(gige_cam_ptr->GainRaw) || 
					!GenApi::IsReadable(gige_cam_ptr->ExposureTimeRaw))
				{
					return kPODevErrException;
				}

				bool is_gain_auto = (gige_cam_ptr->GainAuto.GetCurrentEntry()->GetSymbolic() == "Continuous");
				bool is_exp_auto = (gige_cam_ptr->ExposureAuto.GetCurrentEntry()->GetSymbolic() == "Continuous");
				autoexp_mode = is_gain_auto && is_exp_auto;

				gain = (f32)gige_cam_ptr->GainRaw.GetValue();
				expsoure_time_ms = (f32)gige_cam_ptr->ExposureTimeRaw.GetValue() / 1000;
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (!GenApi::IsReadable(usb_cam_ptr->GainAuto) || 
					!GenApi::IsReadable(usb_cam_ptr->ExposureAuto) || 
					!GenApi::IsReadable(usb_cam_ptr->Gain) || 
					!GenApi::IsReadable(usb_cam_ptr->ExposureTime))
				{
					return kPODevErrException;
				}

				bool is_gain_auto = (usb_cam_ptr->GainAuto.GetCurrentEntry()->GetSymbolic() == "Continuous");
				bool is_exp_auto = (usb_cam_ptr->ExposureAuto.GetCurrentEntry()->GetSymbolic() == "Continuous");
				autoexp_mode = is_gain_auto && is_exp_auto;

				gain = (f32)usb_cam_ptr->Gain.GetValue();
				expsoure_time_ms = (f32)usb_cam_ptr->ExposureTime.GetValue() / 1000;
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in getCameraState").arg(mode), LOG_SCOPE_CAM);
				return 0;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
        printlog_lvs2(QString("Basler getCameraState is failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::getCameraColorState(f32& rgain, f32& ggain, f32& bgain)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr)
	{
		return kPODevErrException;
	}

	rgain = 0;
	ggain = 0;
	bgain = 0;
	//TODO: Not Implemented
	printlog_lvs3("Basler getCameraColorState isn't available", LOG_SCOPE_CAM);
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::snapSoftwareTrigger()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	Pylon::CInstantCamera* cam_device_ptr = (Pylon::CInstantCamera*)m_cam_param_ptr->getCamDevice();
	if (!cam_device_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		if (!m_cam_param_ptr->supportFunc(kCamSupportWaitTriggerReady))
		{
			cam_device_ptr->WaitForFrameTriggerReady(500, Pylon::TimeoutHandling_Return);
		}
		cam_device_ptr->ExecuteSoftwareTrigger();
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler snapSoftwareTrigger failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setShutterMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!CPOBase::bitCheck(cam_info_ptr->getCapability(), kCamSupportShutter))
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	if (!cam_device_ptr || !cam_spec_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->ShutterMode))
				{
					if (cam_spec_ptr->getShutterMode() == kShutterRolling)
					{
						gige_cam_ptr->ShutterMode.SetValue(Basler_GigECamera::ShutterMode_Rolling);
					}
					else
					{
						if (IsAvailable(gige_cam_ptr->ShutterMode.GetEntry(Basler_GigECamera::ShutterMode_Global)))
						{
							gige_cam_ptr->ShutterMode.SetValue(Basler_GigECamera::ShutterMode_Global);
						}
						else
						{
							gige_cam_ptr->ShutterMode.SetValue(Basler_GigECamera::ShutterMode_GlobalResetRelease);
						}
					}
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->SensorShutterMode))
				{
					if (cam_spec_ptr->getShutterMode() == kShutterRolling)
					{
						usb_cam_ptr->SensorShutterMode.SetValue(Basler_UsbCameraParams::SensorShutterMode_Rolling);
					}
					else
					{
						if (IsAvailable(usb_cam_ptr->SensorShutterMode.GetEntry(Basler_UsbCameraParams::SensorShutterMode_Global)))
						{
							usb_cam_ptr->SensorShutterMode.SetValue(Basler_UsbCameraParams::SensorShutterMode_Global);
						}
						else
						{
							usb_cam_ptr->SensorShutterMode.SetValue(Basler_UsbCameraParams::SensorShutterMode_GlobalReset);
						}
					}
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setShutterMode").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setShutterMode failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setShutterJitterTime()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	if (!cam_device_ptr || !cam_info_ptr || !cam_spec_ptr)
	{
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setStrobeEnabled()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraStrobe* cam_strobe_ptr = m_cam_param_ptr->getCameraStrobe();
	if (!cam_device_ptr || !cam_info_ptr || !cam_strobe_ptr)
	{
		return kPODevErrException;
	}
	if (!CPOBase::bitCheck(cam_info_ptr->getCapability(), kCamSupportAutoStrobe))
	{
		return kPODevErrUnsupport;
	}

	i32 mode = cam_info_ptr->getReserved(0);
	switch (mode)
	{
		case kPOCamBaslerGigE:
		{
			Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
			if (GenApi::IsWritable(gige_cam_ptr->LineSource))
			{
				gige_cam_ptr->LineSelector.SetValue(Basler_GigECamera::LineSelector_Out1);
				gige_cam_ptr->LineMode.SetValue(Basler_GigECamera::LineMode_Output);
				gige_cam_ptr->LineSource.SetValue(cam_strobe_ptr->isEnabled() ? 
											Basler_GigECamera::LineSource_FlashWindow : 
											Basler_GigECamera::LineSource_UserOutput);
			}
			break;
		}
		case kPOCamBaslerUsb:
		{
			Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
			if (GenApi::IsWritable(usb_cam_ptr->LineSource))
			{
				usb_cam_ptr->LineSelector.SetValue(Basler_UsbCameraParams::LineSelector_Line2);
				usb_cam_ptr->LineMode.SetValue(Basler_UsbCameraParams::LineMode_Output);
				usb_cam_ptr->LineSource.SetValue(cam_strobe_ptr->isEnabled() ? 
											Basler_UsbCameraParams::LineSource_FlashWindow :
											Basler_UsbCameraParams::LineSource_UserOutput1);
			}
			break;
		}
		default:
		{
			printlog_lvs2(QString("Unknown BaslerType[%1] in setStrobeControl").arg(mode), LOG_SCOPE_CAM);
			break;
		}
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setStrobeControl()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraStrobe* cam_strobe_ptr = m_cam_param_ptr->getCameraStrobe();
	if (!cam_device_ptr || !cam_strobe_ptr || !cam_info_ptr)
	{
		return kPODevErrException;
	}
	return kPODevErrUnsupport;
}

i32 CBaslerPylonCamera::setLightForTrigger(bool use_strobe)
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!CPOBase::bitCheck(cam_info_ptr->getCapability(), kCamSupportManualStrobe))
	{
		return kPODevErrUnsupport;
	}
	printlog_lvs3("Basler setLightForTrigger isn't available", LOG_SCOPE_CAM);
	return kPODevErrUnsupport;
}

i32 CBaslerPylonCamera::setColorMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();
	if (!cam_device_ptr || !cam_info_ptr || !cam_color_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		//set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (cam_color_ptr->getColorMode())
		{
			case kCamColorGray:
			{
				if (!m_cam_param_ptr->supportFunc(kCamSupportGray))
				{
					return kPODevErrException;
				}

				switch (mode)
				{
					case kPOCamBaslerGigE:
					{
						Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(gige_cam_ptr->PixelFormat))
						{
							gige_cam_ptr->PixelFormat.SetValue(Basler_GigECamera::PixelFormat_Mono8);
						}
						break;
					}
					case kPOCamBaslerUsb:
					{
						Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(usb_cam_ptr->PixelFormat))
						{
							usb_cam_ptr->PixelFormat.SetValue(Basler_UsbCameraParams::PixelFormat_Mono8);
						}
						break;
					}
					default:
					{
						printlog_lvs2(QString("Unknown BaslerType[%1] in setColorMode").arg(mode), LOG_SCOPE_CAM);
						break;
					}
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

				switch (mode)
				{
					case kPOCamBaslerGigE:
					{
						Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(gige_cam_ptr->PixelFormat))
						{
							gige_cam_ptr->PixelFormat.SetValue(Basler_GigECamera::PixelFormat_RGB8Packed);
						}
						break;
					}
					case kPOCamBaslerUsb:
					{
						Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(usb_cam_ptr->PixelFormat))
						{
							usb_cam_ptr->PixelFormat.SetValue(Basler_UsbCameraParams::PixelFormat_RGB8);
						}
						break;
					}
					default:
					{
						printlog_lvs2(QString("Unknown BaslerType[%1] in setColorMode").arg(mode), LOG_SCOPE_CAM);
						break;
					}
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

				switch (mode)
				{
					case kPOCamBaslerGigE:
					{
						Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(gige_cam_ptr->PixelFormat))
						{
							gige_cam_ptr->PixelFormat.SetValue(Basler_GigECamera::PixelFormat_YUV422Packed);
						}
						break;
					}
					case kPOCamBaslerUsb:
					{
						Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
						if (GenApi::IsWritable(usb_cam_ptr->PixelFormat))
						{
							usb_cam_ptr->PixelFormat.SetValue(Basler_UsbCameraParams::PixelFormat_YCbCr422_8);
						}
						break;
					}
					default:
					{
						printlog_lvs2(QString("Unknown BaslerType[%1] in setColorMode").arg(mode), LOG_SCOPE_CAM);
						break;
					}
				}
				//update color mode
				cam_color_ptr->setColorMode(kCamColorRGB8);
				break;
			}
			default:
			{
				return kPODevErrException;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setColorMode failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setColorGain()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	assert(false);
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setWhiteBalanceMode()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	
	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraColor* cam_color_ptr = m_cam_param_ptr->getCameraColor();
	if (!cam_device_ptr || !cam_info_ptr || !cam_color_ptr)
	{
		return kPODevErrException;
	}
	bool wb_mode;
	{
		anlock_guard_ptr(cam_color_ptr);
		wb_mode = cam_color_ptr->m_wb_mode;
	}
	
	try
	{
		//set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->AutoFunctionAOIUsageWhiteBalance) &&
					GenApi::IsWritable(gige_cam_ptr->BalanceWhiteAuto))
				{
					gige_cam_ptr->AutoFunctionAOIUsageWhiteBalance.SetValue(wb_mode);
					gige_cam_ptr->BalanceWhiteAuto.SetValue(wb_mode ? Basler_GigECamera::BalanceWhiteAuto_Off :
														Basler_GigECamera::BalanceWhiteAuto_Continuous);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->AutoFunctionAOIUseWhiteBalance) &&
					GenApi::IsWritable(usb_cam_ptr->BalanceWhiteAuto))
				{
					usb_cam_ptr->AutoFunctionAOIUseWhiteBalance.SetValue(wb_mode);
					usb_cam_ptr->BalanceWhiteAuto.SetValue(wb_mode ? Basler_UsbCameraParams::BalanceWhiteAuto_Off :
														Basler_UsbCameraParams::BalanceWhiteAuto_Continuous);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setWhiteBalanceMode").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setWhiteBalanceMode failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setColorAWBOnce()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	if (!cam_device_ptr || !cam_info_ptr)
	{
		return kPODevErrException;
	}

	try
	{
		//set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->AutoFunctionAOIUsageWhiteBalance) &&
					GenApi::IsWritable(gige_cam_ptr->BalanceWhiteAuto))
				{
					gige_cam_ptr->AutoFunctionAOIUsageWhiteBalance.SetValue(true);
					gige_cam_ptr->BalanceWhiteAuto.SetValue(Basler_GigECamera::BalanceWhiteAuto_Once);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->AutoFunctionAOIUseWhiteBalance) &&
					GenApi::IsWritable(usb_cam_ptr->BalanceWhiteAuto))
				{
					usb_cam_ptr->AutoFunctionAOIUseWhiteBalance.SetValue(true);
					usb_cam_ptr->BalanceWhiteAuto.SetValue(Basler_UsbCameraParams::BalanceWhiteAuto_Once);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setColorAWBOnce").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setColorAWBOnce failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCorrectionGamma()
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
	f32 gamma;
	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	if (!cam_device_ptr || !cam_info_ptr || !cam_corr_ptr)
	{
		return kPODevErrException;
	}
	{
		anlock_guard_ptr(cam_corr_ptr);
		gamma = cam_corr_ptr->m_gamma;
	}
	
	try
	{
		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->Gamma))
				{
					gige_cam_ptr->Gamma.SetValue(gamma);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->Gamma))
				{
					usb_cam_ptr->Gamma.SetValue(gamma);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setCorrectionGamma").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setCorrectionGamma failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCorrectionContrast()
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
	f32 contrast;
	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	if (!cam_device_ptr || !cam_info_ptr || !cam_corr_ptr)
	{
		return kPODevErrException;
	}
	{
		anlock_guard_ptr(cam_corr_ptr);
		contrast = (f32)cam_corr_ptr->m_contrast;
	}

	try
	{
		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->BslContrast))
				{
					gige_cam_ptr->BslContrast.SetValue(contrast);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->BslContrast))
				{
					usb_cam_ptr->BslContrast.SetValue(contrast);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setCorrectionContrast").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setCorrectionContrast failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCorrectionSaturation()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->isColorMode() || !m_cam_param_ptr->supportFunc(kCamSupportSaturation))
	{
		return kPODevErrUnsupport;
	}

	// set mediatype and image datatype
	f32 saturation;
	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	if (!cam_device_ptr || !cam_info_ptr || !cam_corr_ptr)
	{
		return kPODevErrException;
	}
	{
		anlock_guard_ptr(cam_corr_ptr);
		saturation = (f32)cam_corr_ptr->m_saturation;
	}

	try
	{
		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->BslSaturation))
				{
					gige_cam_ptr->BslSaturation.SetValue(saturation);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->BslSaturation))
				{
					usb_cam_ptr->BslSaturation.SetValue(saturation);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setCorrectionSaturation").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setCorrectionSaturation failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setCorrectionSharpness()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportSharpness))
	{
		return kPODevErrUnsupport;
	}

	f32 sharpness;
	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraCorrection* cam_corr_ptr = m_cam_param_ptr->getCameraCorrection();
	if (!cam_device_ptr || !cam_info_ptr || !cam_corr_ptr)
	{
		return kPODevErrException;
	}
	{
		anlock_guard_ptr(cam_corr_ptr);
		sharpness = (f32)cam_corr_ptr->m_sharpness;
	}

	try
	{
		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->SharpnessEnhancementAbs))
				{
					gige_cam_ptr->SharpnessEnhancementAbs.SetValue(sharpness);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->SharpnessEnhancement))
				{
					usb_cam_ptr->SharpnessEnhancement.SetValue(sharpness);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setCorrectionSharpness").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setCorrectionSharpness failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setNoiseReduction()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	if (!m_cam_param_ptr->supportFunc(kCamSupportNoiseReduction))
	{
		return kPODevErrUnsupport;
	}

	void* cam_device_ptr = m_cam_param_ptr->getCamDevice();
	CameraInfo* cam_info_ptr = m_cam_param_ptr->getCameraInfo();
	CameraSpec* cam_spec_ptr = m_cam_param_ptr->getCameraSpec();
	if (!cam_device_ptr || !cam_info_ptr || !cam_spec_ptr)
	{
		return kPODevErrException;
	}
	
	try
	{
		bool noise_reduction = false;
		{
			anlock_guard_ptr(cam_spec_ptr);
			noise_reduction = cam_spec_ptr->m_noise_reduce;
		}

		// set mediatype and image datatype
		i32 mode = cam_info_ptr->getReserved(0);
		switch (mode)
		{
			case kPOCamBaslerGigE:
			{
				Pylon::CBaslerGigEInstantCamera* gige_cam_ptr = (Pylon::CBaslerGigEInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(gige_cam_ptr->PgiMode))
				{
					gige_cam_ptr->PgiMode.SetValue(noise_reduction ? 
											Basler_GigECamera::PgiMode_On : Basler_GigECamera::PgiMode_Off);
				}
				break;
			}
			case kPOCamBaslerUsb:
			{
				Pylon::CBaslerUsbInstantCamera* usb_cam_ptr = (Pylon::CBaslerUsbInstantCamera*)cam_device_ptr;
				if (GenApi::IsWritable(usb_cam_ptr->PgiMode))
				{
					usb_cam_ptr->PgiMode.SetValue(noise_reduction ? 
											Basler_UsbCameraParams::PgiMode_On: Basler_UsbCameraParams::PgiMode_Off);
				}
				break;
			}
			default:
			{
				printlog_lvs2(QString("Unknown BaslerType[%1] in setNoiseReduction").arg(mode), LOG_SCOPE_CAM);
				break;
			}
		}
	}
	catch (const Pylon::GenericException &e)
	{
		printlog_lvs2(QString("Basler setNoiseReduction failed, %1").arg(e.GetDescription()), LOG_SCOPE_CAM);
		return kPODevErrException;
	}
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::setColorTemperature()
{
	if (!m_cam_param_ptr)
	{
		return kPODevErrException;
	}
	//TODO: Not Implemented
	printlog_lvs3("Basler setColorTemperature isn't available", LOG_SCOPE_CAM);
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::saveCameraParamToFile(const postring& cam_param_file)
{
	//TODO: Not Implemented
	printlog_lvs3("Basler saveCameraParamToFile isn't available", LOG_SCOPE_CAM);
	return kPODevErrNone;
}

i32 CBaslerPylonCamera::loadCameraParamToFile(const postring& cam_param_file)
{
	//TODO: Not Implemented
	printlog_lvs3("Basler loadCameraParamToFile isn't available", LOG_SCOPE_CAM);
	return kPODevErrNone;
}
#endif
#endif

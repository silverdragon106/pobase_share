#include "camera_setting.h"
#include "base.h"

//////////////////////////////////////////////////////////////////////////
CameraDevice::CameraDevice()
{
	m_cam_name = "";
	m_cam_type = kPOCamUnknown;
	memset(m_cam_reserved, 0, sizeof(m_cam_reserved));
}

CameraDevice::~CameraDevice()
{
	m_cam_blob.freeBuffer();
}

//////////////////////////////////////////////////////////////////////////
CameraTrigger::CameraTrigger()
{
	reset();
}

void CameraTrigger::reset()
{
	m_trigger_mode = kCamTriggerContinuous;
	m_trigger_delay = 0;
	m_trigger_interval = kCamSnapInterval;
	m_trigger_signal = kCamTriggerRisingEdge;
	m_trigger_unit = kCamTriggerUnitMs;
}

void CameraTrigger::init()
{
	lock_guard();
	reset();
}

i32 CameraTrigger::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_trigger_mode);
	len += sizeof(m_trigger_delay);
	len += sizeof(m_trigger_interval);
	len += sizeof(m_trigger_signal);
	len += sizeof(m_trigger_unit);
	return len;
}

i32 CameraTrigger::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_trigger_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_trigger_delay, buffer_ptr, buffer_size);
	CPOBase::memRead(m_trigger_interval, buffer_ptr, buffer_size);
	CPOBase::memRead(m_trigger_signal, buffer_ptr, buffer_size);
	CPOBase::memRead(m_trigger_unit, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraTrigger::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_trigger_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_trigger_delay, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_trigger_interval, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_trigger_signal, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_trigger_unit, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraTrigger::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_trigger_mode, fp);
	CPOBase::fileWrite(m_trigger_delay, fp);
	CPOBase::fileWrite(m_trigger_interval, fp);
	CPOBase::fileWrite(m_trigger_signal, fp);
	CPOBase::fileWrite(m_trigger_unit, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

bool CameraTrigger::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_trigger_mode, fp);
	CPOBase::fileRead(m_trigger_delay, fp);
	CPOBase::fileRead(m_trigger_interval, fp);
	CPOBase::fileRead(m_trigger_signal, fp);
	CPOBase::fileRead(m_trigger_unit, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraTrigger::isEqual(CameraTrigger& other)
{
	lock_guard();
	anlock_guard(other);
	if (m_trigger_mode == kCamTriggerContinuous)
	{
		return (m_trigger_interval == other.m_trigger_interval &&
				m_trigger_unit == other.m_trigger_unit && 
				other.m_trigger_mode == kCamTriggerContinuous);
	}
	else if (m_trigger_mode == kCamTriggerCamera)
	{
		return (m_trigger_mode == kCamTriggerCamera && 
				m_trigger_unit == other.m_trigger_unit && 
				m_trigger_delay == other.m_trigger_delay && 
				m_trigger_signal == other.m_trigger_signal);
	}

	return (m_trigger_mode == other.m_trigger_mode && 
			m_trigger_unit == other.m_trigger_unit && 
			m_trigger_delay == other.m_trigger_delay);
}

bool CameraTrigger::isSoftTrigger()
{
	lock_guard();
	return CPOBase::checkRange(m_trigger_mode, kCamTriggerManual, kCamTriggerModeCount);
}

bool CameraTrigger::isManualTrigger()
{
	lock_guard();
	return (m_trigger_mode == kCamTriggerManual);
}

bool CameraTrigger::isExtTrigger()
{
	lock_guard();
	return CPOBase::checkRange(m_trigger_mode, kCamTriggerIO, kCamTriggerModeCount);
};

CameraTrigger  CameraTrigger::getValue()
{
	lock_guard();
	return *this;
}

void CameraTrigger::setValue(CameraTrigger& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

//////////////////////////////////////////////////////////////////////////
CameraExposure::CameraExposure()
{
	reset();
}

void CameraExposure::reset()
{
	m_autoexp_mode = kCamAEModeContinuous;
	m_autogain_min = 0;
	m_autogain_max = 100;
	m_autoexp_min = 0;
	m_autoexp_max = 120;
	m_auto_brightness = 120;
	m_autoexp_window.reset();

	m_gain = 15.5f;
	m_exposure = 40.0f;
}

void CameraExposure::init()
{
	lock_guard();
	reset();
}

void CameraExposure::updateValidation(CameraInfo* cam_info_ptr)
{
	if (!cam_info_ptr)
	{
		return;
	}

	//update gain, exposure, auto brightness setting
	if (!CPOBase::checkRange(m_gain, cam_info_ptr->m_gain_min, cam_info_ptr->m_gain_max))
	{
		m_gain = (cam_info_ptr->m_gain_min + cam_info_ptr->m_gain_max) / 2;
	}
	if (!CPOBase::checkRange(m_exposure, cam_info_ptr->m_exposure_min, cam_info_ptr->m_exposure_max))
	{
		m_exposure = (cam_info_ptr->m_exposure_min + cam_info_ptr->m_exposure_max) / 2;
	}
	if (!CPOBase::checkRange(m_auto_brightness, cam_info_ptr->m_brightness_min, cam_info_ptr->m_brightness_max))
	{
		m_auto_brightness = (cam_info_ptr->m_brightness_min + cam_info_ptr->m_brightness_max) / 2;
	}

	//check gain, exposure range
	if (!CPOBase::checkRange(m_autoexp_min, cam_info_ptr->m_exposure_min, cam_info_ptr->m_exposure_max))
	{
		m_autoexp_min = cam_info_ptr->m_exposure_min;
	}
	if (!CPOBase::checkRange(m_autoexp_max, m_autoexp_min, cam_info_ptr->m_exposure_max))
	{
		m_autoexp_max = cam_info_ptr->m_exposure_max;
	}
	if (!CPOBase::checkRange(m_autogain_min, cam_info_ptr->m_gain_min, cam_info_ptr->m_gain_max))
	{
		m_autogain_min = cam_info_ptr->m_gain_min;
	}
	if (!CPOBase::checkRange(m_autogain_max, m_autogain_min, cam_info_ptr->m_gain_max))
	{
		m_autogain_max = cam_info_ptr->m_gain_max;
	}

	//check auto ROI window
	i32 max_w = cam_info_ptr->m_max_width;
	i32 max_h = cam_info_ptr->m_max_height;
	if (!CPOBase::checkRange(m_autoexp_window.x1, 0, max_w))
	{
		m_autoexp_window.x1 = 0;
	}
	if (!CPOBase::checkRange(m_autoexp_window.y1, 0, max_h))
	{
		m_autoexp_window.y1 = 0;
	}
	if (!CPOBase::checkRange(m_autoexp_window.x2, m_autoexp_window.x1, max_w))
	{
		m_autoexp_window.x2 = max_w;
	}
	if (!CPOBase::checkRange(m_autoexp_window.y2, m_autoexp_window.y1, max_h))
	{
		m_autoexp_window.y2 = max_h;
	}
}

i32 CameraExposure::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_autoexp_mode);
	len += sizeof(m_autogain_min);
	len += sizeof(m_autogain_max);
	len += sizeof(m_autoexp_min);
	len += sizeof(m_autoexp_max);
	len += sizeof(m_auto_brightness);
	len += sizeof(m_autoexp_window);

	len += sizeof(m_gain);
	len += sizeof(m_exposure);
	return len;
}

i32 CameraExposure::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_autoexp_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_autogain_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_autogain_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_autoexp_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_autoexp_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_auto_brightness, buffer_ptr, buffer_size);
	CPOBase::memRead(m_autoexp_window, buffer_ptr, buffer_size);

	CPOBase::memRead(m_gain, buffer_ptr, buffer_size);
	CPOBase::memRead(m_exposure, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraExposure::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_autoexp_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_autogain_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_autogain_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_autoexp_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_autoexp_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_auto_brightness, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_autoexp_window, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_gain, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_exposure, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraExposure::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_autoexp_mode, fp);
	CPOBase::fileRead(m_autogain_min, fp);
	CPOBase::fileRead(m_autogain_max, fp);
	CPOBase::fileRead(m_autoexp_min, fp);
	CPOBase::fileRead(m_autoexp_max, fp);
	CPOBase::fileRead(m_auto_brightness, fp);
	CPOBase::fileRead(m_autoexp_window, fp);

	CPOBase::fileRead(m_gain, fp);
	CPOBase::fileRead(m_exposure, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraExposure::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_autoexp_mode, fp);
	CPOBase::fileWrite(m_autogain_min, fp);
	CPOBase::fileWrite(m_autogain_max, fp);
	CPOBase::fileWrite(m_autoexp_min, fp);
	CPOBase::fileWrite(m_autoexp_max, fp);
	CPOBase::fileWrite(m_auto_brightness, fp);
	CPOBase::fileWrite(m_autoexp_window, fp);

	CPOBase::fileWrite(m_gain, fp);
	CPOBase::fileWrite(m_exposure, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

void CameraExposure::setValue(CameraExposure& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

CameraExposure CameraExposure::getValue()
{
	lock_guard();
	return *this;
}

//////////////////////////////////////////////////////////////////////////
CameraInfo::CameraInfo()
{
	reset();
}

CameraInfo::~CameraInfo()
{
	m_cam_blob.freeBuffer();
}

void CameraInfo::reset()
{
	m_cam_name = "";
	m_cam_capability = kCamSupportNone;
	m_cam_blob.freeBuffer();
	memset(m_cam_reserved, 0, sizeof(m_cam_reserved));

	m_max_width = 0;
	m_max_height = 0;

	m_gain_min = 0;
	m_gain_max = 0;
	m_exposure_min = 0;
	m_exposure_max = 0;
	m_brightness_min = 0;
	m_brightness_max = 0;

	m_red_gain_min = 0;
	m_red_gain_max = 0;
	m_green_gain_min = 0;
	m_green_gain_max = 0;
	m_blue_gain_min = 0;
	m_blue_gain_max = 0;

	m_gamma_min = 0;
	m_gamma_max = 0;
	m_contrast_min = 0;
	m_contrast_max = 0;
	m_saturation_min = 0;
	m_saturation_max = 0;
	m_sharpness_min = 0;
	m_sharpness_max = 0;
}

void CameraInfo::init()
{
	lock_guard();
	reset();
}

void CameraInfo::release()
{
	lock_guard();
	m_cam_blob.freeBuffer();
}

i32 CameraInfo::memSize()
{
	lock_guard();
	i32 len = 0;

	len += CPOBase::getStringMemSize(m_cam_name);
	len += sizeof(m_cam_capability);
	len += sizeof(m_max_width);
	len += sizeof(m_max_height);

	len += sizeof(m_gain_min);
	len += sizeof(m_gain_max);
	len += sizeof(m_exposure_min);
	len += sizeof(m_exposure_max);
	len += sizeof(m_brightness_min);
	len += sizeof(m_brightness_max);

	len += sizeof(m_red_gain_min);
	len += sizeof(m_red_gain_max);
	len += sizeof(m_green_gain_min);
	len += sizeof(m_green_gain_max);
	len += sizeof(m_blue_gain_min);
	len += sizeof(m_blue_gain_max);

	len += sizeof(m_gamma_min);
	len += sizeof(m_gamma_max);
	len += sizeof(m_contrast_min);
	len += sizeof(m_contrast_max);
	len += sizeof(m_saturation_min);
	len += sizeof(m_saturation_max);
	len += sizeof(m_sharpness_min);
	len += sizeof(m_sharpness_max);
	return len;
}

i32 CameraInfo::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(buffer_ptr, buffer_size, m_cam_name);
	CPOBase::memRead(m_cam_capability, buffer_ptr, buffer_size);
	CPOBase::memRead(m_max_width, buffer_ptr, buffer_size);
	CPOBase::memRead(m_max_height, buffer_ptr, buffer_size);

	CPOBase::memRead(m_gain_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_gain_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_exposure_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_exposure_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_brightness_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_brightness_max, buffer_ptr, buffer_size);

	CPOBase::memRead(m_red_gain_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_red_gain_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_green_gain_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_green_gain_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_blue_gain_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_blue_gain_max, buffer_ptr, buffer_size);

	CPOBase::memRead(m_gamma_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_gamma_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_contrast_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_contrast_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_saturation_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_saturation_max, buffer_ptr, buffer_size);
	CPOBase::memRead(m_sharpness_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_sharpness_max, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraInfo::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(buffer_ptr, buffer_size, m_cam_name);
	CPOBase::memWrite(m_cam_capability, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_max_width, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_max_height, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_gain_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_gain_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_exposure_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_exposure_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_brightness_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_brightness_max, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_red_gain_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_red_gain_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_green_gain_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_green_gain_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_blue_gain_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_blue_gain_max, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_gamma_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_gamma_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_contrast_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_contrast_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_saturation_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_saturation_max, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_sharpness_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_sharpness_max, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraInfo::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(fp, m_cam_name);
	CPOBase::fileRead(m_cam_capability, fp);
	CPOBase::fileRead(m_max_width, fp);
	CPOBase::fileRead(m_max_height, fp);

	CPOBase::fileRead(m_gain_min, fp);
	CPOBase::fileRead(m_gain_max, fp);
	CPOBase::fileRead(m_exposure_min, fp);
	CPOBase::fileRead(m_exposure_max, fp);
	CPOBase::fileRead(m_brightness_min, fp);
	CPOBase::fileRead(m_brightness_max, fp);

	CPOBase::fileRead(m_red_gain_min, fp);
	CPOBase::fileRead(m_red_gain_max, fp);
	CPOBase::fileRead(m_green_gain_min, fp);
	CPOBase::fileRead(m_green_gain_max, fp);
	CPOBase::fileRead(m_blue_gain_min, fp);
	CPOBase::fileRead(m_blue_gain_max, fp);

	CPOBase::fileRead(m_gamma_min, fp);
	CPOBase::fileRead(m_gamma_max, fp);
	CPOBase::fileRead(m_contrast_min, fp);
	CPOBase::fileRead(m_contrast_max, fp);
	CPOBase::fileRead(m_saturation_min, fp);
	CPOBase::fileRead(m_saturation_max, fp);
	CPOBase::fileRead(m_sharpness_min, fp);
	CPOBase::fileRead(m_sharpness_max, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraInfo::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(fp, m_cam_name);
	CPOBase::fileWrite(m_cam_capability, fp);
	CPOBase::fileWrite(m_max_width, fp);
	CPOBase::fileWrite(m_max_height, fp);

	CPOBase::fileWrite(m_gain_min, fp);
	CPOBase::fileWrite(m_gain_max, fp);
	CPOBase::fileWrite(m_exposure_min, fp);
	CPOBase::fileWrite(m_exposure_max, fp);
	CPOBase::fileWrite(m_brightness_min, fp);
	CPOBase::fileWrite(m_brightness_max, fp);

	CPOBase::fileWrite(m_red_gain_min, fp);
	CPOBase::fileWrite(m_red_gain_max, fp);
	CPOBase::fileWrite(m_green_gain_min, fp);
	CPOBase::fileWrite(m_green_gain_max, fp);
	CPOBase::fileWrite(m_blue_gain_min, fp);
	CPOBase::fileWrite(m_blue_gain_max, fp);

	CPOBase::fileWrite(m_gamma_min, fp);
	CPOBase::fileWrite(m_gamma_max, fp);
	CPOBase::fileWrite(m_contrast_min, fp);
	CPOBase::fileWrite(m_contrast_max, fp);
	CPOBase::fileWrite(m_saturation_min, fp);
	CPOBase::fileWrite(m_saturation_max, fp);
	CPOBase::fileWrite(m_sharpness_min, fp);
	CPOBase::fileWrite(m_sharpness_max, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

i32 CameraInfo::getMaxChannel()
{
	lock_guard();
	if (CPOBase::bitCheck(m_cam_capability, kCamSupportColor))
	{
		return kPORGBChannels;
	}
	else if (CPOBase::bitCheck(m_cam_capability, kCamSupportGray))
	{
		return kPOGrayChannels;
	}
	return 0;
}

CameraInfo CameraInfo::getValue()
{
	lock_guard();

	CameraInfo tmp_info;
	tmp_info.m_cam_name = m_cam_name;
	tmp_info.m_cam_capability = m_cam_capability;

	tmp_info.m_max_width = m_max_width;
	tmp_info.m_max_height = m_max_height;
	
	tmp_info.m_gain_min = m_gain_min;
	tmp_info.m_gain_max = m_gain_max;
	tmp_info.m_exposure_min = m_exposure_min;
	tmp_info.m_exposure_max = m_exposure_max;
	tmp_info.m_brightness_min = m_brightness_min;
	tmp_info.m_brightness_max = m_brightness_max;

	tmp_info.m_red_gain_min = m_red_gain_min;
	tmp_info.m_red_gain_max = m_red_gain_max;
	tmp_info.m_green_gain_min = m_green_gain_min;
	tmp_info.m_green_gain_max = m_green_gain_max;
	tmp_info.m_blue_gain_min = m_blue_gain_min;
	tmp_info.m_blue_gain_max = m_blue_gain_max;

	tmp_info.m_gamma_min = m_gamma_min;
	tmp_info.m_gamma_max = m_gamma_max;
	tmp_info.m_contrast_min = m_contrast_min;
	tmp_info.m_contrast_max = m_contrast_max;
	tmp_info.m_saturation_min = m_saturation_min;
	tmp_info.m_saturation_max = m_saturation_max;
	tmp_info.m_sharpness_min = m_sharpness_min;
	tmp_info.m_sharpness_max = m_sharpness_max;
	return tmp_info;
}

void CameraInfo::setValue(CameraInfo& other)
{
	lock_guard();
	anlock_guard(other);
	
	m_cam_name = other.m_cam_name;
	m_cam_capability = other.m_cam_capability;

	m_max_width = other.m_max_width;
	m_max_height = other.m_max_height;

	m_gain_min = other.m_gain_min;
	m_gain_max = other.m_gain_max;
	m_exposure_min = other.m_exposure_min;
	m_exposure_max = other.m_exposure_max;
	m_brightness_min = other.m_brightness_min;
	m_brightness_max = other.m_brightness_max;

	m_red_gain_min = other.m_red_gain_min;
	m_red_gain_max = other.m_red_gain_max;
	m_green_gain_min = other.m_green_gain_min;
	m_green_gain_max = other.m_green_gain_max;
	m_blue_gain_min = other.m_blue_gain_min;
	m_blue_gain_max = other.m_blue_gain_max;

	m_gamma_min = other.m_gamma_min;
	m_gamma_max = other.m_gamma_max;
	m_contrast_min = other.m_contrast_min;
	m_contrast_max = other.m_contrast_max;
	m_saturation_min = other.m_saturation_min;
	m_saturation_max = other.m_saturation_max;
	m_sharpness_min = other.m_sharpness_min;
	m_sharpness_max = other.m_sharpness_max;
}

//////////////////////////////////////////////////////////////////////////
CameraState::CameraState()
{
	reset();
}

void CameraState::reset()
{
	m_capture_width = 0;
	m_capture_height = 0;

	m_is_autoexp_mode = false;
	m_exposure = 0;
	m_gain = 0;

	m_rgain = 0;
	m_ggain = 0;
	m_bgain = 0;

	m_focus = 0;
	m_focus_min = PO_MAXINT;
	m_focus_max = 0;
}

void CameraState::init()
{
	lock_guard();
	reset();
}

void CameraState::setCameraFocus(f32 focus)
{
	lock_guard();
	if (focus <= 0)
	{
		return;
	}

	m_focus = focus;
	m_focus_max = po::_max(m_focus_max, focus)*0.97f + focus*0.03f;
	m_focus_min = po::_min(m_focus_min, focus)*0.97f + focus*0.03f;
}

void CameraState::clearFocusHistory()
{
	lock_guard();
	reset();
}

i32 CameraState::memSize()
{
	lock_guard();
	i32 len = 0;
	len += sizeof(m_capture_width);
	len += sizeof(m_capture_height);

	len += sizeof(m_is_autoexp_mode);
	len += sizeof(m_exposure);
	len += sizeof(m_gain);
	
	len += sizeof(m_rgain);
	len += sizeof(m_ggain);
	len += sizeof(m_bgain);

	len += sizeof(m_focus);
	len += sizeof(m_focus_min);
	len += sizeof(m_focus_max);
	return len;
}

i32 CameraState::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_capture_width, buffer_ptr, buffer_size);
	CPOBase::memRead(m_capture_height, buffer_ptr, buffer_size);

	CPOBase::memRead(m_is_autoexp_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_exposure, buffer_ptr, buffer_size);
	CPOBase::memRead(m_gain, buffer_ptr, buffer_size);
	
	CPOBase::memRead(m_rgain, buffer_ptr, buffer_size);
	CPOBase::memRead(m_ggain, buffer_ptr, buffer_size);
	CPOBase::memRead(m_bgain, buffer_ptr, buffer_size);

	CPOBase::memRead(m_focus, buffer_ptr, buffer_size);
	CPOBase::memRead(m_focus_min, buffer_ptr, buffer_size);
	CPOBase::memRead(m_focus_max, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraState::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_capture_width, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_capture_height, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_is_autoexp_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_exposure, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_gain, buffer_ptr, buffer_size);
	
	CPOBase::memWrite(m_rgain, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_ggain, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_bgain, buffer_ptr, buffer_size);
	
	CPOBase::memWrite(m_focus, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_focus_min, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_focus_max, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraState::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}
	CPOBase::fileRead(m_capture_width, fp);
	CPOBase::fileRead(m_capture_height, fp);

	CPOBase::fileRead(m_is_autoexp_mode, fp);
	CPOBase::fileRead(m_exposure, fp);
	CPOBase::fileRead(m_gain, fp);

	CPOBase::fileRead(m_rgain, fp);
	CPOBase::fileRead(m_ggain, fp);
	CPOBase::fileRead(m_bgain, fp);

	CPOBase::fileRead(m_focus, fp);
	CPOBase::fileRead(m_focus_min, fp);
	CPOBase::fileRead(m_focus_max, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraState::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_capture_width, fp);
	CPOBase::fileWrite(m_capture_height, fp);

	CPOBase::fileWrite(m_is_autoexp_mode, fp);
	CPOBase::fileWrite(m_exposure, fp);
	CPOBase::fileWrite(m_gain, fp);

	CPOBase::fileWrite(m_rgain, fp);
	CPOBase::fileWrite(m_ggain, fp);
	CPOBase::fileWrite(m_bgain, fp);

	CPOBase::fileWrite(m_focus, fp);
	CPOBase::fileWrite(m_focus_min, fp);
	CPOBase::fileWrite(m_focus_max, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

CameraState CameraState::getValue()
{
	lock_guard();
	return *this;
}

void CameraState::setValue(CameraState& cam_state)
{
	lock_guard();
	anlock_guard(cam_state);
	*this = cam_state;
}

//////////////////////////////////////////////////////////////////////////
CameraFocus::CameraFocus()
{
	reset();
}

void CameraFocus::reset()
{
	m_auto_focus = false;
	m_auto_focus_region.reset();
	m_focus_pos = 0;
	m_focus_length = 0;
}

void CameraFocus::init()
{
	lock_guard();
	reset();
}

CameraFocus CameraFocus::getValue()
{
	lock_guard();
	return *this;
}

void CameraFocus::setValue(CameraFocus& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CameraFocus::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_auto_focus);
	len += sizeof(m_auto_focus_region);
	len += sizeof(m_focus_pos);
	len += sizeof(m_focus_length);
	return len;
}

i32 CameraFocus::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_auto_focus, buffer_ptr, buffer_size);
	CPOBase::memRead(m_auto_focus_region, buffer_ptr, buffer_size);
	CPOBase::memRead(m_focus_pos, buffer_ptr, buffer_size);
	CPOBase::memRead(m_focus_length, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraFocus::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_auto_focus, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_auto_focus_region, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_focus_pos, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_focus_length, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraFocus::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_auto_focus, fp);
	CPOBase::fileRead(m_auto_focus_region, fp);
	CPOBase::fileRead(m_focus_pos, fp);
	CPOBase::fileRead(m_focus_length, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraFocus::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_auto_focus, fp);
	CPOBase::fileWrite(m_auto_focus_region, fp);
	CPOBase::fileWrite(m_focus_pos, fp);
	CPOBase::fileWrite(m_focus_length, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CameraColor::CameraColor()
{
	reset();
}

void CameraColor::reset()
{
	m_color_mode = kCamColorAny;
	m_wb_mode = true;

	m_red_gain = 1.0f;
	m_green_gain = 1.0f;
	m_blue_gain = 1.0f;
}

void CameraColor::init()
{
	lock_guard();
	reset();
}

void CameraColor::updateValidation(CameraInfo* cam_info_ptr)
{
	if (!cam_info_ptr)
	{
		return;
	}
	if (!CPOBase::checkRange(m_red_gain, cam_info_ptr->m_red_gain_min, cam_info_ptr->m_red_gain_max))
	{
		m_red_gain = (cam_info_ptr->m_red_gain_min + cam_info_ptr->m_red_gain_max) / 2;
	}
	if (!CPOBase::checkRange(m_green_gain, cam_info_ptr->m_green_gain_min, cam_info_ptr->m_green_gain_max))
	{
		m_green_gain = (cam_info_ptr->m_green_gain_min + cam_info_ptr->m_green_gain_max) / 2;
	}
	if (!CPOBase::checkRange(m_blue_gain, cam_info_ptr->m_blue_gain_min, cam_info_ptr->m_blue_gain_max))
	{
		m_blue_gain = (cam_info_ptr->m_blue_gain_min + cam_info_ptr->m_blue_gain_max) / 2;
	}
}

CameraColor CameraColor::getValue()
{
	lock_guard();
	return *this;
}

void CameraColor::setValue(CameraColor& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CameraColor::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_color_mode);
	len += sizeof(m_wb_mode);

	len += sizeof(m_red_gain);
	len += sizeof(m_green_gain);
	len += sizeof(m_blue_gain);
	return len;
}

i32 CameraColor::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* stpos = buffer_ptr;

	CPOBase::memRead(m_color_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_wb_mode, buffer_ptr, buffer_size);

	CPOBase::memRead(m_red_gain, buffer_ptr, buffer_size);
	CPOBase::memRead(m_green_gain, buffer_ptr, buffer_size);
	CPOBase::memRead(m_blue_gain, buffer_ptr, buffer_size);
	return buffer_ptr - stpos;
}

i32 CameraColor::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* stpos = buffer_ptr;

	CPOBase::memWrite(m_color_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_wb_mode, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_red_gain, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_green_gain, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_blue_gain, buffer_ptr, buffer_size);
	return buffer_ptr - stpos;
}

bool CameraColor::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_color_mode, fp);
	CPOBase::fileRead(m_wb_mode, fp);

	CPOBase::fileRead(m_red_gain, fp);
	CPOBase::fileRead(m_green_gain, fp);
	CPOBase::fileRead(m_blue_gain, fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraColor::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_color_mode, fp);
	CPOBase::fileWrite(m_wb_mode, fp);

	CPOBase::fileWrite(m_red_gain, fp);
	CPOBase::fileWrite(m_green_gain, fp);
	CPOBase::fileWrite(m_blue_gain, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

void CameraColor::setColorMode(i32 color_mode)
{
	lock_guard();
	m_color_mode = color_mode;
}

i32 CameraColor::getChannel()
{
	switch (m_color_mode)
	{
		case kCamColorGray:		return kPOGrayChannels;
		case kCamColorYUV422:	return kPOYUVChannels;
		case kCamColorRGB8:		return kPORGBChannels;
	}
	return kPOAnyChannels;
}

//////////////////////////////////////////////////////////////////////////
CameraCorrection::CameraCorrection()
{
	reset();
}

void CameraCorrection::reset()
{
	m_gamma = 1.0f;
	m_contrast= 100;
	m_saturation = 100;
	m_sharpness = 0;
}

void CameraCorrection::init()
{
	lock_guard();
	reset();
}

void CameraCorrection::updateValidation(CameraInfo* cam_info_ptr)
{
	if (!cam_info_ptr)
	{
		return;
	}
	if (!CPOBase::checkRange(m_gamma, cam_info_ptr->m_gamma_min, cam_info_ptr->m_gamma_max))
	{
		m_gamma = (cam_info_ptr->m_gamma_min + cam_info_ptr->m_gamma_max) / 2;
	}
	if (!CPOBase::checkRange(m_contrast, cam_info_ptr->m_contrast_min, cam_info_ptr->m_contrast_max))
	{
		m_contrast = (cam_info_ptr->m_contrast_min + cam_info_ptr->m_contrast_max) / 2;
	}
	if (!CPOBase::checkRange(m_saturation, cam_info_ptr->m_saturation_min, cam_info_ptr->m_saturation_max))
	{
		m_saturation = (cam_info_ptr->m_saturation_min + cam_info_ptr->m_saturation_max) / 2;
	}
	if (!CPOBase::checkRange(m_sharpness, cam_info_ptr->m_sharpness_min, cam_info_ptr->m_sharpness_max))
	{
		m_sharpness = (cam_info_ptr->m_sharpness_min + cam_info_ptr->m_sharpness_max) / 2;
	}
}

CameraCorrection CameraCorrection::getValue()
{
	lock_guard();
	return *this;
}

void CameraCorrection::setValue(CameraCorrection& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CameraCorrection::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_gamma);
	len += sizeof(m_contrast);
	len += sizeof(m_saturation);
	len += sizeof(m_sharpness);
	return len;
}

i32 CameraCorrection::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* stpos = buffer_ptr;

	CPOBase::memRead(m_gamma, buffer_ptr, buffer_size);
	CPOBase::memRead(m_contrast, buffer_ptr, buffer_size);
	CPOBase::memRead(m_saturation, buffer_ptr, buffer_size);
	CPOBase::memRead(m_sharpness, buffer_ptr, buffer_size);
	return buffer_ptr - stpos;
}

i32 CameraCorrection::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* stpos = buffer_ptr;

	CPOBase::memWrite(m_gamma, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_contrast, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_saturation, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_sharpness, buffer_ptr, buffer_size);
	return buffer_ptr - stpos;
}

bool CameraCorrection::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_gamma, fp);
	CPOBase::fileRead(m_contrast, fp);
	CPOBase::fileRead(m_saturation, fp);
	CPOBase::fileRead(m_sharpness, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraCorrection::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_gamma, fp);
	CPOBase::fileWrite(m_contrast, fp);
	CPOBase::fileWrite(m_saturation, fp);
	CPOBase::fileWrite(m_sharpness, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CameraMultiCap::CameraMultiCap()
{
	reset();
}

void CameraMultiCap::reset()
{
	m_enable_multicap = false;
	m_capture_count = 0;
	m_measure_mode = 0;
	m_error_mode = 0;
}

void CameraMultiCap::init()
{
	lock_guard();
	reset();
}

CameraMultiCap CameraMultiCap::getValue()
{
	lock_guard();
	return *this;
}

void CameraMultiCap::setValue(CameraMultiCap& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CameraMultiCap::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_enable_multicap);
	len += sizeof(m_capture_count);
	len += sizeof(m_measure_mode);
	len += sizeof(m_error_mode);
	return len;
}

i32 CameraMultiCap::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_enable_multicap, buffer_ptr, buffer_size);
	CPOBase::memRead(m_capture_count, buffer_ptr, buffer_size);
	CPOBase::memRead(m_measure_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_error_mode, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraMultiCap::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_enable_multicap, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_capture_count, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_measure_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_error_mode, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraMultiCap::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_enable_multicap, fp);
	CPOBase::fileRead(m_capture_count, fp);
	CPOBase::fileRead(m_measure_mode, fp);
	CPOBase::fileRead(m_error_mode, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraMultiCap::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_enable_multicap, fp);
	CPOBase::fileWrite(m_capture_count, fp);
	CPOBase::fileWrite(m_measure_mode, fp);
	CPOBase::fileWrite(m_error_mode, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CameraHDR::CameraHDR()
{
	reset();
}

void CameraHDR::reset()
{
	m_enable_hdr = false;
	m_hdr_mode = 0;

	m_min_hdr = 0;
	m_max_hdr = 0;
	m_hdr_input = 0;
}

void CameraHDR::init()
{
	lock_guard();
	reset();
}

CameraHDR CameraHDR::getValue()
{
	lock_guard();
	return *this;
}

void CameraHDR::setValue(CameraHDR& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

i32 CameraHDR::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_enable_hdr);
	len += sizeof(m_hdr_mode);
	len += sizeof(m_min_hdr);
	len += sizeof(m_max_hdr);
	len += sizeof(m_hdr_input);
	return len;
}

i32 CameraHDR::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_enable_hdr, buffer_ptr, buffer_size);
	CPOBase::memRead(m_hdr_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_min_hdr, buffer_ptr, buffer_size);
	CPOBase::memRead(m_max_hdr, buffer_ptr, buffer_size);
	CPOBase::memRead(m_hdr_input, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraHDR::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_enable_hdr, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_hdr_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_min_hdr, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_max_hdr, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_hdr_input, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraHDR::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_enable_hdr, fp);
	CPOBase::fileRead(m_hdr_mode, fp);
	CPOBase::fileRead(m_min_hdr, fp);
	CPOBase::fileRead(m_max_hdr, fp);
	CPOBase::fileRead(m_hdr_input, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraHDR::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_enable_hdr, fp);
	CPOBase::fileWrite(m_hdr_mode, fp);
	CPOBase::fileWrite(m_min_hdr, fp);
	CPOBase::fileWrite(m_max_hdr, fp);
	CPOBase::fileWrite(m_hdr_input, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CameraRange::CameraRange()
{
	reset();
}

void CameraRange::reset()
{
	m_range.reset();
	m_rotation = kPORotation0;
	m_is_flip_x = false;
	m_is_flip_y = false;
	m_is_invert = false;
}

void CameraRange::init()
{
	lock_guard();
	reset();
}

void CameraRange::updateValidation(CameraInfo* cam_info_ptr)
{
	i32 max_w = cam_info_ptr->m_max_width;
	i32 max_h = cam_info_ptr->m_max_height;

	if (!CPOBase::checkRange(m_range.x1, 0, max_w))
	{
		m_range.x1 = 0;
	}
	if (!CPOBase::checkRange(m_range.y1, 0, max_h))
	{
		m_range.y1 = 0;
	}
	if (!CPOBase::checkRange(m_range.x2, m_range.x1, max_w))
	{
		m_range.x2 = max_w;
	}
	if (!CPOBase::checkRange(m_range.y2, m_range.y1, max_h))
	{
		m_range.y2 = max_h;
	}
	if (m_range.getWidth() == 0 || m_range.getHeight() == 0)
	{
		m_range = Recti(0, 0, max_w, max_h);
	}
}

i32 CameraRange::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_range);
	len += sizeof(m_rotation);
	len += sizeof(m_is_flip_x);
	len += sizeof(m_is_flip_y);
	len += sizeof(m_is_invert);
	return len;
}

i32 CameraRange::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_range, buffer_ptr, buffer_size);
	CPOBase::memRead(m_rotation, buffer_ptr, buffer_size);
	CPOBase::memRead(m_is_flip_x, buffer_ptr, buffer_size);
	CPOBase::memRead(m_is_flip_y, buffer_ptr, buffer_size);
	CPOBase::memRead(m_is_invert, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraRange::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_range, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_rotation, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_is_flip_x, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_is_flip_y, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_is_invert, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraRange::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_range, fp);
	CPOBase::fileRead(m_rotation, fp);
	CPOBase::fileRead(m_is_flip_x, fp);
	CPOBase::fileRead(m_is_flip_y, fp);
	CPOBase::fileRead(m_is_invert, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraRange::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_range, fp);
	CPOBase::fileWrite(m_rotation, fp);
	CPOBase::fileWrite(m_is_flip_x, fp);
	CPOBase::fileWrite(m_is_flip_y, fp);
	CPOBase::fileWrite(m_is_invert, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

CameraRange CameraRange::getValue()
{
	lock_guard();
	return *this;
}

void CameraRange::setValue(CameraRange& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

//////////////////////////////////////////////////////////////////////////
CameraSpec::CameraSpec()
{
	reset();
}

void CameraSpec::reset()
{
	m_noise_reduce = true;
	m_anti_flick = false;
	m_ambient_freq = kCamEnvFrequency50Hz;

	m_color_temperature = kCamColorTempAuto;

	m_jitter_time = 0; //0ms
	m_shutter_mode = kShutterRolling;
}

void CameraSpec::init()
{
	lock_guard();
	reset();
}

i32 CameraSpec::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_noise_reduce);
	len += sizeof(m_anti_flick);
	len += sizeof(m_ambient_freq);

	len += sizeof(m_color_temperature);

	len += sizeof(m_jitter_time);
	len += sizeof(m_shutter_mode);
	return len;
}

i32 CameraSpec::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_noise_reduce, buffer_ptr, buffer_size);
	CPOBase::memRead(m_anti_flick, buffer_ptr, buffer_size);
	CPOBase::memRead(m_ambient_freq, buffer_ptr, buffer_size);

	CPOBase::memRead(m_color_temperature, buffer_ptr, buffer_size);
	
	CPOBase::memRead(m_jitter_time, buffer_ptr, buffer_size);
	CPOBase::memRead(m_shutter_mode, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraSpec::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_noise_reduce, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_anti_flick, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_ambient_freq, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_color_temperature, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_jitter_time, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_shutter_mode, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraSpec::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_noise_reduce, fp);
	CPOBase::fileRead(m_anti_flick, fp);
	CPOBase::fileRead(m_ambient_freq, fp);

	CPOBase::fileRead(m_color_temperature, fp);

	CPOBase::fileRead(m_jitter_time, fp);
	CPOBase::fileRead(m_shutter_mode, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraSpec::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_noise_reduce, fp);
	CPOBase::fileWrite(m_anti_flick, fp);
	CPOBase::fileWrite(m_ambient_freq, fp);

	CPOBase::fileWrite(m_color_temperature, fp);

	CPOBase::fileWrite(m_jitter_time, fp);
	CPOBase::fileWrite(m_shutter_mode, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

CameraSpec CameraSpec::getValue()
{
	lock_guard();
	return *this;
}

void CameraSpec::setValue(CameraSpec& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

void CameraSpec::updateValidation()
{
	m_jitter_time = po::_max(m_jitter_time, 0);
	m_shutter_mode = CPOBase::updateRange(m_shutter_mode, (i32)kShutterRolling, (i32)(kShutterTypeCount - 1));
	m_color_temperature = CPOBase::updateRange(m_color_temperature,(u8)kCamColorTempAuto, (u8)(kCamColorTempCount - 1));
}

//////////////////////////////////////////////////////////////////////////
CameraStrobe::CameraStrobe()
{
	reset();
}

void CameraStrobe::init()
{
	lock_guard();
	reset();
}

void CameraStrobe::reset()
{
	m_use_strobe = true;
	m_is_auto_strobe = true;
	m_strobe_level = kStrobeLevelLow;
	m_strobe_pwm_delay = 0;
	m_strobe_pwm_width = 1000;
}

CameraStrobe CameraStrobe::getValue()
{
	return *this;
}

void CameraStrobe::setValue(CameraStrobe& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

void CameraStrobe::setValue(CameraStrobe* other_ptr)
{
	if (!other_ptr)
	{
		return;
	}

	lock_guard();
	anlock_guard_ptr(other_ptr);
	*this = *other_ptr;
}

i32 CameraStrobe::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_use_strobe);
	len += sizeof(m_is_auto_strobe);
	len += sizeof(m_strobe_level);
	len += sizeof(m_strobe_pwm_delay);
	len += sizeof(m_strobe_pwm_width);
	return len;
}

i32 CameraStrobe::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_use_strobe, buffer_ptr, buffer_size);
	CPOBase::memRead(m_is_auto_strobe, buffer_ptr, buffer_size);
	CPOBase::memRead(m_strobe_level, buffer_ptr, buffer_size);
	CPOBase::memRead(m_strobe_pwm_delay, buffer_ptr, buffer_size);
	CPOBase::memRead(m_strobe_pwm_width, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraStrobe::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_use_strobe, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_is_auto_strobe, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_strobe_level, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_strobe_pwm_delay, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_strobe_pwm_width, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraStrobe::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_use_strobe, fp);
	CPOBase::fileRead(m_is_auto_strobe, fp);
	CPOBase::fileRead(m_strobe_level, fp);
	CPOBase::fileRead(m_strobe_pwm_delay, fp);
	CPOBase::fileRead(m_strobe_pwm_width, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraStrobe::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_use_strobe, fp);
	CPOBase::fileWrite(m_is_auto_strobe, fp);
	CPOBase::fileWrite(m_strobe_level, fp);
	CPOBase::fileWrite(m_strobe_pwm_delay, fp);
	CPOBase::fileWrite(m_strobe_pwm_width, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

//////////////////////////////////////////////////////////////////////////
CameraSetting::CameraSetting()
{
	init();
}

void CameraSetting::init()
{
	m_cam_id = -1;
	m_cam_handle = -1;
	m_cam_type = kPOCamUnknown;
	m_cam_ready = false;
	m_cam_used = false;

	m_cam_trigger.init();
	m_cam_exposure.init();
	m_cam_focus.init();
	m_cam_color.init();
	m_cam_multicap.init();
	m_cam_hdr.init();
	m_cam_range.init();
	m_cam_spec.init();
	m_cam_state.init();
	m_cam_strobe.init();
}

void CameraSetting::setValue(CameraSetting* other_ptr, i32 mode)
{
	if (CPOBase::bitCheck(mode, kCamSettingUpdateInfo))
	{
		m_cam_id = other_ptr->m_cam_id;

		m_cam_info.setValue(other_ptr->m_cam_info);
		m_cam_state.setValue(other_ptr->m_cam_state);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateTrigger))
	{
		m_cam_trigger.setValue(other_ptr->m_cam_trigger);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateCalib))
	{
		m_cam_calib.setValue(other_ptr->m_cam_calib);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateExposure))
	{
		m_cam_exposure.setValue(other_ptr->m_cam_exposure);
		m_cam_exposure.updateValidation(&m_cam_info);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateFocus))
	{
		m_cam_focus.setValue(other_ptr->m_cam_focus);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateColor))
	{
		m_cam_color.setValue(other_ptr->m_cam_color);
		m_cam_color.updateValidation(&m_cam_info);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateCorrection))
	{
		m_cam_correction.setValue(other_ptr->m_cam_correction);
		m_cam_correction.updateValidation(&m_cam_info);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateMultiCapture))
	{
		m_cam_multicap.setValue(other_ptr->m_cam_multicap);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateHDR))
	{
		m_cam_hdr.setValue(other_ptr->m_cam_hdr);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateGeometric))
	{
		m_cam_range.setValue(other_ptr->m_cam_range);
		m_cam_range.updateValidation(&m_cam_info);
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateShutter))
	{
		m_cam_spec.setValue(other_ptr->m_cam_spec);
		m_cam_spec.updateValidation();
	}
	if (CPOBase::bitCheck(mode, kCamSettingUpdateStrobe))
	{
		m_cam_strobe.setValue(other_ptr->m_cam_strobe);
	}
}

i32 CameraSetting::memSize()
{
	i32 len = 0;
	len += sizeof(m_cam_id);
	len += sizeof(m_cam_ready);

	len += m_cam_info.memSize();
	len += m_cam_trigger.memSize();
	len += m_cam_exposure.memSize();
	len += m_cam_focus.memSize();
	len += m_cam_color.memSize();
	len += m_cam_multicap.memSize();
	len += m_cam_hdr.memSize();
	len += m_cam_range.memSize();
	len += m_cam_spec.memSize();
	len += m_cam_strobe.memSize();
	len += m_cam_calib.memSize();
	len += m_cam_state.memSize();
	return len;
}

i32 CameraSetting::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	init();
	u8* buffer_pos = buffer_ptr;

#if defined(POR_DEVICE)
	i32 tmp_cam_id;
	bool tmp_is_ready;
	CPOBase::memRead(tmp_cam_id, buffer_ptr, buffer_size);
	CPOBase::memRead(tmp_is_ready, buffer_ptr, buffer_size);
#else
	CPOBase::memRead(m_cam_id, buffer_ptr, buffer_size);
	CPOBase::memRead(m_cam_ready, buffer_ptr, buffer_size);
#endif

	m_cam_info.memRead(buffer_ptr, buffer_size);
	m_cam_trigger.memRead(buffer_ptr, buffer_size);
	m_cam_exposure.memRead(buffer_ptr, buffer_size);
	m_cam_focus.memRead(buffer_ptr, buffer_size);
	m_cam_color.memRead(buffer_ptr, buffer_size);
	m_cam_multicap.memRead(buffer_ptr, buffer_size);
	m_cam_hdr.memRead(buffer_ptr, buffer_size);
	m_cam_range.memRead(buffer_ptr, buffer_size);
	m_cam_spec.memRead(buffer_ptr, buffer_size);
	m_cam_strobe.memRead(buffer_ptr, buffer_size);
	m_cam_calib.memRead(buffer_ptr, buffer_size);
	m_cam_state.memRead(buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraSetting::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memWrite(m_cam_id, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_cam_ready, buffer_ptr, buffer_size);

	m_cam_info.memWrite(buffer_ptr, buffer_size);
	m_cam_trigger.memWrite(buffer_ptr, buffer_size);
	m_cam_exposure.memWrite(buffer_ptr, buffer_size);
	m_cam_focus.memWrite(buffer_ptr, buffer_size);
	m_cam_color.memWrite(buffer_ptr, buffer_size);
	m_cam_multicap.memWrite(buffer_ptr, buffer_size);
	m_cam_hdr.memWrite(buffer_ptr, buffer_size);
	m_cam_range.memWrite(buffer_ptr, buffer_size);
	m_cam_spec.memWrite(buffer_ptr, buffer_size);
	m_cam_strobe.memWrite(buffer_ptr, buffer_size);
	m_cam_calib.memWrite(buffer_ptr, buffer_size);
	m_cam_state.memWrite(buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraSetting::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	m_cam_info.fileRead(fp);
	m_cam_trigger.fileRead(fp);
	m_cam_exposure.fileRead(fp);
	m_cam_focus.fileRead(fp);
	m_cam_color.fileRead(fp);
	m_cam_multicap.fileRead(fp);
	m_cam_hdr.fileRead(fp);
	m_cam_range.fileRead(fp);
	m_cam_spec.fileRead(fp);
	m_cam_strobe.fileRead(fp);
	m_cam_state.fileRead(fp);
	m_cam_calib.fileRead(fp);
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraSetting::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	m_cam_info.fileWrite(fp);
	m_cam_trigger.fileWrite(fp);
	m_cam_exposure.fileWrite(fp);
	m_cam_focus.fileWrite(fp);
	m_cam_color.fileWrite(fp);
	m_cam_multicap.fileWrite(fp);
	m_cam_hdr.fileWrite(fp);
	m_cam_range.fileWrite(fp);
	m_cam_spec.fileWrite(fp);
	m_cam_strobe.fileWrite(fp);
	m_cam_state.fileWrite(fp);
	m_cam_calib.fileWrite(fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

void CameraSetting::setCameraTrigger(const CameraTrigger& trigger_mode)
{
	m_cam_trigger = trigger_mode;
}

void CameraSetting::updateValidation()
{
	m_cam_exposure.updateValidation(&m_cam_info);
	m_cam_color.updateValidation(&m_cam_info);
	m_cam_correction.updateValidation(&m_cam_info);
	m_cam_range.updateValidation(&m_cam_info);
	m_cam_spec.updateValidation();
}

void CameraSetting::memReadChanged(i32 flag, u8*& buffer_ptr, i32& buffer_size)
{
	//read changed geometric parameters from buffer
	if (CPOBase::bitCheck(flag, kPOSubFlagCamGeoInvert))
	{
		CPOBase::memRead(m_cam_range.m_is_invert, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamGeoFlip))
	{
		CPOBase::memRead(m_cam_range.m_is_flip_x, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_range.m_is_flip_y, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamGeoRotation))
	{
		CPOBase::memRead(m_cam_range.m_rotation, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamGeoRange))
	{
		CPOBase::memRead(m_cam_range.m_range, buffer_ptr, buffer_size);
	}

	//read changed exposure parameters from buffer
	if (CPOBase::bitCheck(flag, kPOSubFlagCamGain))
	{
		CPOBase::memRead(m_cam_exposure.m_gain, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamExposure))
	{
		CPOBase::memRead(m_cam_exposure.m_exposure, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamAEMode))
	{
		CPOBase::memRead(m_cam_exposure.m_autoexp_mode, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamAEGain))
	{
		CPOBase::memRead(m_cam_exposure.m_autogain_min, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_exposure.m_autogain_max, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamAEExposure))
	{
		CPOBase::memRead(m_cam_exposure.m_autoexp_min, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_exposure.m_autoexp_max, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamAEBrightness))
	{
		CPOBase::memRead(m_cam_exposure.m_auto_brightness, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamAEWindow))
	{
		CPOBase::memRead(m_cam_exposure.m_autoexp_window, buffer_ptr, buffer_size);
	}

	//read changed color parameter from buffer
	if (CPOBase::bitCheck(flag, kPOSubFlagCamColorMode))
	{
		CPOBase::memRead(m_cam_color.m_color_mode, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamColorWBMode))
	{
		CPOBase::memRead(m_cam_color.m_wb_mode, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamColorGain))
	{
		CPOBase::memRead(m_cam_color.m_red_gain, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_color.m_green_gain, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_color.m_blue_gain, buffer_ptr, buffer_size);
	}

	//read changed Correction parameters form buffer
	if (CPOBase::bitCheck(flag, kPOSubFlagCamCorrectionGamma))
	{
		CPOBase::memRead(m_cam_correction.m_gamma, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamCorrectionContrast))
	{
		CPOBase::memRead(m_cam_correction.m_contrast, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamCorrectionSaturation))
	{
		CPOBase::memRead(m_cam_correction.m_saturation, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamCorrectionSharpness))
	{
		CPOBase::memRead(m_cam_correction.m_sharpness, buffer_ptr, buffer_size);
	}

	//read changed spec and strobe parameter from buffer
	if (CPOBase::bitCheck(flag, kPOSubFlagCamShutter))
	{
		CPOBase::memRead(m_cam_spec.m_jitter_time, buffer_ptr, buffer_size);
		CPOBase::memRead(m_cam_spec.m_shutter_mode, buffer_ptr, buffer_size);
	}
	if (CPOBase::bitCheck(flag, kPOSubFlagCamStrobe))
	{
		m_cam_strobe.memRead(buffer_ptr, buffer_size);
	}
}

void CameraSetting::getCameraResolution(i32& w, i32& h, i32& channel)
{
	CameraInfo cam_info = getCameraInfo()->getValue();
	CameraRange cam_range = getCameraRange()->getValue();
	CameraColor cam_color = getCameraColor()->getValue();
	
	Recti range = cam_range.getRange();
	w = range.getWidth();
	h = range.getHeight();
	channel = cam_color.getChannel();

	w = (w > 0) ? w : cam_info.m_max_width;
	h = (h > 0) ? h : cam_info.m_max_height;
	channel = (channel > 0) ? channel : cam_info.getMaxChannel();
}

//////////////////////////////////////////////////////////////////////////
CameraSet::CameraSet()
{
	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		m_cam_id[i] = PO_DEVICE_CAMPORT;
		m_cam_setting[i].m_cam_id = i;
	}
	m_cam_available = kPOCamUnknown;
}

CameraSet::~CameraSet()
{
}

void CameraSet::init()
{
	lock_guard();
	m_cam_available = kPOCamUnknown;
	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		m_cam_id[i] = PO_DEVICE_CAMPORT;
		m_cam_setting[i].init();
		m_cam_setting[i].m_cam_id = i;
	}
}

bool CameraSet::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		if (!m_cam_setting[i].fileRead(fp))
		{
			return false;
		}
	}
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraSet::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		m_cam_setting[i].fileWrite(fp);
	}

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

i32 CameraSet::memSize()
{
	lock_guard();
	i32 len = 0;
	i32 i, count = PO_CAM_COUNT;
	for (i = 0; i < count; i++)
	{
		len += m_cam_setting[i].memSize();
	}
	len += sizeof(count);
	return len;
}

i32 CameraSet::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	i32 i, count = PO_CAM_COUNT;
	CPOBase::memRead(count, buffer_ptr, buffer_size);
	for (i = 0; i < count; i++)
	{
		m_cam_setting[i].memRead(buffer_ptr, buffer_size);
	}

	return buffer_ptr - buffer_pos;
}

i32 CameraSet::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	i32 i, count = PO_CAM_COUNT;
	CPOBase::memWrite(count, buffer_ptr, buffer_size);
	for (i = 0; i < count; i++)
	{
		m_cam_setting[i].memWrite(buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

bool CameraSet::isReady(i32 index)
{
	lock_guard();
	if (!CPOBase::checkIndex(index, PO_CAM_COUNT))
	{
		return false;
	}
	return m_cam_setting[index].isReady();
}

i32 CameraSet::findCamIndex(postring& camera_name)
{
	lock_guard();
	postring cam_id_name;
	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		cam_id_name = m_cam_id[i];
		CPOBase::toLower(cam_id_name);
		if (cam_id_name == camera_name)
		{
			return i;
		}
	}
	return -1;
}

CameraSetting* CameraSet::getCamSetting(i32 index)
{
	lock_guard();
	if (!CPOBase::checkIndex(index, PO_CAM_COUNT))
	{
		return NULL;
	}
	return m_cam_setting + index;
}

CameraSetting* CameraSet::findFirstCamSetting()
{
	lock_guard();
	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		if (m_cam_setting[i].isUnusedReady())
		{
			return m_cam_setting + i;
		}
	}
	return NULL;
}

void CameraSet::import(CameraSet& other)
{
	lock_guard();
	for (i32 i = 0; i < PO_CAM_COUNT; i++)
	{
		if (m_cam_id[i] != other.m_cam_id[i])
		{
			continue;
		}
		m_cam_setting[i].setValue(other.m_cam_setting + i, kCamSettingUpdateAllCtrl);
	}
}

i32 CameraSet::getAvailableCamType()
{
	lock_guard();
	return m_cam_available;
}

void CameraSet::setAvailableCamType(i32 cam_available)
{
	lock_guard();
	m_cam_available = cam_available;
}

i32 CameraSet::memExportSize()
{
	lock_guard();
	i32 len = 0;
	i32 i, cam_count = PO_CAM_COUNT;

	len += sizeof(i32) * 2; //sign code;
	len += sizeof(m_cam_available);
	len += sizeof(cam_count);
	for (i = 0; i < PO_CAM_COUNT; i++)
	{
		len += CPOBase::getStringMemSize(m_cam_id[i]);
		len += m_cam_setting[i].memSize();
	}
	return len;
}

bool CameraSet::memImport(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	if (!CPOBase::memSignRead(buffer_ptr, buffer_size, PO_SIGN_CODE))
	{
		return false;
	}

	init();
	i32 i, cam_count = PO_CAM_COUNT;

	CPOBase::memRead(m_cam_available, buffer_ptr, buffer_size);
	CPOBase::memRead(cam_count, buffer_ptr, buffer_size);

	cam_count = po::_min(PO_CAM_COUNT, cam_count);
	for (i = 0; i < cam_count; i++)
	{
		CPOBase::memRead(buffer_ptr, buffer_size, m_cam_id[i]);
		m_cam_setting[i].memRead(buffer_ptr, buffer_size);
	}

	return CPOBase::memSignRead(buffer_ptr, buffer_size, PO_SIGN_ENDCODE);
}

bool CameraSet::memExport(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	i32 i, cam_count = PO_CAM_COUNT;
	CPOBase::memSignWrite(buffer_ptr, buffer_size, PO_SIGN_CODE);
	CPOBase::memWrite(m_cam_available, buffer_ptr, buffer_size);
	CPOBase::memWrite(cam_count, buffer_ptr, buffer_size);

	for (i = 0; i < PO_CAM_COUNT; i++)
	{
		CPOBase::memWrite(buffer_ptr, buffer_size, m_cam_id[i]);
		m_cam_setting[i].memWrite(buffer_ptr, buffer_size);
	}

	return CPOBase::memSignWrite(buffer_ptr, buffer_size, PO_SIGN_ENDCODE);
}

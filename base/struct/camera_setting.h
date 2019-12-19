#pragma once

#include "struct.h"
#include "camera_calib.h"

const i32 kCamSnapInterval = 40;

enum CamCapSupport
{
	kCamSupportNone					= 0x00000000,
	kCamSupportGray					= 0x00000001,
	kCamSupportColor				= 0x00000002,
	kCamSupportRange				= 0x00000004,
	kCamSupportAutoExp				= 0x00000008,
	kCamSupportManualExp			= 0x00000010,
	kCamSupportAEBrightness			= 0x00000020,
	kCamSupportAEWindow				= 0x00000040,
	kCamSupportAutoWb				= 0x00000080,
	kCamSupportAutoWbOnce			= 0x00000100,
	kCamSupportManualWb				= 0x00000200,
	kCamSupportRedGain				= 0x00000400,
	kCamSupportGreenGain			= 0x00000800,
	kCamSupportBlueGain				= 0x00001000,
	kCamSupportShutter				= 0x00002000,
	kCamSupportAutoStrobe			= 0x00004000,
	kCamSupportManualStrobe			= 0x00008000,
	kCamSupportGamma				= 0x00010000,
	kCamSupportSaturation			= 0x00020000,
	kCamSupportContrast				= 0x00040000,
	kCamSupportSharpness			= 0x00080000,
	kCamSupportNoiseReduction		= 0x00100000,

	kCamSupportAutoFocus			= 0x01000000,
	kCamSupportLighting				= 0x02000000,
	kCamSupportMultiExp				= 0x04000000,
	kCamSupportHDR					= 0x08000000,

	kCamSupportAlwaysTrigger		= 0x10000000,
	kCamSupportWaitTriggerReady		= 0x20000000
};

enum CamColorMode
{
	kCamColorGray = 0,
	kCamColorRGB8,
	kCamColorYUV422,
	kCamColorAny,

	kCamColorCount
};

enum CamColorTemperature
{
	kCamColorTempAuto = 0,
	kCamColorTempPreset,
	kCamColorTempManual,

	kCamColorTempCount
};

enum CamAutoMode
{
	kCamAEModeOff = 0,
	kCamAEModeOnce,
	kCamAEModeContinuous,

	kCamAEModeCount
};

enum CamTriggerMode
{
	kCamTriggerContinuous = 0,
	kCamTriggerCamera,
	kCamTriggerManual,
	kCamTriggerIO,
	kCamTriggerRS,
	kCamTriggerNetwork,

	kCamTriggerModeCount
};

enum CameraTriggerSignal
{
	kCamTriggerRisingEdge,
	kCamTriggerFallingEdge,
	kCamTriggerSignalNone,
};

enum CameraTriggerUnit
{
	kCamTriggerUnitMs = 0,
	kCamTriggerUnitPulse,

	kCamTriggerUnitCount
};

enum CameraLightMode
{
	kCamLightDCMode = 0,
	kCamLightPWMMode,

	kLightModeCount
};

enum CameraLightPolar
{
	kCamLightPolarPos = 0,
	kCamLightPolarNeg,

	kCamLightPolarCount
};

enum CameraMultiCapMeasure
{
	kCamMultiCapMeasureAverage = 0,
	kCamMultiCapMeasureMin,
	kCamMultiCapMeasureMax,

	kCamMultiCapMeasureCount
};

enum CameraMultiCapError
{
	kCamMultiCapErrorIncluded = 0,
	kCamMultiCapErrorExcluded,

	kCamMultiCapErrorCount
};

enum CameraEnvMode
{
	kCamEnvFrequency50Hz = 0,
	kCamEnvFrequency60Hz
};

enum CameraHDRMode
{
	kCamHDRMode,
	kCamHDRModeHighContrast,

	kCamHDRModeCount
};

enum CSUpdateMode
{
	kCamSettingUpdateInfo			= 0x001,
	kCamSettingUpdateTrigger		= 0x002,
	kCamSettingUpdateCalib			= 0x200,
	kCamSettingUpdateExposure		= 0x004,
	kCamSettingUpdateFocus			= 0x008,
	kCamSettingUpdateMultiCapture	= 0x010,
	kCamSettingUpdateHDR			= 0x020,
	kCamSettingUpdateGeometric		= 0x040,
	kCamSettingUpdateColor			= 0x080,
	kCamSettingUpdateCorrection		= 0x100,
	kCamSettingUpdateShutter		= 0x200,
	kCamSettingUpdateStrobe			= 0x400,
	kCamSettingUpdateAll			= 0xFFF,
	kCamSettingUpdateAllCtrl		= (kCamSettingUpdateAll - kCamSettingUpdateInfo),
};

enum CamFocusType
{
	kFocusTop,
	kFocusCenter,
	kFocusBot,
	kCamFocusCount
};

enum CamShutterTypes
{
	kShutterRolling,
	kShutterGlobal,
	kShutterTypeCount
};

enum CamStrobeLevelTypes
{
	kStrobeLevelLow,
	kStrobeLevelHigh,
	kStrobeLevelCount
};

//////////////////////////////////////////////////////////////////////////
#pragma pack(push, 4)

class CameraDevice
{
public:
	CameraDevice();
	~CameraDevice();

public:
	i32							m_cam_type;
	postring					m_cam_name;
	BlobData					m_cam_blob;
	i32							m_cam_reserved[4];
};

class CameraInfo : public CLockGuard
{
public:
	CameraInfo();
	~CameraInfo();

	void						init();
	void						reset();
	void						release();

	CameraInfo					getValue();
	void						setValue(CameraInfo& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	i32							getMaxChannel();

	inline i32					getMaxWidth() { lock_guard(); return m_max_width; };
	inline i32					getMaxHeight() { lock_guard(); return m_max_height; };
	inline u32					getCapability() { lock_guard(); return m_cam_capability; };
	inline i32					getReserved(i32 index) { lock_guard(); return m_cam_reserved[index]; };

public:
	postring					m_cam_name;
	u32							m_cam_capability;	// CamCapSupport

	i32							m_max_width;
	i32							m_max_height;

	f32							m_gain_min;
	f32							m_gain_max;
	f32							m_exposure_min;
	f32							m_exposure_max;
	f32							m_brightness_min;
	f32							m_brightness_max;

	f32							m_red_gain_min;
	f32							m_red_gain_max;
	f32							m_green_gain_min;
	f32							m_green_gain_max;
	f32							m_blue_gain_min;
	f32							m_blue_gain_max;

	f32							m_gamma_min;
	f32							m_gamma_max;
	f32							m_contrast_min;
	f32							m_contrast_max;
	f32							m_saturation_min;
	f32							m_saturation_max;
	f32							m_sharpness_min;
	f32							m_sharpness_max;

	BlobData					m_cam_blob;
	i32							m_cam_reserved[2];
};

class CameraTrigger : public CLockGuard
{
public:
	CameraTrigger();

	void						init();
	void						reset();

	CameraTrigger				getValue();
	void						setValue(CameraTrigger& trigger);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	bool						isEqual(CameraTrigger& other);
	bool						isSoftTrigger();
	bool						isExtTrigger();
	bool						isManualTrigger();

	inline u8					getTriggerMode()	{ lock_guard(); return m_trigger_mode; };
	inline i32					getTriggerDelay()	{ lock_guard(); return m_trigger_delay; };
	inline i32					getTriggerUnit()	{ lock_guard(); return m_trigger_unit; };
	inline i32					getTriggerSignal()	{ lock_guard(); return m_trigger_signal; };
	inline i32					getTriggerInterval(){ lock_guard(); return m_trigger_interval; };

public:
	u8							m_trigger_mode;
	u32							m_trigger_delay;
	u32							m_trigger_interval;
	u8							m_trigger_signal;
	u8							m_trigger_unit;
};

class CameraState : public CLockGuard
{
public:
	CameraState();

	void						init();
	void						reset();

	CameraState					getValue();
	void						setValue(CameraState& cam_state);

	void						setCameraFocus(f32 focus);
	void						clearFocusHistory();

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	i32							m_capture_width;
	i32							m_capture_height;

	bool						m_is_autoexp_mode;
	f32							m_exposure;
	f32							m_gain;

	f32							m_rgain;
	f32							m_ggain;
	f32							m_bgain;

	f32							m_focus;
	f32							m_focus_min;
	f32							m_focus_max;
};

class CameraExposure : public CLockGuard
{
public:
	CameraExposure();

	void						init();
	void						reset();
	void						updateValidation(CameraInfo* cam_info_ptr);

	CameraExposure				getValue();
	void						setValue(CameraExposure& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	CamAutoMode					m_autoexp_mode;			// Auto Exposure - CamAutoMode
	f32							m_autogain_min;
	f32							m_autogain_max;
	f32							m_autoexp_min;
	f32							m_autoexp_max;
	f32							m_auto_brightness;
	Recti						m_autoexp_window;

	f32							m_gain;
	f32							m_exposure;
};

class CameraFocus : public CLockGuard
{
public:
	CameraFocus();

	void						init();
	void						reset();

	CameraFocus					getValue();
	void						setValue(CameraFocus& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	bool						m_auto_focus;
	Recti						m_auto_focus_region;
	i32							m_focus_pos;
	i32							m_focus_length;
};

class CameraColor : public CLockGuard
{
public:
	CameraColor();

	void						init();
	void						reset();
	void						updateValidation(CameraInfo* cam_info_ptr);

	CameraColor					getValue();
	void						setValue(CameraColor& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	i32							getChannel();

	void						setColorMode(i32 color_mode);
	inline i32					getColorMode() { lock_guard(); return m_color_mode; };
	inline bool					isColorMode() { lock_guard(); return m_color_mode > kCamColorGray; };

public:
	i32							m_color_mode;
	bool						m_wb_mode;

	f32							m_red_gain;
	f32							m_green_gain;
	f32							m_blue_gain;
};

class CameraCorrection : public CLockGuard
{
public:
	CameraCorrection();

	void						init();
	void						reset();
	void						updateValidation(CameraInfo* cam_info_ptr);

	CameraCorrection			getValue();
	void						setValue(CameraCorrection& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	f32							m_gamma;
	f32							m_contrast;
	f32							m_saturation;
	f32							m_sharpness;
};

class CameraMultiCap : public CLockGuard
{
public:
	CameraMultiCap();

	void						init();
	void						reset();

	CameraMultiCap				getValue();
	void						setValue(CameraMultiCap& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	bool						m_enable_multicap;
	i32							m_capture_count;
	u8							m_measure_mode;
	u8							m_error_mode;
};

class CameraHDR : public CLockGuard
{
public:
	CameraHDR();

	void						init();
	void						reset();

	CameraHDR					getValue();
	void						setValue(CameraHDR& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	bool						m_enable_hdr;
	u8							m_hdr_mode;

	i32							m_min_hdr;
	i32							m_max_hdr;
	i32							m_hdr_input;
};

class CameraRange : public CLockGuard
{
public:
	CameraRange();

	void						init();
	void						reset();
	void						updateValidation(CameraInfo* cam_info_ptr);

	CameraRange					getValue();
	void						setValue(CameraRange& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	inline u8					getRotation() { lock_guard(); return m_rotation; };
	inline Recti				getRange()	{ lock_guard(); return m_range; };

	inline bool					isFlipX()	{ lock_guard(); return m_is_flip_x; };
	inline bool					isFlipY()	{ lock_guard(); return m_is_flip_y; };
	inline bool					isInvert()	{ lock_guard(); return m_is_invert; };

public:
	Recti						m_range;
	u8							m_rotation;

	bool						m_is_flip_x;
	bool						m_is_flip_y;
	bool						m_is_invert;
};

class CameraSpec : public CLockGuard
{
public:
	CameraSpec();

	void						init();
	void						reset();
	void						updateValidation();

	CameraSpec					getValue();
	void						setValue(CameraSpec& other);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	inline i32					getAntiFlickMode() { lock_guard(); return m_anti_flick; };
	inline i32					getAmbientFreq() { lock_guard(); return m_ambient_freq; };
	inline i32					getJitterTimeus() { lock_guard(); return m_jitter_time; };
	inline i32					getShutterMode() { lock_guard(); return m_shutter_mode; };

public:
	//noise
	bool						m_noise_reduce;
	bool						m_anti_flick;
	i32							m_ambient_freq;	//hz

	//color
	u8							m_color_temperature;

	//shutter
	i32							m_jitter_time;	//us
	i32							m_shutter_mode;
};

class CameraStrobe : public CLockGuard
{
public:
	CameraStrobe();

	void						init();
	void						reset();

	CameraStrobe				getValue();
	void						setValue(CameraStrobe& other);
	void						setValue(CameraStrobe* other_ptr);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

	inline bool					isEnabled() { lock_guard(); return m_use_strobe; };
	inline bool					isAutoStrobeMode() { lock_guard(); return m_is_auto_strobe; };
	inline i32					getStrobeLevel() { lock_guard(); return m_strobe_level; };
	inline i32					getStrobePWMDelay() { lock_guard(); return m_strobe_pwm_delay; };
	inline i32					getStrobePWMWidth() { lock_guard(); return m_strobe_pwm_width; };

public:
	bool						m_use_strobe;

	bool						m_is_auto_strobe;
	i32							m_strobe_level;
	i32							m_strobe_pwm_delay;	//us
	i32							m_strobe_pwm_width; //us
};

class CameraSetting : public CLockGuard
{
public:
	CameraSetting();

	void						init();
	void						setValue(CameraSetting* other_ptr, i32 mode);

	void						getCameraResolution(i32& w, i32& h, i32& channel);
	void						setCameraTrigger(const CameraTrigger& trigger_mode);
	void						updateValidation();

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);
	void						memReadChanged(i32 flag, u8*& buffer_ptr, i32& buffer_size);

	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);
	
	/* 카메라핸들 -1과 카메라장치의 지적자 NULL의 의미는 동일하다.
	왜냐하면 카메라핸들을 int형으로 사용하는경우 초기값이 -1이기때문이다. */
	inline void*				getCamDevice() const{ return (m_cam_handle != -1) ? (void*)m_cam_handle : NULL; };

	inline const i32			getCamID() const		{ return m_cam_id; };
	inline const i32			getCamType() const		{ return m_cam_type; };
	inline const i32			getCamHandle() const	{ return m_cam_handle; };

	inline CameraInfo*			getCameraInfo()			{ return &m_cam_info; };
	inline CameraRange*			getCameraRange()		{ return &m_cam_range; };
	inline CameraState*			getCameraState()		{ return &m_cam_state; };
	inline CameraTrigger*		getCameraTrigger()		{ return &m_cam_trigger; };
	inline CameraExposure*		getCameraExposure()		{ return &m_cam_exposure; };
	inline CameraColor*			getCameraColor()		{ return &m_cam_color; };
	inline CameraCorrection*	getCameraCorrection()	{ return &m_cam_correction; };
	inline CameraSpec*			getCameraSpec()			{ return &m_cam_spec; };
	inline CameraStrobe*		getCameraStrobe()		{ return &m_cam_strobe; };

	inline CameraCalib*			getCameraCalib()		{ return &m_cam_calib; };

	inline bool					isReady()				{ return m_cam_ready; };
	inline bool					isUsed()				{ return m_cam_used; };
	inline bool					isUnusedReady()			{ return m_cam_ready && !m_cam_used; };
	inline bool					isExtTrigger() 			{ return m_cam_trigger.isExtTrigger(); };
	inline bool					isManualTrigger()		{ return m_cam_trigger.isManualTrigger(); };
	inline bool					isColorMode()			{ return m_cam_color.isColorMode(); };

	inline bool					supportFunc(i32 mode)	{ return (m_cam_info.m_cam_capability & mode) == mode; };

	inline void					setUsed()				{ m_cam_used = true; };
	inline void					setUnused()				{ m_cam_used = false; };

public:
	i32							m_cam_id;			// Camera ID
	i64							m_cam_handle;		// Camera Device Handle
	i32							m_cam_type;			// Camera Device Type

	CameraInfo					m_cam_info;
	CameraTrigger				m_cam_trigger;		// Trigger Info
	CameraExposure				m_cam_exposure;		// Exposure Info
	CameraFocus					m_cam_focus;		// Focus Info
	CameraColor					m_cam_color;		// Color Info
	CameraCorrection			m_cam_correction;	// Correction Info
	CameraMultiCap				m_cam_multicap;		// Multi-Capture Info
	CameraHDR					m_cam_hdr;			// HDR Info
	CameraRange					m_cam_range;		// Image Range Info
	CameraSpec					m_cam_spec;			// Else spec
	CameraStrobe				m_cam_strobe;
	CameraState					m_cam_state;
	CameraCalib					m_cam_calib;

	bool						m_cam_ready;
	bool						m_cam_used;		
};
typedef std::vector<CameraSetting> CameraArray;

//////////////////////////////////////////////////////////////////////////
class CameraSet : public CLockGuard
{
public:
	CameraSet();
	~CameraSet();

	void						init();
	void						import(CameraSet& other);

	bool						isReady(i32 index);

	i32							findCamIndex(postring& cam_dev);
	i32							getAvailableCamType();
	void						setAvailableCamType(i32 cam_available);

	CameraSetting*				findFirstCamSetting();
	CameraSetting*				getCamSetting(i32 index);

	i32							memSize();
	i32							memRead(u8*& buffer_ptr, i32& buffer_size);
	i32							memWrite(u8*& buffer_ptr, i32& buffer_size);

	i32							memExportSize();
	bool						memImport(u8*& buffer_ptr, i32& buffer_size);
	bool						memExport(u8*& buffer_ptr, i32& buffer_size);

	bool						fileRead(FILE* fp);
	bool						fileWrite(FILE* fp);

public:
	i32							m_cam_available;
	postring					m_cam_id[PO_CAM_COUNT];
	CameraSetting				m_cam_setting[PO_CAM_COUNT];
};

#pragma pack(pop)

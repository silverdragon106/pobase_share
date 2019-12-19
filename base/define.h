#pragma once

#include "config.h"
#include "types.h"
#include "po_action.h"
#include "po_error.h"
#include "po_info.h"
#include <math.h>
#include <mutex>
#include <QString>

#if defined(POR_WINDOWS)
  #if defined(POR_DEBUG)
    #define po_new new( _NORMAL_BLOCK , __FILE__ , __LINE__ )
  #else
    #define po_new new
  #endif
  #define po_sprintf(a, b, ...)		_snprintf_s(a, b, _TRUNCATE, __VA_ARGS__)
#elif defined(POR_LINUX)
  #define po_new new
  #define po_sprintf(a, b, ...)		snprintf(a, b, __VA_ARGS__)
#endif

//////////////////////////////////////////////////////////////////////////
#if !defined(PO_CUR_DEVICE)
#define PO_CUR_DEVICE				0
#define PO_NETWORK_PORT				12345
#define PO_LOG_FILESIZE				5000000//about 4MB
#endif

#if !defined(PO_DEVICE_ID)
#define PO_DEVICE_ID				1001
#define PO_DEVICE_NAME				"POPorIPDev"
#define PO_DEVICE_MODELNAME			"PO2017FD"
#define PO_DEVICE_VERSION			"1.00"
#define PO_DEVICE_CAMPORT			"UNKPORT"
#endif

#if !defined(PO_DATABASE_FILENAME)
#define PO_DATABASE_FILENAME		"po_sqlite.db"
#define PO_MYSQL_HOSTNAME			"localhost"
#define PO_MYSQL_DATABASE			"po_db"
#define PO_MYSQL_USERNAME			"root"
#define PO_MYSQL_PASSWORD			"123456"
#define PO_MYSQL_REMOTEUSERNAME		"po_user"
#define PO_MYSQL_REMOTEPASSWORD		"123456"
#define PO_MYSQL_PORT				3306
#define PO_MYSQL_RECLIMIT			500
#define PO_MYSQL_LOGLIMIT			100000
#endif

#if !defined(PO_PLAINIO_TCP_PORT)
#define PO_PLAINIO_TCP_PORT				49210
#define PO_PLAINIO_UDP_PORT				49212
#define PO_PLAINIO_PORT					"COM2"
#define PO_PLAINIO_BAUDBAND				115200
#define PO_PLAINIO_DATABITS				8
#define PO_PLAINIO_PARITY				0
#define PO_PLAINIO_STOPBITS				1
#define PO_PLAINIO_FLOWCTRL				0
#endif

#if !defined(PO_MODNET_TCP_PORT)
#define PO_MODNET_TCP_PORT			502
#define PO_MODNET_UDP_PORT			504
#endif

#if !defined(PO_MODBUS_ADDRESS)
#define PO_MODBUS_ADDRESS			0
#define PO_MODBUS_PORT				"COM1"
#define PO_MODBUS_BAUDBAND			115200
#define PO_MODBUS_DATABITS			8
#define PO_MODBUS_PARITY			0
#define PO_MODBUS_STOPBITS			1
#define PO_MODBUS_FLOWCTRL			0
#endif

#if !defined(PO_IO_TIMEOUT)
#define PO_IO_TIMEOUT				1000
#define PO_IO_RETRY_COUNT			5
#endif

#if !defined(PO_FTP_HOSTNAME)
#define PO_FTP_HOSTNAME				"127.0.0.1"
#define PO_FTP_USERNAME				"anonymous"
#define PO_FTP_PASSWORD				""
#define PO_FTP_PORT					21
#endif

#if !defined(PO_OPC_PORT)
#define PO_OPC_PORT					4840
#endif

//define common constants
#define PO_MAXPATH					256
#define PO_MAXSTRING				65535
#define PO_MAXINT					0x7FFFFFFF
#define PO_MAXVAL					1E+20
#define PO_MINVAL					-1E+20
#define PO_TIMEOUT					3600000
#define PO_SHORT_TIMEOUT			3000
#define PO_EPSILON					1E-8
#define PO_DELTA					1E-2
#define PO_PI						3.14159265359f
#define PO_PI2						6.28318530718f
#define PO_PI_HALF					1.570796325f
#define PO_SIGN_CODE				0x1A2B3C5E
#define PO_SIGN_ENDCODE				0x7F8F9FBF
#define PO_SETTING_BEGIN_CODE		0xF2D2E2B3
#define PO_SETTING_END_CODE			0x4F5F6F78
#define PO_UPDATE_HEADER_CODE		0x16899087

//define constants for POApplications
#define PO_MAX_PERMISSION			32
#define PO_PASSWORD_MINLEN			6
#define PO_TOUCH_CALIB_POINTS		5
#define PO_TOUCH_CALIB_ERROR        100

#define POIO_CMDQUEUE				256
#define POIO_CMDMAXLEN				256
#define POIO_READSIZE				1024000
#define POIO_HALFREADSIZE			(POIO_READSIZE / 2)
#define POIO_SMALLREADSIZE			(POIO_READSIZE / 10)

#define PODB_MAX_ALARM				20
#define PODB_MAX_ACTION				20
#define PODB_MAX_INFO				40
#define POFTP_MAX_DEV				2
#define POFTP_MAX_IMAGE				1000

#define PODB_SQL_MODE				"MYSQL"
#define PODB_SQLITE_MODE			"SQLITE"

//preprocesser define for coding style 
#define POSAFE_DELETE(p)			{ if(p) { delete (p);     (p)=NULL; } }
#define POSAFE_DELETE_ARRAY(p)		{ if(p) { delete[] (p);   (p)=NULL; } }
#define POSAFE_RELEASE(p)			{ if(p) { (p)->Release(); (p)=NULL; } }

#define POSAFE_CLEAR(x)				{ i32 _p_index, _p_count = (i32)x.size(); \
									  for (_p_index = 0; _p_index < _p_count; _p_index++) \
									  { \
										if (x[_p_index]) delete x[_p_index]; \
									  } \
									  x.clear(); \
									}

#define po_max(a,b)					((a) > (b) ? (a) : (b))
#define po_min(a,b)					((a) < (b) ? (a) : (b))
#define po_sgn(a)					(((a) >= 0) ? 1 : -1)

#define QThreadStart()				if (!isRunning()) {start();}
#define QThreadStop()				if (isRunning() && !isFinished()) {wait(PO_TIMEOUT); if (!isFinished()) { terminate(); wait(); }}
#define QThreadStop1(x)				if (isRunning() && !isFinished()) {wait(x); if (!isFinished()) { terminate(); wait(); }}
#define QThreadStop2(a, x)			if (a->isRunning() && !a->isFinished()) {a->wait(x); if (!a->isFinished()) { a->terminate(); a->wait(); }}

#define QEventLoopStop()			if (isRunning()) {QThread::exit(0); QThread::wait(PO_TIMEOUT);}
#define QEventLoopStop1(a)			if (a->isRunning()) {a->exit(0); a->wait(PO_TIMEOUT);}

#define POMutex						std::recursive_mutex
#define POMutexLocker				std::lock_guard<std::recursive_mutex>

#if defined(POR_SUPPORT_UNICODE)
    #define tfopen                  _wfopen
    #define tremove                 _wremove
    #define fromTCharArray          fromWCharArray
    #define toStdTString            toStdWString

  #if !defined(_T)
    #define _T(str)                 L##str
  #endif

#else
    #define tfopen                  fopen
    #define tremove                 remove
    #define fromTCharArray          fromUtf8
    #define toStdTString            toStdString

  #if !defined(_T)
    #define _T(str)                 str
  #endif
#endif

//define enum types
enum POApplication
{
	kPOAppNone = 0,
	kPOAppCVIM,
	kPOAppMVS,
	kPOAppSC
};

enum PacketCmdType
{
	kPOCmdNone = 0,

	//request packet_ptr types
	kPOCmdHeartBeat,				// 상,하위기망통신련결을 진행하기 위한 HeartBeat명령
	kPOCmdPing,						// 상,하위기사이의 망련결의 정확성을 검사하기위하여 주기적으로 Ping명령이다.

	kPOCmdAuth = 10,				// 하위기를 리용하기위한 사용자인증을 진행하는 명령
	kPOCmdPreConnect,				// 하위기접속을 위하여 미리 전송되는 명령으로써 현재 하위기가 암호인증을 요구하는가 안하는가를 되돌린다.
	kPOCmdConnect,					// 하위기접속을 위한 명령
	kPOCmdChangeIP,					// 하위기의 IP를 변경시키는 명령
	kPOCmdChangePwd,				// 하위기의 인증암호를 변경시키는 명령
	kPOCmdDongleUnplug,				// 상위기가 Dongle을 리용하는 경우, Dongle인증이 실패하였음을 통보

	kPOCmdSync = 20,				// 하위기와의 동기화명령
	kPOCmdUnSync,					// 하위기와의 동기화해제명령
	kPOCmdWake,						// 하위기장치 절전상태해제명령
	kPOCmdSleep,					// 하위기장치 절전명령
	kPOCmdReboot,					// 하위기장치 재시동명령
	kPOCmdPowerOff,					// 하위기장치 끄기명령
	kPOCmdDevImport,				// 외부자료로부터 하위기장치설정을  진행하는 명령
	kPOCmdDevExport,				// 하위기장치설정을 외부자료로 보관하는 명령
	kPOCmdDevUpdate,				// 하위기를 갱신명령
	kPOCmdDevFactoryReset,			// 하위기설정을 공장초기값으로 복귀하는 명령
	kPOCmdDevTouchCalibration,		// 하위기장치의 Touch화면을 교정하는 명령
	kPOCmdActionNotice,				// 사용자조작로그를 위하여 하위기에 명령으로 통지하는 않고 진행된 사용자조작을 알려주는 명령
	kPOCmdEmulator,					// 모의기조종(play, stop, ex...)
	KPOCmdChangeDebugLevel,			// 하위기의 데바그준위를 변경시키는 명령

	//response packet_ptr types
	kPOCmdReady = 40,				// TCP접속이 이루어진후에 하위기가 상위기와 련결준비가 되였다는것을 통지하는 명령
	kPOCmdDevUpdated,				// 하위기가 정확히 갱신되였음을 통지하는 명령

	//IVS request and response packet_ptr types
	kIVSCmdOnline = 200,			// 하위기를 실행상태로 변경시키는 명령(IVS)
	kIVSCmdOffline,					// 하위기를 비실행상태로 변경시키는 명령(IVS)
	kIVSCmdLogin,					// 하위기리용권한을 점유하는 명령(IVS)
	kIVSCmdLogout,					// 하위기리용권한을 해제하는 명령(IVS)
	kIVSCmdChangePwd,				// 하위기의 인증암호를 변경시키는 명령(IVS)
	kIVSCmdImport,					// 외부설정파일로부터 하위기장치설정을 진행하는 명령(IVS)
	kIVSCmdExport,					// 하위기장치설정값을 외부자료로 보관하는 명령(IVS)
	kIVSCmdConnect,					// 하위기조종을 위하여 련결을 확립하는 명령(IVS)
	kIVSCmdDisconnect,				// 하위기조종을 위한 련결을 해제하는 명령(IVS)
	kIVSCmdUpdate,					// 하위기를 갱신명령(IVS)
	kIVSCmdUpdateCancel,			// 하위기갱신을 취소시키는 명령(IVS)
	kIVSCmdUpdateConfirm,			// 하위기갱신을 확정하는 명령(IVS)

	kIVSCmdStatus = 220,			// 하위기상태를 IVS로 통지하는 명령

	kPOCmdExtend = 500,				// 프로젝트의 요구따르는 추가명령번호를 추가하기위한 명령시작번호
	kPOCmdCount = 1000				// 프로젝트의 요구따르는 추가명령의 최대명령번호
};

enum PacketSubType
{
	kPOSubTypeNone = 0,

	kPOSubTypeEmuSetImage = 10,		// 모의기에 화상자료(Raw자료)를 설정(LocalEmu방식)
	kPOSubTypeEmuSetPath,			// 모의기에 화상경로를 설정(LocalEmu방식)
	kPOSubTypeEmuPlay,				// 모의기를 Play(LocalEmu방식)
	kPOSubTypeEmuStop,				// 모의기를 Stop(LocalEmu방식)
	kPOSubTypeEmuInterval,			// 모의기 화상절환속도설정(LocalEmu방식)
	kPOSubTypeEmuSelected,			// 모의기에 사용자설정화상을 선택(LocalEmu방식)
	kPOSubTypeEmuSelectStop,		// 모의기에 사용자가설정한 화상을 선택하고 정지(LocalEmu방식)
	kPOSubTypeEmuThumb,				// 모의기에 설정된 화상렬자료(LocalEmu방식)

	kPOSubTypeEmuDownSample,		// 모의기에 한개의 화상자료전송(OneEmu방식)

	kPOSubTypeDevExportData = 20,	// 하위기설정자료
	
	kPOSubTypeAppUpdate	= 30,		// 하위기갱신명령
	kPOSubTypeAppUpdateCancel,		// 하위기갱신취소명령

	kPOSubTypeExtend = 100,
};

/* PacketSubFlagType은 파케트자료구조의 0번째 예약int형에 존재한다. */
enum PacketSubFlagType
{
	// sub request command for camera setting
	kPOSubFlagCamGeoInvert				= 0x0000001,
	kPOSubFlagCamGeoFlip				= 0x0000002,
	kPOSubFlagCamGeoRotation			= 0x0000004,
	kPOSubFlagCamGeoRange				= 0x0000008,

	kPOSubFlagCamGain					= 0x0000010,
	kPOSubFlagCamExposure				= 0x0000020,
	kPOSubFlagCamAEMode					= 0x0000040,
	kPOSubFlagCamAEGain					= 0x0000080,
	kPOSubFlagCamAEExposure				= 0x0000100,
	kPOSubFlagCamAEBrightness			= 0x0000200,
	kPOSubFlagCamAEWindow				= 0x0000400,
	kPOSubFlagCamColorMode				= 0x0000800,
	kPOSubFlagCamColorWBMode			= 0x0001000,
	kPOSubFlagCamColorWBAutoOnce		= 0x0002000,
	kPOSubFlagCamColorGain				= 0x0004000,
	kPOSubFlagCamCorrectionGamma		= 0x0008000,
	kPOSubFlagCamCorrectionContrast		= 0x0010000,
	kPOSubFlagCamCorrectionSaturation	= 0x0020000,
	kPOSubFlagCamCorrectionSharpness	= 0x0040000,
	kPOSubFlagCamShutter				= 0x0080000,
	kPOSubFlagCamStrobe					= 0x0100000,
	kPOSubFlagCamCtrl					= 0x0FFFFFF,

	kPOSubFlagCamInitFirst				= 0x1000000,
	kPOSubFlagCamSync					= 0x2000000,
	kPOSubFlagCamUnSync					= 0x4000000,
	kPOSubFlagCamClearFocus				= 0x8000000,

	kPOMixFlagCamExposure = 
					(kPOSubFlagCamGain | kPOSubFlagCamExposure | kPOSubFlagCamAEMode |
					kPOSubFlagCamAEGain | kPOSubFlagCamAEExposure |
					kPOSubFlagCamAEBrightness | kPOSubFlagCamAEWindow),
	kPOMixFlagCamColor = 
					(kPOSubFlagCamColorMode | kPOSubFlagCamColorWBMode |
					kPOSubFlagCamColorWBAutoOnce | kPOSubFlagCamColorGain),
	kPOMixFlagCamCorrection = 
					(kPOSubFlagCamCorrectionGamma| kPOSubFlagCamCorrectionContrast |
					kPOSubFlagCamCorrectionSaturation | kPOSubFlagCamCorrectionSharpness),
	kPOMixFlagCamStrobe = 
					(kPOSubFlagCamStrobe),
	kPOMixFlagCamRange = 
					(kPOSubFlagCamGeoInvert | kPOSubFlagCamGeoFlip | 
					kPOSubFlagCamGeoRotation | kPOSubFlagCamGeoRange)
};

enum HeaderType
{
	kPOPacketRequest = 0,
	kPOPacketRespOK,
	kPOPacketRespFail
};

enum HeartBeatType
{
	kPOHeartBeatUdp,
	kPOHeartBeatTcp
};

//////////////////////////////////////////////////////////////////////////
enum POSerialMode
{
	kPOSerialNone = 0,
	kPOSerialRs232,
	kPOSerialRs485,
	kPOSerialRs422,

	kPOSerialModeCount
};

enum PODesc
{
	kPODescDevice			= 0x0001,
	kPODescCamera			= 0x0002,
	kPODescLight			= 0x0004,
	kPODescDatabase			= 0x0008,
	kPODescUsbDongle		= 0x0020,
	kPODescIOManager		= 0x0040,

	kPODescSInstance		= 0x0100,
	kPODescGUIControl		= 0x0200,
	kPODescEmulator			= 0x0400,
	kPODescFileLog			= 0x0800,

	kPODescIPCStream		= 0x1000,
	kPODescVideoStream		= 0x2000,

	kPODescCmdServer		= 0x010000,
	kPODescIVSServer		= 0x020000,
	kPODescHeartBeat		= 0x040000,

	kPODescAllComponents	= 0xFFFFFF,

	kPODescNetInited		= 0x1000000,
	kPODescNetConnected		= 0x2000000,
	kPODescIPCConnected		= 0x4000000,
	kPODescHighLevel		= 0x8000000
};

enum PODevErrType
{
	kPODevErrNone = 0x00,
	kPODevErrConnect,
	kPODevErrTimeOut,
	kPODevErrProblem,
	kPODevErrUnsupport,
	kPODevErrException
};

enum POSubDevType
{
	kPOSubDevNone = 0x00,
	kPOSubDevDevice,
	kPOSubDevCamera,
	kPOSubDevIOModule,
	kPOSubDevDatabase
};

enum POCommunicationTypes
{
	kPOCommNetwork = 0,
	kPOCommIPC
};

enum POServerType
{
	kPOServerCmd = 0x01,
	kPOServerIVS = 0x02,
	kPOServerAll = 0x03
};

enum POPowerOffType
{
	kPOPowerSleep = 0,
	kPOPowerReboot,
	kPOPowerShutdown,
	kPOTerminateApp
};

enum PODevSetTypes
{
	kDevSetNameFlag = 0x01,
	kDevSetNetAddressFlag = 0x02,
	kDevRequestDateTimeFlag = 0x04,
	kDevSetDateTimeFlag = 0x08
};

enum POTrayIconType
{
	kPOTrayIconPrepare = 0,
	kPOTrayIconOnline,
	kPOTrayIconOffline,
	kPOTrayIconDisconnect
};

enum TerminatorTypes
{
	kPOTermCRLF = 0,
	kPOTermCR,
	kPOTermLF,

	kPOTermTypeCount
};

enum POCommDeviceType
{
	kPOCommNone				= 0x0000,
	kPOCommIOInternal		= 0x0000,
	kPOCommIOSerial			= 0x0001,
	kPOCommPlainSerial		= 0x0010,
	kPOCommPlainTcp			= 0x0020,
	kPOCommPlainUdp			= 0x0040,
	kPOCommModbusSerial		= 0x0100,
	kPOCommModbusTcp		= 0x0200,
	kPOCommModbusUdp		= 0x0400,
	kPOCommWebServer		= 0x1000,
	kPOCommOpc				= 0x2000,
	kPOCommFtp				= 0x4000,
	
	kPOCommAllMode			= 0xFFFF,

	kPOCommIO				= (kPOCommIOInternal | kPOCommIOSerial),
	kPOCommPlain			= (kPOCommPlainSerial | kPOCommPlainTcp | kPOCommPlainUdp),
	kPOCommModbus			= (kPOCommModbusSerial | kPOCommModbusTcp | kPOCommModbusUdp),

	kPOCommManager			= 0x100000
};

enum POPixelType
{
	kPOBackPixel			= 0x0000,
	kPOForePixel			= 0x0001,
	kPOEdgePixel			= 0x0002,

	kPOValidPixel			= 0xFFF0,
	kPOEdgeInner			= 0xFFFE,
	kPOEdgeOutter			= 0xFFFF,
};

enum POPixelOperTypes
{
	kPOPixelOperNone		= 0x00,
	kPOPixelOperSubPixel	= 0x01,
	kPOPixelOperOutterEdge	= 0x02,
	kPOPixelOperInnerEdge	= 0x04,
	kPOPixelOperClosedEdge	= 0x08,

	KPOPixelOperAllEdge		= (kPOPixelOperOutterEdge | kPOPixelOperInnerEdge)
};

enum POLanguageType
{
	kPOLangEnglish = 0,
	kPOLangChinese,
	kPOLangKorean,

	kPOLangCount
};

enum POCameraKind
{
	kPOCamUnknown		= 0x0000,
	kPOCamMindVision	= 0x0001,
	kPOCamBaslerPylon	= 0x0002,
	kPOCamEmulator		= 0x0004
};

enum POConnectionType
{
	kPOConnNone	= 0x00,
	kPOConnViewer = 0x01,
	kPOConnAdmin = 0x02,
	kPOConnCount
};

enum POModbusEndianType
{
	kPOModbusBigEndian = 0x00,
	kPOModbusLittleEndian = 0x01
};

enum POFTPConnType
{
	kPOConnStandardFTP = 0,
	kPOConnSecureFTP,
	kPOConnMitsubishiFTP
};

enum POSerialBaudRate
{
	kPOSerialBaudRate1200 = 1200,
	kPOSerialBaudRate2400 = 2400,
	kPOSerialBaudRate4800 = 4800,
	kPOSerialBaudRate9600 = 9600,
	kPOSerialBaudRateBaud19200 = 19200,
	kPOSerialBaudRateBaud38400 = 38400,
	kPOSerialBaudRateBaud57600 = 57600,
	kPOSerialBaudRateBaud115200 = 115200
};

enum POSerialDataBits
{
	kPOSerialDataBits5 = 5,
	kPOSerialDataBits6 = 6,
	kPOSerialDataBits7 = 7,
	kPOSerialDataBits8 = 8
};

enum POSerialParity
{
	kPOSerialParityNone = 0,
	kPOSerialParityOdd = 1,
	kPOSerialParityEven = 2,
	kPOSerialParityMark = 3,
	kPOSerialParitySpace = 4
};

enum POSerialStopBits
{
	kPOSerialStopBitsOne = 1,
	kPOSerialStopBitsTwo = 2,
	kPOSerialStopBitsOneAndHalf = 3
};

enum POSerialFlowControl
{
	kPOSerialFlowControlNone = 0,
	kPOSerialFlowControlSoftware = 1,
	kPOSerialFlowControlHardware = 2
};

enum POEncoderTypes
{
	kPOEncoderNone = 0,
	kPOEncoderIPCRaw,
	kPOEncoderNetworkRaw,
	kPOEncoderGStreamerMJpeg,
	kPOEncoderGStreamerH264,
	kPOEncoderFFMpegMJpeg,
	kPOEncoderFFMpegH264,
};

enum PODevInfoTypes
{
	kPODevInfoAll = 0,

	kPODevInfoSettingData,
	kPODevInfoJobDBData,
	kPODevInfoImportStart,
	kPODevInfoImportFinish,
	kPODevInfoExportStart,
	kPODevInfoExportFinish,
};

enum POAppUpdateCPTTypes
{
	kPOAppUpdateCPTUnknown = 0,
	kPOAppUpdateCPTInvalid,
	kPOAppUpdateCPTValid
};

enum POColorConvertType
{
	kPOColorCvt2Gray = 0,
	kPOColorCvt2RGB,
	kPOColorCvt2YUV,
	kPOColorCvt2HSV,
	kPOColorCvt2Red,
	kPOColorCvt2Green,
	kPOColorCvt2Blue,
	kPOColorCvt2Hue,
	kPOColorCvt2Saturation,
	kPOColorCvt2Intensity,

	kPOColorCvtTypeCount
};

enum POColorChannels
{
	kPOAnyChannels = -1,
	kPO1Channels = 1,
	kPOGrayChannels = 1,
	kPO2Channels = 2,
	kPO3Channels = 3,
	kPOYUVChannels = 3,
	kPORGBChannels = 3,
	kPO4Channels = 4,
	kPORGBXChannels = 4
};

enum POImageRotation
{
	kPORotation0 = 0,
	kPORotation90,
	kPORotation180,
	kPORotation270,

	kPORotationCount
};

enum POLineWidthTypes
{
	kPOLineWidthAuto = 0,
	kPOLineWidth1Pixels,
	kPOLineWidth2Pixels,
	kPOLineWidth3Pixels,
	kPOLineWidth4Pixels,
	kPOLineWidth5Pixels,
	kPOLineWidth6Pixels,
	kPOLineWidth7Pixels,
	kPOLineWidth8Pixels,
	kPOLineWidth9Pixels,
	kPOLineWidth10Pixels
};

enum POLineStyleTypes
{
	kPOLineStyleAuto = 0,
	kPOLineStyleLine,
	kPOLineStyleDot
};

enum POColorTypes
{
	kPOColorAuto = 0,
	kPOColorBlack,
	kPOColorWhite,
	kPOColorGray,
	kPOColorRed,
	kPOColorGreen,
	kPOColorBlue,
	kPOColorYellow,
	kPOColorPink
};

enum POShapeTypes
{
	kPOShapeNone = 0,
	kPOShapeLine,
	kPOShapeEdge,
	kPOShapeCircle,
	kPOShapeEllispe
};

enum POAlignMode
{
	kPOAlignCenter	= 0,
	kPOAlignTopLeft,
	kPOAlignTopRight,
	kPOAlignCenterLeft,
	kPOAlignCenterRight,
	kPOAlignBottomLeft,
	kPOAlignBottomRight,

	kPOAlignModeCount
};

namespace po
{
template <typename T, typename U> static inline T _min(T a, U b)
{
	return a < (T)b ? a : (T)b;
}

template <typename T, typename U> static inline T _max(T a, U b)
{
	return a > (T)b ? a : (T)b;
}

template <typename T> static inline i32 _sgn(T a)
{
	return (a >= 0) ? 1 : -1;
}

inline u16 swap_endian16(u16 x)
{
	return (x >> 8) | (x << 8);
};

inline u32 swap_endian32(u32 x)
{
	return (x >> 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x << 24);
}

inline u32 swap_endian32(u16* p)
{
	u32 x = *((u32*)p);
	return (x >> 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x << 24);
}

inline u64 swap_endian64(u16* p)
{
	u64 x = *((u64*)p);
	return ((x >> 56)
		| ((0x000000000000FF00) & (x >> 40)) | ((0x0000000000FF0000) & (x >> 24))
		| ((0x00000000FF000000) & (x >> 8)) | ((0x000000FF00000000) & (x << 8))
		| ((0x0000FF0000000000) & (x << 24)) | ((0x00FF000000000000) & (x << 40))
		| (x << 56));
};

inline u64 swap_endian64(u64 x)
{
	return ((x >> 56)
		| ((0x000000000000FF00) & (x >> 40)) | ((0x0000000000FF0000) & (x >> 24))
		| ((0x00000000FF000000) & (x >> 8)) | ((0x000000FF00000000) & (x << 8))
		| ((0x0000FF0000000000) & (x << 24)) | ((0x00FF000000000000) & (x << 40))
		| (x << 56));
};

inline i32 IpA2N(QString str)
{
	QStringList bytes = str.split(".");
	if (bytes.size() >= 4)
	{
		uchar ipbytes[4];
		ipbytes[0] = (uchar)bytes[3].toInt();
		ipbytes[1] = (uchar)bytes[2].toInt();
		ipbytes[2] = (uchar)bytes[1].toInt();
		ipbytes[3] = (uchar)bytes[0].toInt();
		return ipbytes[0] | ipbytes[1] << 8 | ipbytes[2] << 16 | ipbytes[3] << 24;
	}
	return 0;
}

inline postring IpN2A(i32 ip)
{
	postring str = "";
	str += std::to_string((unsigned char)(ip >> 24)) + ".";
	str += std::to_string((unsigned char)(ip >> 16)) + ".";
	str += std::to_string((unsigned char)(ip >> 8)) + ".";
	str += std::to_string((unsigned char)(ip));

	return str;
}

inline QString DurN2A(i32 secs)
{
	//secs /= 1000;
	i32 days = secs / 86400;
	i32 hours = (secs % 86400) / 3600;
	i32 mins = ((secs % 86400) % 3600) / 60;
	secs = secs % 60;

	if (days > 0 && hours > 0 && mins > 0)
	{
		return QString("%1d %2h %3m %4s").arg(days).arg(hours).arg(mins).arg(secs);
	}
	else if (hours > 0 && mins > 0)
	{
		return QString("%1h %2m %3s").arg(hours).arg(mins).arg(secs);
	}
	else if (mins > 0)
	{
		return QString("%1m %2s").arg(mins).arg(secs);
	}
	return QString("%1s").arg(secs);
}
};

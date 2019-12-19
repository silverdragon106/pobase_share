#include "usb_dongle.h"
#include "logger/logger.h"
#include "base.h"
#include "proc/cryptograph/md5.h"
#include "proc/cryptograph/sha512.h"

const i32 kDongleLisenseCount = 3;
const u32 kDongleLicenseID[kDongleLisenseCount] =
{
#if defined(PO_PROJECT_ID1)	//1st LisenceID
	PO_PROJECT_ID1,
#else	
    0,
#endif
#if defined(PO_PROJECT_ID2)	//2nd LisenceID
	PO_PROJECT_ID2,
#else	
    0,
#endif
#if defined(PO_PROJECT_ID3)	//3rd LisenceID
    PO_PROJECT_ID3
#else	
    0
#endif
};

const i32 kSSApiChkInterval = 3000; //about 3s
const u8 kSSApiPassword[16] = { 0xA8, 0x9A, 0xB0, 0xFA, 0x68, 0x43, 0x6B, 0xDA,
								0xE1, 0xA2, 0xF2, 0x0C, 0xF6, 0xAC, 0xFC, 0x59 };

#if defined(POR_WITH_DONGLE)
#if defined(POR_DEBUG)
#pragma comment(lib, "slm_runtime_api_dev.lib")
#else
#pragma comment(lib, "slm_runtime_api.lib")
#endif

SS_UINT32 SSAPI app_ss_msg_core(SS_UINT32 message, void* wparam, void* lparam)
{
	if (message == 0)
	{
		return SS_OK;
	}

	char err_message[1024] = { 0 };
	printf("Elite5 ss msg enter... message type:0x%08X\n", message);
	printlog_lv0(err_message);
	return SS_ERROR_DEBUG_FOUNDED;
}
#endif

CUSBDongle::CUSBDongle()
{
	m_is_inited = false;
	m_is_thread_cancel = false;

#if defined(POR_WITH_DONGLE)
	m_hslm = 0;
#endif
	m_auth_str_id = "";
	m_auth_str_password = "";
}

CUSBDongle::~CUSBDongle()
{
	exitInstance();
}

//////////////////////////////////////////////////////////////////////////
// Open
// - Init Elite5 library for checking USB Dongle License.
// @apiPwd : API Password
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::initInstance()
{
	if (!m_is_inited)
	{
		m_is_inited = true;

#if defined(POR_WITH_DONGLE)
		ST_INIT_PARAM st_init_param = { 0 };

		//0. test slm_init
		st_init_param.version = SLM_CALLBACK_VERSION02;
		st_init_param.pfn = &(app_ss_msg_core);

		//1. set API password. you can get this API password on your website. it must be binary data as following.
		memcpy(st_init_param.password, kSSApiPassword, sizeof(SS_BYTE) * 16);

		//2. init sense shield.
		if (slm_init(&st_init_param) != SS_OK)
		{
			printlog_lv0("Dongle init error!");
			return false;
		}

		bool is_login = false;
		for (i32 i = 0; i < kDongleLisenseCount; i++)
		{
			if (kDongleLicenseID[i] > 0 && logIn(i, kDongleLicenseID[i]))
			{
				is_login = true;
				break;
			}
		}
		if (!is_login || !isAvailable())
		{
			printlog_lv0("Dongle InitInstance Error!");
			return false;
		}

		//check lock time
		u32 lock_time, pc_time;
		u8 rand_buffer[SLM_FIXTIME_RAND_LENGTH];
		u32 code = slm_adjust_time_request(m_hslm, rand_buffer, &lock_time, &pc_time);
		printlog_lv1(QString("lock time:%1, pc time:%2").arg(lock_time).arg(pc_time));

		if (code != SS_OK)
		{
			printlog_lv0("Dongle adjust time request error!");
			return false;
		}

		//change led control
		ST_LED_CONTROL st_ctrl;
		st_ctrl.index = 0; //blue
		st_ctrl.state = 2; //open
		st_ctrl.interval = 2000; //ms
		slm_led_control(m_hslm, &st_ctrl);
		printlog_lv0("Dongle InitOK.");

#elif defined(POR_WITH_AUTHENTICATION)
#if !defined(POR_PRODUCT)
		printlog_lv0(QString("Generated Authentication is [%1]")
				.arg(QString::fromStdString(getGenAuthentication())));
#endif
		CPOBase::toLower(m_auth_str_password);
		if (getGenAuthentication() != m_auth_str_password)
		{
			return false;
		}
#endif
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Close
// - Close Elite5 library and release USB Dongle Device.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::exitInstance()
{
	if (m_is_inited)
	{
		m_is_inited = false;
		
#if defined(POR_WITH_DONGLE)
		//restore led control
		ST_LED_CONTROL st_ctrl;
		st_ctrl.index = 0; //blue
		st_ctrl.state = 1; //open
		st_ctrl.interval = 0; //ms
		slm_led_control(m_hslm, &st_ctrl);

		logOut();
		slm_cleanup();
		printlog_lv0("Dongle ExitOK.");
#endif
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Login
// - Check USB dongle license.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::logIn(i32 index, u32 lic_id)
{
#if defined(POR_WITH_DONGLE)
	SS_UINT32 slm_code = SS_OK;
	ST_LOGIN_PARAM login_struct = { 0 };
	INFO_FORMAT_TYPE login_fmt;

	login_struct.license_id = lic_id;
	login_struct.size = sizeof(ST_LOGIN_PARAM);
	login_struct.timeout = 86400;
	login_struct.login_mode = SLM_LOGIN_MODE_LOCAL;
	login_fmt = STRUCT;

	// login
	slm_code = slm_login(&login_struct, login_fmt, &(m_hslm), NULL);
	if (slm_code != SS_OK)
	{
		printlog_lv0(QString("Dongle login error! license index:%1").arg(index));
		return false;
	}

	m_is_thread_cancel = false;
	QThreadStart();
#endif
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Logout
// - Logout USB dongle license.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::logOut()
{
#if defined(POR_WITH_DONGLE)
	m_is_thread_cancel = true;
	QThreadStop();

	SS_UINT32 slm_code = SS_ERROR;
	if (m_hslm != 0)
	{
		slm_code = slm_logout(m_hslm);
	}
	return slm_code == SS_OK;
#else
	return true;
#endif
}

//////////////////////////////////////////////////////////////////////////
// ReadData
// - Read data from usb dongle device.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::readData()
{
	// if error, then Stop();
	return false;
}

//////////////////////////////////////////////////////////////////////////
// WriteData
// - Write data from usb dongle device.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::writeData()
{
	// if error, then Stop();
	return false;
}

//////////////////////////////////////////////////////////////////////////
// Encrypt Data
// - Encrypt data
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::encryptData()
{
	// if error, then Stop();
	return false;
}

//////////////////////////////////////////////////////////////////////////
// Decrypt Data
// - Decrypt data
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::decryptData()
{
	// if error, then Stop();
	return false;
}

//////////////////////////////////////////////////////////////////////////
// IsAvailable
// - Check USB Dongle License is available.
//////////////////////////////////////////////////////////////////////////
bool CUSBDongle::isAvailable()
{
#if defined(POR_WITH_DONGLE)
	if (m_hslm == 0)
	{
		return false;
	}
	return (slm_keep_alive(m_hslm) == SS_OK);
#else
	return true;
#endif
}

void CUSBDongle::run()
{
	singlelog_lv0("The USBDongle thread is");

	while (!m_is_thread_cancel)
	{
		if (!isAvailable())
		{
			printlog_lv0("Dongle is not available now...");
			emit licenseError();
			break;
		}
		QThread::msleep(kSSApiChkInterval);
	}
}

postring CUSBDongle::getGenAuthentication()
{
#if defined(POR_WITH_AUTHENTICATION)
	return md5_with_id_hex8(sha512_with_id(m_auth_str_id, PO_PROJECT_ID1), PO_PROJECT_ID1);
#else
	return postring();
#endif
}

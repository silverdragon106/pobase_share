#include "window_func.h"

#if defined(POR_WINDOWS)

#include "base.h"
#include "logger/logger.h"
#include <iphlpapi.h>
#include <stdio.h>
#include <time.h>

#define POW_MILISECOND		((i64) 10000)
#define POW_SECOND			((i64) 10000000)
#define POW_MINUTE			(60 * POW_SECOND)
#define POW_HOUR			(60 * POW_MINUTE)
#define POW_DAY				(24 * POW_HOUR)

i32 CWinBase::stringNCaseCmp(const char* s1, const char* s2)
{
    return _stricmp(s1, s2);
}

DateTime CWinBase::getSystemDateTime()
{
	SYSTEMTIME stm;
	GetSystemTime(&stm);
	return getDateTimeFromSysTime(stm);
}

DateTime CWinBase::addMsToCurrentTime(i64 ms_diff)
{
	FILETIME ftm;
	SYSTEMTIME stm, dtm;
	GetSystemTime(&stm);
	SystemTimeToFileTime(&stm, &ftm);

	i64 ftm64 = (((i64)ftm.dwHighDateTime) << 32) + ftm.dwLowDateTime;
	ftm64 += (ms_diff*POW_MILISECOND);
	ftm.dwLowDateTime = (DWORD)(ftm64 & 0xFFFFFFFF);
	ftm.dwHighDateTime = (DWORD)(ftm64 >> 32);

	FileTimeToSystemTime(&ftm, &dtm);
	return getDateTimeFromSysTime(dtm);
}

DateTime CWinBase::getDateTimeFromSysTime(SYSTEMTIME& stm)
{
	DateTime dtm;
	dtm.yy = stm.wYear;
	dtm.mm = stm.wMonth;
	dtm.dd = stm.wDay;
	dtm.h = stm.wHour;
	dtm.m = stm.wMinute;
	dtm.s = stm.wSecond;
	dtm.ms = stm.wMilliseconds;
	return dtm;
}

bool CWinBase::setSystemDateTime(const DateTime& dtm)
{
	SYSTEMTIME st;
	st.wYear = dtm.yy;				// set year
	st.wMonth = dtm.mm;				// set month
	st.wDay = dtm.dd;				// set day
	st.wHour = dtm.h;				// set hour
	st.wMinute = dtm.m;				// set minute
	st.wSecond = dtm.s;				// set second
	st.wMilliseconds = dtm.ms;		// set milisecond

	if (!processTokenPrivileges(SE_SYSTEMTIME_NAME))
	{
		printlog_lvs2(QString("CWinBase TokenPrivileges Failed. %1").arg(GetLastError()), LOG_SCOPE_APP);
		return false;
	}
	if (!SetSystemTime(&st))
	{
		printlog_lvs2(QString("CWinBase SetSystemTime Failed. %1").arg(GetLastError()), LOG_SCOPE_APP);
		return false;
	}
	return true;
}

bool CWinBase::processTokenPrivileges(LPCTSTR token_name)
{
	HANDLE token_handle;
	TOKEN_PRIVILEGES tkp;

	// Get a token for this process. 
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token_handle))
	{
		return false;
	}

	// Get the LUID for the shutdown privilege. 
	if (!LookupPrivilegeValue(NULL, token_name, &tkp.Privileges[0].Luid))
	{
		return false;
	}

	tkp.PrivilegeCount = 1;  // one privilege to set    
	tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

	// Get the shutdown privilege for this process. 
	AdjustTokenPrivileges(token_handle, FALSE, &tkp, sizeof(tkp), (PTOKEN_PRIVILEGES)NULL, 0);
	return (GetLastError() == ERROR_SUCCESS);
}

void CWinBase::sleep(u32 sleep_ms)
{
	::Sleep(sleep_ms);
}

bool CWinBase::executeProcess(const char* process_name, const char* current_path)
{
	PROCESS_INFORMATION processInfo = { 0 };
	STARTUPINFOA startupInfo = { 0 };
	startupInfo.cb = sizeof(startupInfo);

	//Create the process
	return CreateProcessA(NULL, (LPSTR)process_name, NULL, NULL, FALSE, NORMAL_PRIORITY_CLASS, NULL, 
		(LPSTR)current_path, &startupInfo, &processInfo);
}

bool CWinBase::setHostName(const postring& host_name)
{
	if (!SetComputerNameA(host_name.c_str()))
	{
		printlog_lvs2(QString("CWinBase SetHostName Failed. ErrCode:%1").arg(GetLastError()) , LOG_SCOPE_APP);
		return false;
	}
	return true;
}

bool CWinBase::sleepSystem()
{
	printlog_lv0("SleepSystem is not implement yet.");
	return true;
}

bool CWinBase::rebootSystem()
{
	if (!processTokenPrivileges(SE_SHUTDOWN_NAME))
	{
		return false;
	}
	if (!ExitWindowsEx(EWX_REBOOT | EWX_FORCEIFHUNG, SHTDN_REASON_MAJOR_OPERATINGSYSTEM | SHTDN_REASON_FLAG_PLANNED))
	{
		return false;
	}
	return true;
}

bool CWinBase::poweroffSystem()
{
	if (!processTokenPrivileges(SE_SHUTDOWN_NAME))
	{
		return false;
	}
	if (!ExitWindowsEx(EWX_POWEROFF | EWX_FORCEIFHUNG, SHTDN_REASON_MAJOR_OPERATINGSYSTEM | SHTDN_REASON_FLAG_PLANNED))
	{
		return false;
	}
	return true;
}

bool CWinBase::getNetworkAdapters(NetAdapterArray& adapter_vec)
{
	adapter_vec.clear();

	DWORD err_code;
	PIP_ADAPTER_INFO adapter_info_ptr, adapter_ptr;
	DWORD adapter_info_size;
	PIP_ADDR_STRING addr_str_ptr;

	// Enumerate all of the adapter specific information using the IP_ADAPTER_INFO structure.
	// Note:  IP_ADAPTER_INFO contains a linked list of adapter entries.
	adapter_info_size = 0;
	if ((err_code = GetAdaptersInfo(NULL, &adapter_info_size)) != 0)
	{
		if (err_code != ERROR_BUFFER_OVERFLOW)
		{
			printlog_lv1(QString("GetAdaptersInfo() sizing failed with error %1").arg(err_code));
			return false;
		}
	}

	// Allocate memory from sizing information
	if ((adapter_info_ptr = (PIP_ADAPTER_INFO)GlobalAlloc(GPTR, adapter_info_size)) == NULL)
	{
		printlog_lv1("Memory allocation error!");
		return false;
	}

	// Get actual adapter information
	if ((err_code = GetAdaptersInfo(adapter_info_ptr, &adapter_info_size)) != 0)
	{
		printlog_lv1(QString("GetAdaptersInfo() failed with error %1").arg(err_code));
		return false;
	}

	NetAdapter adapter;
	adapter_ptr = adapter_info_ptr;

	//add physical ip address
	adapter.init();
	bool is_found_adapter = false;

	while (adapter_ptr)
	{
		is_found_adapter = false;
		switch (adapter_ptr->Type)
		{
			case MIB_IF_TYPE_LOOPBACK:
			case MIB_IF_TYPE_ETHERNET:
			{
				is_found_adapter = true;
				break;
			}
			default:
			{
				break;
			}
		}
		
		if (is_found_adapter)
		{
			adapter.adapter_name = adapter_ptr->Description;
			adapter.is_conf_dhcp = adapter_ptr->DhcpEnabled;
			adapter.ip_gateway = CPOBase::convertIPAddress(adapter_ptr->GatewayList.IpAddress.String);
			adapter.ip_dns_server = CPOBase::convertIPAddress(adapter_ptr->PrimaryWinsServer.IpAddress.String);
			adapter.mac_address = (QString("%1:%2:%3:%4:%5:%6")
										.arg(adapter_ptr->Address[0], 2, 16, QChar('0'))
										.arg(adapter_ptr->Address[1], 2, 16, QChar('0'))
										.arg(adapter_ptr->Address[2], 2, 16, QChar('0'))
										.arg(adapter_ptr->Address[3], 2, 16, QChar('0'))
										.arg(adapter_ptr->Address[4], 2, 16, QChar('0'))
										.arg(adapter_ptr->Address[5], 2, 16, QChar('0'))).toStdString();

			addr_str_ptr = &(adapter_ptr->IpAddressList);
			while (addr_str_ptr)
			{
				adapter.ip_address = CPOBase::convertIPAddress(addr_str_ptr->IpAddress.String);
				adapter.ip_subnet = CPOBase::convertIPAddress(addr_str_ptr->IpMask.String);
				if (adapter.isValid())
				{
					adapter_vec.push_back(adapter);
				}
				addr_str_ptr = addr_str_ptr->Next;
			}
		}
		adapter_ptr = adapter_ptr->Next;
	}

	//add puseduo loopback 
	adapter.init();
	adapter.adapter_name = "LookBack";
	adapter.ip_address = 0x7F000001;
	adapter.ip_subnet = 0xFF000000;
	adapter.ip_gateway = 0;
	adapter.ip_dns_server = 0;
	adapter.is_loopback = true;
	adapter.is_conf_dhcp = false;
	adapter_vec.push_back(adapter);
	return true;
}

bool CWinBase::setNetworkAddress(i32 ip_prev_address, bool is_conf_dhcp, i32 ip_address, i32 ip_submask,
							i32 ip_gateway, i32 ip_dns_server)
{
	DWORD err_code;
	DWORD adapter_info_size;
	PIP_ADAPTER_INFO adapter_info_ptr, adapter_ptr;
	PIP_ADDR_STRING addr_str_ptr;

	// Enumerate all of the adapter specific information using the IP_ADAPTER_INFO structure.
	// Note:  IP_ADAPTER_INFO contains a linked list of adapter entries.
	adapter_info_size = 0;
	if ((err_code = GetAdaptersInfo(NULL, &adapter_info_size)) != 0)
	{
		if (err_code != ERROR_BUFFER_OVERFLOW)
		{
			printlog_lv1(QString("GetAdaptersInfo() sizing failed with error %1").arg(err_code));
			return false;
		}
	}

	// Allocate memory from sizing information
	if ((adapter_info_ptr = (PIP_ADAPTER_INFO)GlobalAlloc(GPTR, adapter_info_size)) == NULL)
	{
		printlog_lv1("Memory allocation error!");
		return false;
	}

	// Get actual adapter information
	if ((err_code = GetAdaptersInfo(adapter_info_ptr, &adapter_info_size)) != 0)
	{
		printlog_lv1(QString("GetAdaptersInfo() failed with error %1").arg(err_code));
		return false;
	}

	bool is_found = false;
	bool is_found_adapter = false;
	i32 adap_ip_address, net_index;

	adapter_ptr = adapter_info_ptr;
	while (adapter_ptr)
	{
		is_found_adapter = false;
		switch (adapter_ptr->Type)
		{
			case MIB_IF_TYPE_LOOPBACK:
			case MIB_IF_TYPE_ETHERNET:
			{
				is_found_adapter = true;
				break;
			}
			default:
			{
				break;
			}
		}

		if (is_found_adapter)
		{
			addr_str_ptr = &(adapter_ptr->IpAddressList);
			while (addr_str_ptr)
			{
				adap_ip_address = CPOBase::convertIPAddress(addr_str_ptr->IpAddress.String);
				if (adap_ip_address == ip_prev_address)
				{
					is_found = true;
					break;
				}
				addr_str_ptr = addr_str_ptr->Next;
			}

			if (is_found)
			{
				net_index = adapter_ptr->Index;
				break;
			}
		}
		adapter_ptr = adapter_ptr->Next;
	}

	if (is_found)
	{
		if (is_conf_dhcp)
		{
			postring cmd_argv = "/C netsh interface ip set address " + std::to_string(net_index) + " dhcp";
			::ShellExecuteA(NULL, NULL, "cmd.exe", cmd_argv.c_str(), NULL, SW_HIDE);

			cmd_argv = "/C netsh interface ip set dns " + std::to_string(net_index) + " dhcp";
			::ShellExecuteA(NULL, NULL, "cmd.exe", cmd_argv.c_str(), NULL, SW_HIDE);
		}
		else
		{
			postring ip_string, mask_ip_string, gateway_ip_string;
			CPOBase::convertIPAddress(ip_address, ip_string);
			CPOBase::convertIPAddress(ip_submask, mask_ip_string);
			CPOBase::convertIPAddress(ip_gateway, gateway_ip_string);
			postring cmd_argv = "/C netsh interface ip set address " + std::to_string(net_index) + " static ";
			cmd_argv += ip_string + " ";
			cmd_argv += mask_ip_string + " ";
			cmd_argv += gateway_ip_string;
			::ShellExecuteA(NULL, NULL, "cmd.exe", cmd_argv.c_str(), NULL, SW_HIDE);

			postring dns_ip_string;
			CPOBase::convertIPAddress(ip_dns_server, dns_ip_string);
			cmd_argv = "/C netsh interface ip set dns " + std::to_string(net_index) + " static ";
			cmd_argv += dns_ip_string;
			::ShellExecuteA(NULL, NULL, "cmd.exe", cmd_argv.c_str(), NULL, SW_HIDE);
		}
	}
	return true;
}

#if !defined(POR_DEVICE)
#include <powerbase.h>
#include <highlevelmonitorconfigurationapi.h>
#include "gamma_ramp.h"

#pragma comment(lib, "PowrProf.lib")
#pragma comment(lib, "Dxva2.lib")
bool CWinBase::setScreenBrightness(i32 brightness)
{
#if 1
	return false;
#else
	static GammaRamp gamma;
	if (gamma.LoadLibraryIfNeeded())
	{
		/* 128 is original brightness in other words brightness 80 is original */
		gamma.SetBrightness(NULL, brightness * 160 / 100);
	}
	return true;
#endif
}

bool CWinBase::setScreenTimeout(u32 time_out_ms)
{
#if 1
	return false;
#else
	SYSTEM_POWER_POLICY powerPolicy;
	u32 ret;
	u32 size = sizeof(SYSTEM_POWER_POLICY);

	ret = CallNtPowerInformation(SystemPowerPolicyAc, NULL, 0, &powerPolicy, size);

	if ((ret != ERROR_SUCCESS) || (size != sizeof(SYSTEM_POWER_POLICY)))
	{
		return false;
	}

	// if newtimeout is 0, then this function is disabled.
	/* ms to seconds */
	powerPolicy.VideoTimeout = time_out_ms / 1000;
	ret = CallNtPowerInformation(SystemPowerPolicyAc, &powerPolicy, size, NULL, 0);

	if ((ret != ERROR_SUCCESS))
	{
		return false;
	}

	return true;
#endif
}

bool CWinBase::forceScreenOff(bool b)
{
	singlelog_lv0("forceScreenOff-");
	// Don't use following instructions.
	// Because it freeze the application.
	//::SendMessage(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2);
	return true;
}
#endif
#endif

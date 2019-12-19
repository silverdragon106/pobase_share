#pragma once
#include "config.h"

#if defined(POR_WINDOWS)

#include "struct.h"
#include <windows.h>

class CWinBase
{
public:
	static i32              stringNCaseCmp(const char* s1, const char* s2);

	static DateTime			getSystemDateTime();
	static DateTime			addMsToCurrentTime(i64 ms_diff);
	static bool				setSystemDateTime(const DateTime& dtm);
	
	static bool				sleepSystem();
	static bool				rebootSystem();
	static bool				poweroffSystem();

	static void				sleep(u32 sleep_ms);
	static bool				executeProcess(const char* process_name, const char* current_path = NULL);

    static bool				setHostName(const postring& host_name);
	static bool				getNetworkAdapters(NetAdapterArray& adapter_vec);
	static bool				setNetworkAddress(i32 ip_prev_address, bool is_conf_dhcp, i32 ip_address,
										i32 ip_submask, i32 ip_gateway, i32 ip_dns_server);

#if !defined(POR_DEVICE)
	static bool				setScreenBrightness(i32 brightness/* [0, 100] */);
	static bool				setScreenTimeout(u32 time_out_ms /* miliseconds */ );
	static bool				forceScreenOff(bool b = true);
#endif
	
private:
	static DateTime			getDateTimeFromSysTime(SYSTEMTIME& stm);
	static bool				processTokenPrivileges(LPCTSTR token_name);
};
#endif

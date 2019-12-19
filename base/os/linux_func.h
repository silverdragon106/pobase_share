#pragma once

#include "config.h"

#ifdef POR_LINUX

#include "struct.h"
#include <initializer_list>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <QPoint>

#ifdef POR_IMVS2_ON_AM5728
extern "C"
{
#include "imvs2_middle_layer.h"
}
#endif

class CLinuxBase
{
public:
    static i32              stringNCaseCmp(const char* s1, const char* s2);

    static void             catchUnixSignals(std::initializer_list<i32> quitSignals);
    static void             ignoreUnixSignals(std::initializer_list<i32> ignoreSignals);
    static void             catchUnixSignals(std::initializer_list<i32> quitSignals, void(*quit_callback)(int));

    static DateTime			getSystemDateTime();
    static DateTime			addMsToCurrentTime(i64 ms_diff);
    static bool				setSystemDateTime(const DateTime& dtm);

    static bool				sleepSystem();
    static bool				rebootSystem();
    static bool				poweroffSystem();

    static bool				setHostName(const postring& host_name);
    static bool				getNetworkAdapters(NetAdapterArray& adapter_vec);
    static bool				setNetworkAddress(i32 ip_prev_address, bool is_conf_dhcp, i32 ip_address,
                                        i32 ip_submask, i32 ip_gateway, i32 ip_dns_server);

    static void				sleep(u32 sleep_ms);
    static bool				executeProcess(const i32 argc, char* argv[]);

    static bool				setSystemDateTime(u16 yy, u8 mm, u8 dd, u8 h, u8 m, u8 s);
    static bool				setScreenBrightness(i32 brightness/* [0, 100] */);
    static bool				setScreenTimeout(u32 time_out_ms /* miliseconds */);
    static bool				forceScreenOff(bool b=true);

#ifdef POR_IMVS2_ON_AM5728
    static i32              getStorageCount();
    static i32				getStorageInfo(iMvs2StorageInfo* storages, i32* cnt);

    static void             beepOn();
    static void             beepOff();
    static void				beepVolume(i32 vol);
    static void             beepDelay(i32 ms);
    static void             startTurnOffSplash();
    static void             stopTurnOffSplash();

    static bool				isTouchScreenCalibrated();
    static i32				startTouchScreenCalib();
    static vector2df		getLastTouchedAdcPos();
    static i32				finishTouchScreenCalib(float* coeff);
    static i32              cancelTouchScreenCalib();
    static void             syncPath(const char* path);
#endif
};
#endif

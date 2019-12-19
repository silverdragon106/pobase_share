#include "linux_func.h"
#include "logger/logger.h"

#ifdef POR_LINUX
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/reboot.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <linux/input.h>

#include "qt_base.h"
#include <QCoreApplication>

#define POW_MILISECOND		(0.001f)
#define POW_SECOND			(1)
#define POW_MINUTE			(60 * POW_SECOND)

i32 CLinuxBase::stringNCaseCmp(const char* s1, const char* s2)
{
    return strcasecmp(s1, s2);
}

void signalHandler(i32 sig)
{
    // blocking and not aysnc-signal-safe func are valid
    printlog_lv0(QString("Quit the application by signal(%1).").arg(sig));
    printlog_lv0("___________________________________________");
    QCoreApplication::quit();
}

void CLinuxBase::ignoreUnixSignals(std::initializer_list<i32> ignore_signals)
{
    // all these signals will be ignored.
    for (i32 sig : ignore_signals)
        signal(sig, SIG_IGN);
}

void CLinuxBase::catchUnixSignals(std::initializer_list<i32> quit_signals)
{
    sigset_t blocking_mask;
    sigemptyset(&blocking_mask);
    for (auto sig : quit_signals)
    {
        sigaddset(&blocking_mask, sig);
    }

    struct sigaction sa;
    sa.sa_handler = signalHandler;
    sa.sa_mask    = blocking_mask;
    sa.sa_flags   = 0;

    for (auto sig : quit_signals)
    {
        sigaction(sig, &sa, NULL);
    }
}

void CLinuxBase::catchUnixSignals(std::initializer_list<i32> quit_signals, void(*quit_callback)(int))
{
    sigset_t blocking_mask;
    sigemptyset(&blocking_mask);
    for (auto sig : quit_signals)
    {
        sigaddset(&blocking_mask, sig);
    }

    struct sigaction sa;
    sa.sa_handler = quit_callback;
    sa.sa_mask    = blocking_mask;
    sa.sa_flags   = 0;

    for (auto sig : quit_signals)
    {
        sigaction(sig, &sa, NULL);
    }
}

DateTime CLinuxBase::getSystemDateTime()
{
    time_t raw_time;
    time(&raw_time);

    struct tm* tm_ptr = gmtime(&raw_time);
    DateTime dtm;
    dtm.yy = tm_ptr->tm_year + 1900;
    dtm.mm = tm_ptr->tm_mon + 1;
    dtm.dd = tm_ptr->tm_mday;
    dtm.h = tm_ptr->tm_hour;
    dtm.m = tm_ptr->tm_min;
    dtm.s = tm_ptr->tm_sec;
    return dtm;
}

DateTime CLinuxBase::addMsToCurrentTime(i64 ms_diff)
{
    time_t raw_time;
    time(&raw_time);
    raw_time = raw_time + (i64)(ms_diff*POW_MILISECOND);

    struct tm* tm_ptr = gmtime(&raw_time);
    DateTime dtm;
    dtm.yy = tm_ptr->tm_year + 1900;
    dtm.mm = tm_ptr->tm_mon + 1;
    dtm.dd = tm_ptr->tm_mday;
    dtm.h = tm_ptr->tm_hour;
    dtm.m = tm_ptr->tm_min;
    dtm.s = tm_ptr->tm_sec;
    return dtm;
}

bool CLinuxBase::setSystemDateTime(const DateTime& dtm)
{
#if defined(POR_IMVS2_ON_AM5728)
    iMvs2DateTime idtm;
    idtm.yy = dtm.yy;
    idtm.mm = dtm.mm;
    idtm.dd = dtm.dd;
    idtm.h = dtm.h;
    idtm.m = dtm.m;
    idtm.s = dtm.s;
    i32 ret_code = iMvs2SetDateTime(&idtm);
    if (ret_code != kMLSuccess)
    {
        printlog_lv1(QString("SetTime is failed. errcode:%1").arg(ret_code));
        return false;
    }
#else
    struct tm time;
    memset(&time, 0, sizeof(time));

    time.tm_year = dtm.yy - 1900;
    time.tm_mon  = dtm.mm - 1;
    time.tm_mday = dtm.dd;
    time.tm_hour = dtm.h;
    time.tm_min  = dtm.m;
    time.tm_sec  = dtm.s;

    if (time.tm_year < 0)
    {
        time.tm_year = 0;
    }

    time_t raw_time = mktime(&time);

    if (raw_time == (time_t) -1)
    {
        printlog_lv1("SetTime is failed. time invalid");
        return false;
    }

    i32 ret_code = stime(&raw_time);
    if (ret_code < 0)
    {
        printlog_lv1(QString("SetTime is failed. errcode:%1").arg(ret_code));
        return false;
    }
#endif
    return true;
}

bool CLinuxBase::sleepSystem()
{
    sync();
    setuid(0);
    i32 ret_code = reboot(RB_SW_SUSPEND);
    if (ret_code < 0)
    {
        printlog_lv1(QString("Suspend is failed. errcode:%1").arg(ret_code));
        return false;
    }
    return true;
}

bool CLinuxBase::rebootSystem()
{
    sync();
    setuid(0);
    i32 ret_code = reboot(RB_AUTOBOOT);
    if (ret_code < 0)
    {
        printlog_lv1(QString("Reboot is failed. errcode:%1").arg(ret_code));
        return false;
    }
    return true;
}

bool CLinuxBase::poweroffSystem()
{
    sync();
    setuid(0);
    i32 ret_code = reboot(RB_POWER_OFF);
    if (ret_code < 0)
    {
        printlog_lv1(QString("PowerOff is failed. errcode:%1").arg(ret_code));
        return false;
    }
    return true;
}

bool CLinuxBase::executeProcess(const i32 argc, char* argv[])
{
    i32 pid = fork();
    if (pid == 0)
    {
        argv[argc] = NULL;
        execvp(argv[0], argv);
    }
    else if (pid < 0)
    {
        printlog_lv1(QString("Excute Process Failed. errcode:%1").arg(pid));
        return false;
    }

    printlog_lv1(QString("Excute Process: %1").arg(argv[0]));
    return true;
}

bool CLinuxBase::setHostName(const postring& host_name)
{
    if (!sethostname(host_name.c_str(), host_name.size()))
    {
        printlog_lvs2(QString("CWinBase SetHostName Failed. %1").arg(errno) , LOG_SCOPE_APP);
        return false;
    }
    return true;
}

bool CLinuxBase::getNetworkAdapters(NetAdapterArray& adapter_vec)
{
    return QTBase::getNetworkAdapters(adapter_vec);
}

bool CLinuxBase::setNetworkAddress(i32 ip_prev_address, bool is_conf_dhcp, i32 ip_address,
                                    i32 ip_submask, i32 ip_gateway, i32 ip_dns_server)
{
    return false;
}

bool CLinuxBase::setScreenBrightness(i32 brightness)
{
    return false;

#ifdef POR_IMVS2_ON_AM5728
    // conver [0,100] to [0,255]
    i32 bright = brightness / 100 * 255;
    iMvs2LCDSetBacklight(bright);
#else
    const char max_brightness_caps_file_path[] = "/sys/class/backlight/backlight/max_brightness";
    const char brightness_file_path[] = "/sys/class/backlight/backlight/brightness";

    i32 max_brightness = 0;		/* max brightness of display */
    i32 real_brightness = 0;	/* real brightness adopt to display. (= brightness * max_brightness / 100); */

    /* read display capabilities of brightness */
    /* /sys/class/backlight/.../backlight/max_brightness */

    FILE *max_brightness_caps_file = fopen(max_brightness_caps_file_path, "r");
    if (!max_brightness_caps_file)
    {
        //printf("Could not open max brightness file.\n");
        return false;
    }
    /* read max_brightness value */
    fscanf(max_brightness_caps_file, "%d", &max_brightness);
    fclose(max_brightness_caps_file);

    //printf("Max Brightness: %d\n", max_brightness);

    /* calc real brightness */
    real_brightness = brightness * max_brightness / 100;
    //printf("Real Brightness: %d\n", real_brightness);

    /* write real_brightness value */
    FILE *brightness_file = fopen(brightness_file_path, "w");
    if (!brightness_file)
    {
        //printf("Could not open brightness file.\n");
        return false;
    }

    fprintf(brightness_file, "%d", real_brightness);
    fclose(brightness_file);
#endif
    return true;
}

bool CLinuxBase::setScreenTimeout(u32 time_out_ms)
{
    /* must implement on linux operating system. */
    return true;
}

bool CLinuxBase::forceScreenOff(bool b)
{
#ifdef POR_IMVS2_ON_AM5728
    if (b)
    {
        iMvs2LCDOff();
    }
    else
    {
        iMvs2LCDOn();
    }

#else
    //////////////////////////////////////////////////////////////////////////
    // this is 5728 implementation
    //////////////////////////////////////////////////////////////////////////

    const char backlight_power_file_path[] = "/sys/class/backlight/backlight/bl_power";
    const char brightness_file_path[] = "/sys/class/backlight/backlight/brightness";


    i32 backlight_power_file = open(backlight_power_file_path, O_WRONLY);
    if (backlight_power_file == -1)
    {
        //printf("Could not open max brightness file.\n");
        return false;
    }
    char buf[256];
    buf[0] = b ? '1':'0';
    write(backlight_power_file, (const void*)buf, 1);
    close(backlight_power_file);

//	/* write real_brightness value */
//    int brightness_file = open(brightness_file_path, O_WRONLY);
//    if (brightness_file == -1)
//	{
//		//printf("Could not open brightness file.\n");
//		return false;
//	}

//	// off screen => '0'
//    // on screen => '1'
//    buf[0] = b ? '0' : '1';
//    write(brightness_file, (const void*)buf, 1);
//    close(brightness_file);
#endif
    return true;
}

#ifdef POR_IMVS2_ON_AM5728

void CLinuxBase::beepOn()
{
    iMvs2BeepOn();
}

void CLinuxBase::beepOff()
{
    iMvs2BeepOff();
}

void CLinuxBase::beepVolume(i32 vol)
{
    iMvs2BeepVolume(vol);
}

void CLinuxBase::beepDelay(i32 ms)
{
    iMvs2BeepDelay(ms);
}

void CLinuxBase::startTurnOffSplash()
{
    iMvs2TurnOffPsplashStart();
}

void CLinuxBase::stopTurnOffSplash()
{
    iMvs2TurnOffPsplashFinish();
}

i32 CLinuxBase::getStorageCount()
{
    iMvs2StorageInfo storages[EXT_STORAGE_COUNT_MAX];
    i32 cnt = 0;
    int n = 0;
    iMvs2GetStorageInfo(storages, &cnt);
    for (int i = 0; i < cnt; i++)
    {
        if ((storages[i].flag & STORAGE_IS_EXT) > 0)
        {
            n ++;
        }
    }
    return n;
}

i32 CLinuxBase::getStorageInfo(iMvs2StorageInfo* storages, i32* cnt)
{
    return iMvs2GetStorageInfo(storages, cnt);
}

bool CLinuxBase::isTouchScreenCalibrated()
{
    return iMvs2IsCalibrated() == 1;
}

i32	CLinuxBase::startTouchScreenCalib()
{
    return iMvs2CalibStart();
}

vector2df CLinuxBase::getLastTouchedAdcPos()
{
    f32 adc_x = 0, adc_y = 0;
    if (iMvs2CalibGetTouchVal(&adc_x, &adc_y, 300) != kMLFail) {
        return vector2df(adc_x, adc_y);
    }

    return vector2df();
}

i32 CLinuxBase::finishTouchScreenCalib(float* coeff)
{
    return iMvs2CalibFinish(coeff);
}

i32 CLinuxBase::cancelTouchScreenCalib()
{
    // return iMvs2CalibCancel();
    return 0;
}
void CLinuxBase::syncPath(const char *path)
{
    iMvs2Sync((char*)path);
}

#endif
#endif

#pragma once

#include "define.h"
#include <QString>
#include <QDateTime>

enum POECode
{
	kPOSuccess = 0,
	kPOErrCmdFail,
	kPOErrBusy,

	//connect & disconnect & security
	kPOErrSingleInstance = 10,
	kPOErrDongleCheck,
	kPOErrNewPwdLen,
	kPOErrInvalidConnect,
	kPOErrInvalidPassword,
	kPOErrInvalidOper,
	kPOErrInvalidData,
	kPOErrInvalidPacket,
	kPOErrInvalidCamSetting,
	kPOErrInvalidState,
	kPOErrInvalidSetting,
	kPOErrInvalidValue,
	kPOErrDuplicateObject,

	//device update
	kPOErrDeviceUpdateData = 40,
	kPOErrDeviceDupUpdate,
	kPOErrDeviceUpdateVer,
	kPOErrDeviceUpdateEmbedded,

	//streaming
	kPOErrStreamInit = 50,
	kPOErrStreamThrottle,
	kPOErrStreamEncoding,
	kPOErrIPCInit,
	kPOErrIPCThrottle,

	//device such as network, IO module & database & disk... 
	kPOErrNetThrottle = 60,
	kPOErrNetUnknown,
	kPOErrMBComOpen,
	kPOErrMBComThrottle,
	kPOErrMBComUnknown,
	kPOErrIOModuleOpen,
	kPOErrIOModuleUnknown,
	kPOErrDiskInit,
	kPOErrDiskCantRead,
	kPOErrDiskCantWrite,
	kPOErrDiskFull,
	kPOErrDiskUnknown,
	kPOErrDBModuleInit,
	kPOErrDBProcessBlocked,
	kPOErrDBUnknown,

	kPOErrExtend = 100
};

struct POAlarm
{
	POECode			err;
	QString			str_value;
	QDateTime		dtm;

public:
	POAlarm()
	{
		str_value = "";
		err = kPOSuccess;
	};
	POAlarm(POECode& code, const QString& str)
	{
		err = code;
		str_value = str;
		dtm = QDateTime::currentDateTime();
	};
};
typedef std::vector<POAlarm> POAlarmVec;

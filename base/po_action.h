#pragma once

#include "define.h"
#include <QString>
#include <QDateTime>

enum POACode
{
	kPOStringNone = 0,
	kPOStringExpr,
	kPOStringFalse, // based
	kPOStringTrue,
	kPOStringDisable, //based
	kPOStringEnable,
	kPOStringOFF, //based
	kPOStringON,

	kPOStringExtend = 10,

	kPOActionDevImport = 40,
	kPOActionDevExport,
	kPOActionDevUpdate,
	kPOActionDevResetFactory,
	kPOActionDevSleep,
	kPOActionDevReboot,
	kPOActionDevPowerOff,
	kPOActionDevInited,
	kPOActionDevUnInited,
	kPOActionDevPlugCamera,
	kPOActionDevPlugDatabase,
	kPOActionDevPlugDongle,
	kPOActionDevUnplugCamera,
	kPOActionDevUnplugDatabase,
	kPOActionDevUnplugDongle,
	kPOActionDevTerminateApp,

	kPOActionExtend = 100
};

struct POAction
{
	POACode			mode;
	POACode			pexpr;
	POACode			cexpr;
	QString			string_value;
	QDateTime		dtm;

public:
	POAction()
	{
		mode = kPOStringNone;
		pexpr = kPOStringNone;
		cexpr = kPOStringNone;
		string_value = "";
	};

	POAction(i32 mode)
	{
		this->mode = (POACode)mode;
		pexpr = kPOStringNone;
		cexpr = kPOStringNone;
		string_value = "";
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 index)
	{
		this->mode = (POACode)mode;
		pexpr = kPOStringNone;
		cexpr = kPOStringNone;
		string_value = QString("%1").arg(index);
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 prev, i32 cur)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)kPOStringExpr;
		this->cexpr = (POACode)kPOStringExpr;
		string_value = QString("%1,%2").arg(prev).arg(cur);
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, f32 prev, f32 cur)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)kPOStringExpr;
		this->cexpr = (POACode)kPOStringExpr;
		string_value = QString("%1,%2").arg(QString::number(prev, 'f', 2)).arg(QString::number(cur, 'f', 2));
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 index, i32 prev, i32 cur)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)kPOStringExpr;
		this->cexpr = (POACode)kPOStringExpr;
		string_value = QString("%1,%2,%3").arg(index).arg(prev).arg(cur);
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 index, f32 prev, f32 cur)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)kPOStringExpr;
		this->cexpr = (POACode)kPOStringExpr;
		string_value = QString("%1,%2,%3").arg(index).arg(QString::number(prev, 'f', 2)).arg(QString::number(cur, 'f', 2));
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 prev, i32 cur, POACode pstr, POACode cstr)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)pstr;
		this->cexpr = (POACode)cstr;
		string_value = QString("%1,%2").arg(prev).arg(cur);
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, POACode pstr, POACode cstr)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)pstr;
		this->cexpr = (POACode)cstr;
		string_value = "";
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, i32 index, POACode pstr, POACode cstr)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)pstr;
		this->cexpr = (POACode)cstr;
		string_value = QString("%1").arg(index);
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, POACode pstr, POACode cstr, const QString& str)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)pstr;
		this->cexpr = (POACode)cstr;
		string_value = str;
		dtm = QDateTime::currentDateTime();
	};

	POAction(i32 mode, const QString& str)
	{
		this->mode = (POACode)mode;
		this->pexpr = (POACode)kPOStringExpr;
		this->cexpr = (POACode)kPOStringExpr;
		this->string_value = str;
		dtm = QDateTime::currentDateTime();
	};
};

typedef std::vector<POAction> POActionVec;
#pragma once

#include "define.h"
#include <QString>
#include <QDateTime>

enum POICode
{
	kPOInfoNone = 0,

	kPOInfoExtend = 100
};

struct POInfo
{
	POICode			mode;
	QString			string_value;
	QDateTime		dtm;

public:
	POInfo()
	{
		mode = kPOInfoNone;
		string_value = "";
	};

	POInfo(i32 mode)
	{
		this->mode = (POICode)mode;
		string_value = "";
		dtm = QDateTime::currentDateTime();
	};

	POInfo(i32 mode, const QString& str)
	{
		this->mode = (POICode)mode;
		this->string_value = str;
		dtm = QDateTime::currentDateTime();
	};
};

typedef std::vector<POInfo>	POInfoVec;
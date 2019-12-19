#pragma once

#include <QString>
#include <QDateTime>
#include "define.h"
#include "struct.h"

class QTBase
{
public:
	QTBase();
	~QTBase();

public:
	//network
	static bool					getNetworkAdapters(NetAdapterArray& adapter_vec);

	//file, dir
	static bool					clearContents(const postring& dir_path);
	static bool					clearContents(const powstring& dir_path);
	static bool					clearContents(const QString& dir_path);

	static bool					addDir(const postring& str_path, i32 sub_id);
	static bool					copyDir(const QString& src_file_path, const QString& dst_file_path);
	static bool					copyFile(const QString& src_file_name, const QString& dst_file_name);
	static i32					getFileCount(const QString& dir_path, QString file_pattern);

	static bool					deleteMultiFiles(const postring& str_path, const postring& str_file_ext, i32 st_index);
	static bool					deleteMultiFiles(const postring& str_path, const postring& str_file_ext, i32 st_index, i32 base_pow);
	static bool					deleteSubDirMultiFiles(const postring& str_path, const postring& str_file_ext, i32 st_index, i32 base_pow);
	static bool					deleteMultiDirs(const postring& str_path, i32 sub_id);
	static bool					deleteMultiDirsNotIn(const postring& str_path, i32vector sub_id_vec);

	//datetime
	static DateTime				currentDateTime();

	static QString				convertToString(const QDateTime& dtm);
	static QString				convertToString(const DateTime& dtm);
	static DateTime				convertToDateTime(const QString& string);
	static DateTime				convertToDateTime(const QDateTime& dtm);
	static QDateTime			convertToQDateTime(const QString& string);
	static QDateTime			convertToQDateTime(const DateTime& dtm);
};
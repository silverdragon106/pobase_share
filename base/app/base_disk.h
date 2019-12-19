#pragma once
#include "define.h"

#define PO_ROOT_SUBPATH			"/"
#define PO_MODEL_SUBPATH		"/model"
#define PO_SAMPLE_SUBPATH		"/samples"
#define PO_MODELLIST_FILE		"/models.lst"
#define PO_SETTING_FILE			"/setting.dat"
#define PO_DEVICE_INIFILE		"/device.ini"

#define PO_UPDATE_LLPATH		"/update_ll"
#define PO_UPDATE_HLPATH		"/update_hl"
#define PO_UPDATE_MANAGER		"/update.exe"

class CPODisk
{
public:
	CPODisk();
	virtual ~CPODisk();

	virtual potstring			getAppPath() = 0;
	virtual potstring			getModelPath() = 0;
    virtual potstring			getDatabasePath(const postring& db_filename) = 0;

	public:
	static FILE*				fileOpen(const postring& file_path, const postring& open_mode);
	static void					fileClose(FILE* fp);

	static bool					isExistFile(const postring& dir_path);
	static bool					isExistDir(const postring& dir_path);
	static bool					makeDir(const postring& dir_path);
	static postring				getFilePath(const postring& filename);
	static postring				getNonExtPath(const postring& filename);
	static bool					rename(const postring& cur_filename, const postring& new_filename);
	static bool					deleteFile(const postring& cur_filename);

#if defined(POR_SUPPORT_UNICODE)
	static FILE*				fileOpen(const powstring& file_path, const postring& open_mode);

	static bool					isExistFile(const powstring& dir_path);
	static bool					isExistDir(const powstring& dir_path);
	static bool					makeDir(const powstring& dir_path);
	static powstring			getFilePath(const powstring& filename);
	static powstring			getNonExtPath(const powstring& filename);
	static bool					deleteFile(const powstring& cur_filename);
#endif
};

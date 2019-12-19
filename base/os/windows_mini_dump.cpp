#include "windows_mini_dump.h"
#include <time.h>

#if defined(POR_WINDOWS) && !defined(POR_WITH_DONGLE)
#include <dbghelp.h>

#pragma comment(lib,"dbghelp.lib")

CWinMiniDump::CWinMiniDump(void)
{
}

CWinMiniDump::~CWinMiniDump(void)
{
}

std::string CWinMiniDump::formatArgList(const char *fmt, va_list args)
{
	if (!fmt)
	{
		return "";
	}

	char *buffer = 0;
	i32 result = -1, length = 256;
	while (result == -1)
	{
		if (buffer)
		{
			delete[] buffer;
		}
		buffer = po_new char [length + 1];
		memset(buffer, 0, length + 1);
		result = _vsnprintf_s(buffer, length, _TRUNCATE, fmt, args);
		length *= 2;
	}
	std::string s(buffer);
	delete [] buffer;
	return s;
}

std::string CWinMiniDump::formatString(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	std::string s = formatArgList(fmt, args);
	va_end(args);

	return s;
}

std::wstring CWinMiniDump::wstringToString(const std::string& s)
{
	i32 len;
	i32 slength = (i32)s.length() + 1;
	len = ::MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = po_new wchar_t[len];
	::MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring wstr(buf);
	delete [] buf;
	return wstr;
}

std::wstring CWinMiniDump::getDumpFilename()
{
	time_t rawtime;
	struct tm timeinfo;

	std::string date_string;
	std::wstring date_wstring;

	std::wstring module_path;
	std::wstring dump_filename;

	static WCHAR str_module_path[1024];

	time(&rawtime);
	localtime_s(&timeinfo, &rawtime);

	date_string = formatString("%d.%02d.%02d.%02d.%02d.%02d", 
		timeinfo.tm_year + 1900,
		timeinfo.tm_mon + 1,
		timeinfo.tm_mday,
		timeinfo.tm_hour,
		timeinfo.tm_min,
		timeinfo.tm_sec);
	date_wstring = wstringToString(date_string);

	if (::GetModuleFileNameW(0, str_module_path, sizeof(str_module_path) /sizeof(WCHAR)) == 0)
	{
		return std::wstring();
	}

	module_path = str_module_path;
	dump_filename.assign(module_path, 0, module_path.rfind(L"\\") + 1);

	dump_filename = dump_filename + date_wstring + L".dmp";
	return dump_filename;
}

LONG WINAPI myTopLevelFilter(__in PEXCEPTION_POINTERS pExceptionPointer)
{
	CWinMiniDump win_mini_dump;
	MINIDUMP_EXCEPTION_INFORMATION MinidumpExceptionInformation;
	std::wstring dump_filename;

	MinidumpExceptionInformation.ThreadId = ::GetCurrentThreadId();
	MinidumpExceptionInformation.ExceptionPointers = pExceptionPointer;
	MinidumpExceptionInformation.ClientPointers = FALSE;
	
	dump_filename = win_mini_dump.getDumpFilename();
	if (dump_filename.empty())
	{
		::TerminateProcess(::GetCurrentProcess(), 0);
	}

	HANDLE hdump_file = ::CreateFileW(dump_filename.c_str(),
		GENERIC_WRITE, 
		FILE_SHARE_WRITE, 
		NULL, 
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);

	MiniDumpWriteDump(GetCurrentProcess(),
		GetCurrentProcessId(),
		hdump_file,
		MiniDumpNormal,
		&MinidumpExceptionInformation,
		NULL,
		NULL);
	::TerminateProcess(::GetCurrentProcess(), 0);

	return 0;
}

void CWinMiniDump::installSelfMiniDump()
{
	SetUnhandledExceptionFilter(myTopLevelFilter);
}

#endif
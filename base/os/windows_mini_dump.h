#pragma once
#include "define.h"

#if defined(POR_WINDOWS) && !defined(POR_WITH_DONGLE)

#include <windows.h>
#include <string>

LONG WINAPI myTopLevelFilter(__in PEXCEPTION_POINTERS pExceptionPointer);

class CWinMiniDump
{
public:
	CWinMiniDump(void);
	~CWinMiniDump(void);

	void				installSelfMiniDump();
	std::wstring		getDumpFilename();

private:
	std::string			formatArgList(const char *fmt, va_list args);
	std::string			formatString(const char *fmt, ...);
	std::wstring		wstringToString(const std::string& s);
};
#endif
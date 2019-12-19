#include <windows.h>
#include <process.h>
#include <Tlhelp32.h>
#include <winbase.h>
#include <string.h>
#include <tchar.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

#define PROCESS_LOWLEVEL		0
#define PROCESS_HIGHLEVEL		1
#define PROCESS_DELAYTIME		1000

#define MAX_LOOP				10

void checkProcess(int nProcessID, LPWSTR strProcessName, int nDelayTime = 0);
void killProcessByName(LPWSTR strProcessName);
bool executeProcess(LPWSTR strProcessName, LPWSTR strProcessPath);
bool packageUpdate(LPWSTR strCurrentPath, LPWSTR strUpdatePath);
bool deleteDirectory(const TCHAR* sDirPath);
bool isDots(const TCHAR* str);
bool isDirExists(const TCHAR* strDirName);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	//MessageBox(NULL, L"", L"", MB_OK);

	int nArgs = 0;
	LPWSTR* szArgument;
	TCHAR szProcessFilePath[MAX_PATH];

	//check argument list
	szArgument = CommandLineToArgvW(GetCommandLine(), &nArgs);
	if (szArgument == NULL)
	{
		return 0;
	}

	switch (nArgs)
	{
	case 4:
	{
		//argument0 - update.exe
		//argument1 - low-level current directory
		//argument2 - low-level update directory
		//argument3 - low-level executable filenmae

		//update with low-level only
		checkProcess(PROCESS_LOWLEVEL, szArgument[3], PROCESS_DELAYTIME);
		packageUpdate(szArgument[1], szArgument[2]);

		_stprintf(szProcessFilePath, L"%s\\%s", szArgument[1], szArgument[3]);
		executeProcess(szProcessFilePath, szArgument[1]);
		break;
	}
	case 7:
	{
		//argument0 - update.exe
		//argument1 - low-level current directory
		//argument2 - low-level update directory
		//argument3 - low-level executable filenmae
		//argument4 - high-level current directory
		//argument5 - high-level update directory
		//argument6 - high-level executable filenmae

		//update with low-level and embedded high-level
		checkProcess(PROCESS_LOWLEVEL, szArgument[3], PROCESS_DELAYTIME);
		checkProcess(PROCESS_HIGHLEVEL, szArgument[6]);
		packageUpdate(szArgument[1], szArgument[2]);
		packageUpdate(szArgument[4], szArgument[5]);

		_stprintf(szProcessFilePath, L"%s\\%s", szArgument[4], szArgument[6]);
		executeProcess(szProcessFilePath, szArgument[4]);
		break;
	}
	default:
		break;
	}

	LocalFree(szArgument);
	return 0;
}

void checkProcess(int nProcessID, LPWSTR strProcessName, int nDelayTime)
{
	Sleep(nDelayTime);

	switch (nProcessID)
	{
		case PROCESS_LOWLEVEL:
		{
			killProcessByName(strProcessName);
			break;
		}
		case PROCESS_HIGHLEVEL:
		{
			killProcessByName(strProcessName);
			break;
		}
	}
}

bool packageUpdate(LPWSTR strCurrentPath, LPWSTR strUpdatePath)
{
	if (strCurrentPath == NULL || strUpdatePath == NULL)
	{
		return false;
	}

	int loop = 0;
	while (!deleteDirectory(strCurrentPath) && loop < MAX_LOOP)
	{
		loop++;
		Sleep(500);
	}
	if (loop >= MAX_LOOP)
	{
		TCHAR szMesssage[MAX_PATH];
		_stprintf(szMesssage, L"Previous directory delete is failed, error code is %d", GetLastError());
		MessageBox(NULL, szMesssage, L"Error", MB_OK);
		return false;
	}

	loop++;
	while (!MoveFile(strUpdatePath, strCurrentPath) && loop < MAX_LOOP)
	{
		loop++;
		Sleep(500);
	}
	if (loop >= MAX_LOOP)
	{
		TCHAR szMesssage[MAX_PATH];
		_stprintf(szMesssage, L"Update Directory rename is failed, error code is %d", GetLastError());
		MessageBox(NULL, szMesssage, L"Error", MB_OK);
		return false;
	}
	return true;
}

void killProcessByName(LPWSTR strProcessName)
{
	HANDLE hSnapShot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, NULL);

	PROCESSENTRY32 pEntry;
	pEntry.dwSize = sizeof(pEntry);
	BOOL hRes = Process32First(hSnapShot, &pEntry);

	while (hRes)
	{
		if (_tcscmp(pEntry.szExeFile, strProcessName) == 0)
		{
			HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, 0, (DWORD)pEntry.th32ProcessID);
			if (hProcess != NULL)
			{
				TerminateProcess(hProcess, 9);
				CloseHandle(hProcess);
			}
		}
		hRes = Process32Next(hSnapShot, &pEntry);
	}
	CloseHandle(hSnapShot);
}

bool isDots(const TCHAR* str)
{
	if (_tcscmp(str, L".") && _tcscmp(str, L".."))
	{
		return false;
	}
	return true;
}

bool isDirExists(const TCHAR* strDirName)
{
	DWORD ntype = GetFileAttributes(strDirName);
	if (ntype == INVALID_FILE_ATTRIBUTES)
	{
		//something is wrong with your path!
		return false;
	}
	if (ntype & FILE_ATTRIBUTE_DIRECTORY)
	{
		//this is a directory!
		return true;
	}

	//this is not a directory!
	return false;
}

bool deleteDirectory(const TCHAR* sDirPath)
{
	HANDLE hFind;  //file handle
	WIN32_FIND_DATA FindFileData;

	TCHAR DirPath[MAX_PATH];
	TCHAR FileName[MAX_PATH];

	_tcscpy(DirPath, sDirPath);
	_tcscat(DirPath, L"\\*");    //searching all files
	_tcscpy(FileName, sDirPath);
	_tcscat(FileName, L"\\");

	hFind = FindFirstFile(DirPath, &FindFileData); //find the first file
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return true;
	}
	_tcscpy(DirPath, FileName);

	bool bSearch = true;
	while (bSearch) //until we finds an entry
	{ 
		if (FindNextFile(hFind, &FindFileData))
		{
			if (isDots(FindFileData.cFileName))
			{
				continue;
			}
			_tcscat(FileName, FindFileData.cFileName);
			if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				//we have found a directory, recurse
				if (!deleteDirectory(FileName))
				{
					FindClose(hFind);
					return false; //directory couldn't be deleted
				}
				RemoveDirectory(FileName); //remove the empty directory
				_tcscpy(FileName, DirPath);
			}
			else
			{
				if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_READONLY)
				{
					_tchmod(FileName, _S_IWRITE); //change read-only file mode
				}
				if (!DeleteFile(FileName)) //delete the file
				{
					FindClose(hFind);
					return false;
				}
				_tcscpy(FileName, DirPath);
			}
		}
		else
		{
			if (GetLastError() == ERROR_NO_MORE_FILES) //no more files there
			{
				bSearch = false;
			}
			else
			{
				//some error occured, close the handle and return FALSE
				FindClose(hFind);
				return false;
			}
		}
	}

	FindClose(hFind); //closing file handle
	if (!RemoveDirectory(sDirPath)) //remove the empty directory
	{
		int errcode = GetLastError();
		return false;
	}

	return true;
}

bool executeProcess(LPWSTR strProcessName, LPWSTR strProcessPath)
{
	PROCESS_INFORMATION processInfo = { 0 };
	STARTUPINFO startupInfo = { 0 };
	startupInfo.cb = sizeof(startupInfo);

	//Create the process
	if (!CreateProcess(NULL, strProcessName, NULL, NULL, FALSE, 
					NORMAL_PRIORITY_CLASS, NULL, strProcessPath, &startupInfo, &processInfo))
	{
		TCHAR szMesssage[MAX_PATH];
		_stprintf(szMesssage, L"Process execute is failed, error code is %d", GetLastError());
		MessageBox(NULL, szMesssage, L"Error", MB_OK);
		return false;
	}
	return true;
}
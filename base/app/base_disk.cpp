#include "base_disk.h"
#include "base.h"

#include <sys/stat.h> //stat
#include <errno.h>    //errno, ENOENT, EEXIST

#if defined(POR_WINDOWS)
#include <direct.h>   //_mkdir
#endif

CPODisk::CPODisk()
{
}

CPODisk::~CPODisk()
{
}

FILE* CPODisk::fileOpen(const postring& file_path, const postring& open_mode)
{
	return fopen(file_path.c_str(), open_mode.c_str());
}

void CPODisk::fileClose(FILE* fp)
{
	if (!fp)
	{
		return;
	}
	fclose(fp);
}

bool CPODisk::isExistFile(const postring& dir_path)
{
#if defined(POR_WINDOWS)
	struct _stat info;
	return (_stat(dir_path.c_str(), &info) == 0);
#else
	struct stat info;
	return (stat(dir_path.c_str(), &info) == 0);
#endif
}

bool CPODisk::isExistDir(const postring& dir_path)
{
#if defined(POR_WINDOWS)
	struct _stat info;
	if (_stat(dir_path.c_str(), &info) != 0)
	{
		return false;
	}
	return (info.st_mode & _S_IFDIR) != 0;
#else 
	struct stat info;
	if (stat(dir_path.c_str(), &info) != 0)
	{
		return false;
	}
	return (info.st_mode & S_IFDIR) != 0;
#endif
}

bool CPODisk::makeDir(const postring& dir_path)
{
#if defined(POR_WINDOWS)
	i32 ret = _mkdir(dir_path.c_str());
#else
	mode_t mode = 0755;
	i32 ret = mkdir(dir_path.c_str(), mode);
#endif
	if (ret == 0)
	{
		return true;
	}

	switch (errno)
	{
		//parent didn't exist, try to create it
		case ENOENT:
		{
			size_t pos = dir_path.find_last_of('/');
			if (pos == postring::npos)
			{
#if defined(POR_WINDOWS)
				pos = dir_path.find_last_of('\\');
			}
			if (pos == postring::npos)
			{
#endif
				return false;
			}
			if (!makeDir(dir_path.substr(0, pos)))
			{
				return false;
			}

			//now, try to create again
#if defined(POR_WINDOWS)
			return 0 == _mkdir(dir_path.c_str());
#else 
			return 0 == mkdir(dir_path.c_str(), mode);
#endif
		}
		case EEXIST:
		{
			//done!
			return isExistDir(dir_path);
		}
		default:
		{
			return false;
		}
	}
}

bool CPODisk::rename(const postring& cur_filename, const postring& new_filename)
{
	if (std::rename(cur_filename.c_str(), new_filename.c_str()))
	{
		return false;
	}
	return true;
}

bool CPODisk::deleteFile(const postring& cur_filename)
{
	return (std::remove(cur_filename.c_str()) == 0);
}

postring CPODisk::getFilePath(const postring& filename)
{
	size_t pos = filename.find_last_of("/\\");
	if (pos == postring::npos)
	{
		return postring();
	}
	return filename.substr(0, pos);
}

postring CPODisk::getNonExtPath(const postring& filename)
{
	size_t pos = filename.find_last_of(".");
	if (pos == postring::npos)
	{
		return postring();
	}
	return filename.substr(0, pos);
}

#if defined(POR_SUPPORT_UNICODE)
FILE* CPODisk::fileOpen(const powstring& file_path, const postring& open_mode)
{
#if defined(POR_WINDOWS)
	powstring wopen_mode = CPOBase::stringToWString(open_mode);
	return _wfopen(file_path.c_str(), wopen_mode.c_str());
#else
	return NULL;
#endif
}

bool CPODisk::isExistFile(const powstring& dir_path)
{
#if defined(POR_WINDOWS)
    struct _stat info;
    return (_wstat(dir_path.c_str(), &info) == 0);
#else
    struct stat info;
    return (stat(dir_path.c_str(), &info) == 0);
#endif
}

bool CPODisk::isExistDir(const powstring& dir_path)
{
#if defined(POR_WINDOWS)
    struct _stat info;
    if (_wstat(dir_path.c_str(), &info) != 0)
    {
        return false;
    }
    return (info.st_mode & _S_IFDIR) != 0;
#else
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0)
    {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
#endif
}

bool CPODisk::makeDir(const powstring& dir_path)
{
#if defined(POR_WINDOWS)
    i32 ret = _wmkdir(dir_path.c_str());
#else
    mode_t mode = 0755;
    i32 ret = mkdir(dir_path.c_str(), mode);
#endif
    if (ret == 0)
    {
        return true;
    }

    switch (errno)
    {
    //parent didn't exist, try to create it
    case ENOENT:
    {
        size_t pos = dir_path.find_last_of('/');
        if (pos == powstring::npos)
        {
#if defined(POR_WINDOWS)
            pos = dir_path.find_last_of('\\');
        }
        if (pos == powstring::npos)
        {
#endif
            return false;
        }
        if (!makeDir(dir_path.substr(0, pos)))
        {
            return false;
        }
        //now, try to create again
    #if defined(POR_WINDOWS)
        return 0 == _wmkdir(dir_path.c_str());
    #else
        return 0 == mkdir(dir_path.c_str(), mode);
    #endif
    }
    case EEXIST:
    {
        //done!
        return isExistDir(dir_path);
    }
    default:
        return false;
    }
}

powstring CPODisk::getFilePath(const powstring& filename)
{
    size_t pos = filename.find_last_of(L"/\\");
    if (pos == powstring::npos)
    {
        return powstring();
    }
    return filename.substr(0, pos);
}

powstring CPODisk::getNonExtPath(const powstring& filename)
{
	size_t pos = filename.find_last_of(L".");
	if (pos == powstring::npos)
	{
		return powstring();
	}
	return filename.substr(0, pos);
}

bool CPODisk::deleteFile(const powstring& cur_filename)
{
	return (tremove(cur_filename.c_str()) == 0);
}

#endif

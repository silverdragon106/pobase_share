#pragma once

#include "define.h"
#include "window_func.h"
#include "linux_func.h"
#include "qt_base.h"

#if defined(POR_WINDOWS)
    #define COSBase CWinBase
#elif defined(POR_LINUX)
    #define COSBase CLinuxBase
#endif

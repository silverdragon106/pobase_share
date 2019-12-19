#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "types.h"

#if defined(POR_WITH_CUDA)

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include "nvx_types.h"

#define CUDA_PO_MAXINT				0x7FFFFFFF
#define CUDA_PO_PI2					6.28318530718f
#define CUDA_PO_EPSILON				1E-8

#define CUDA_MAX_THREAD				1024
#define CUDA_THREAD_MAXW			32
#define CUDA_THREAD_MAXH			32

enum cuPOShapeTypes
{
	cuShapeNone = 0,
	cuShapeLine,
	cuShapeEdge,
	cuShapeCircle,
	cuShapeEllispe
};

enum cuPixelType
{
	cuBackPixel		= 0x0000,
	cuForePixel		= 0x0001,
	cuEdgePixel		= 0x0002,

	cuValidPixel	= 0xFFF0,
	cuEdgeInner		= 0xFFFE,
	cuEdgeOutter	= 0xFFFF,
};

#if defined(POR_WITH_CUDA_DLL)
    #define DLLEXPORT __declspec(dllexport)
#else
  #if defined(POR_USE_CUDA_DLL)
	#define DLLEXPORT __declspec(dllimport)
  #else
	#define DLLEXPORT
  #endif
#endif
#endif

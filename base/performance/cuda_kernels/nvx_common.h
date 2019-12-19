#pragma once
#include "nvx_base.h"

#if defined(POR_WITH_CUDA)
extern "C"
{
	DLLEXPORT vx_status cuHistogram(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
							i32* hist_ptr);
	DLLEXPORT vx_status cuHistogramEx(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect,
							i32 src_stride, i32 mask_stride, i32* hist_ptr, i32* border_hist_ptr);
}
#endif

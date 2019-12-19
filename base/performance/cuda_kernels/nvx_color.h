#pragma once
#include "nvx_base.h"

#if defined(POR_WITH_CUDA)
extern "C"
{
	DLLEXPORT vx_status cuPalette_u16(u16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* palette_ptr, i32 palette_size, u8* dst_img_ptr, i32 dst_stride);
}
#endif

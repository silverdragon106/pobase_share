#pragma once
#include "nvx_base.h"

#if defined(POR_WITH_CUDA)
extern "C"
{
	DLLEXPORT vx_status cuImage2RuntableInvert(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
							i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count);
	DLLEXPORT vx_status cuImage2RuntableWithMask(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
							u8* mask_img_ptr, i32 mask_stride,
							i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count);
	DLLEXPORT vx_status cuImage2RuntableWithMaskVal(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
							i32 mask_val, i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count);
	DLLEXPORT vx_status cuImage2Runtable(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
							i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count);

	DLLEXPORT vx_status cuRuntable2Image(u8* dst_img_ptr, i32 w, i32 h, i32 dst_stride,
							i32* pxy_ptr, u16* run2_ptr, i32 value);
}
#endif


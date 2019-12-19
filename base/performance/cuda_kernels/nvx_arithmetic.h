#pragma once
#include "nvx_base.h"

#if defined(POR_WITH_CUDA)

extern "C"
{
	DLLEXPORT vx_status cuAbs_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							i16* dst_img_ptr, i32 dst_stride);

	DLLEXPORT vx_status cuMul_u8u8u16(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
							u8* img2_ptr, i32 img2_stride,
							u16* dst_img_ptr, i32 dst_stride);

	DLLEXPORT vx_status cuAddConst_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* dst_img_ptr, i32 dst_stride, i32 value);
	DLLEXPORT vx_status cuAddConst_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							i16* dst_img_ptr, i32 dst_stride, i32 value);

	DLLEXPORT vx_status cuMulConstf32_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* dst_img_ptr, i32 dst_stride, f32 value);
	DLLEXPORT vx_status cuMulConsti32_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* dst_img_ptr, i32 dst_stride, i32 value);
	DLLEXPORT vx_status cuMulConstf32_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							i16* dst_img_ptr, i32 dst_stride, f32 value);
	DLLEXPORT vx_status cuMulConsti32_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							i16* dst_img_ptr, i32 dst_stride, i32 value);

	DLLEXPORT vx_status cuMin_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
							u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride);
	DLLEXPORT vx_status cuMax_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
							u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride);

	DLLEXPORT vx_status cuClipMin_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* dst_img_ptr, i32 dst_stride, i32 value);
	DLLEXPORT vx_status cuClipMax_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* dst_img_ptr, i32 dst_stride, i32 value);

	DLLEXPORT vx_status cuCut_u16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* mask_img_ptr, i32 mask_stride, i32 threshold, 
							i16* dst_img_ptr, i32 dst_stride, i32* valid_pixels);
	DLLEXPORT vx_status cuSubtractEx_u8(u8* a_img_ptr, vx_rectangle_t rect, i32 a_stride, f32 alpha,
							u8* b_img_ptr, i32 b_stride, f32 beta,
							u8* dst_img_ptr, i32 dst_stride);

	DLLEXPORT vx_status cuMask_u8_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
							u8* mask_img_ptr, i32 mask_stride,
							u8* dst_img_ptr, i32 dst_stride);
}
#endif

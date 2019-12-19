#pragma once
#include "config.h"
#include "base.h"

#if defined(POR_WITH_OVX)
#include "VX/vx.h"

//add, sub, mul, div, clip
vx_status	_vxAbs_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
					i16* dst_img_ptr, i32 dst_stride);
vx_status	_vxAddConst_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride, 
					u8* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxAddConst_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
				   i16* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxMul_u8u8u16(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u16* dst_img_ptr, i32 dst_stride);
vx_status	_vxMulConsti32_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxMulConstf32_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
						   u8* dst_img_ptr, i32 dst_stride, f32 value);
vx_status	_vxMulConsti32_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
					i16* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxMin_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride);
vx_status	_vxMax_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride);
vx_status	_vxCut_u16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* mask_img_ptr, i32 mask_stride, i32 threshold,
					i16* dst_img_ptr, i32 dst_stride, i32& valid_pixels);
vx_status	_vxClipMin_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxClipMax_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value);
vx_status	_vxSubtractEx_u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride, f32 alpha,
					u8* img2_ptr, i32 img2_stride, f32 beta, u8* dst_img_ptr, i32 dst_stride);
vx_status	_vxMask_u8_u8(u8* img_ptr, vx_rectangle_t rect, i32 img_stride,
					u8* mask_ptr, i32 mask_stride, u8* dst_img_ptr, i32 dst_stride);

//color
vx_status	_vxPalette_u16(u16* img_ptr, vx_rectangle_t rect, i32 img_stride,
					u8* palette_ptr, i32 palette_size, u8* dst_img_ptr, i32 dst_stride);

//histogram
vx_status	_vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect,
					i32 src_stride, i32 mask_stride, i32* hist_ptr, i32* border_hist_ptr);

vx_status	__vxHistogram(u8* img_ptr, vx_rectangle_t rect, i32 src_stride, i32* hist_ptr);
vx_status	__vxHistogram(u8* img_ptr, vx_rectangle_t rect, i32 src_stride, i32* hist_ptr, i32* border_hist_ptr);
vx_status	__vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect, i32 src_stride,
					i32 mask_stride, i32* hist_ptr);
vx_status	__vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect, i32 src_stride, i32 mask_stride,
					i32* hist_ptr, i32* border_hist_ptr);

//threshold
i32			_vxCalcThreshold(i32* hist_ptr, i32 th_min, i32 th_max);

//connect components
i32			_vxConnectComponents(u8* img_ptr, vx_rectangle_t src_rect, i32 src_stride,
					u8* mask_img_ptr, vx_rectangle_t mask_rect, i32 mask_stride,
					u16* label_img_ptr, vx_rectangle_t dst_rect, i32 dst_stride);

//else other
bool		_vxIsInvertedBackground(i32 th, i32* hist_ptr, i32* border_hist_ptr);

#endif
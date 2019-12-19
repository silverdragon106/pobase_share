#pragma once
#include "nvx_base.h"

#if defined(POR_WITH_CUDA)
extern "C"
{
	DLLEXPORT vx_status cuDrawCircle(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
							f32 cx, f32 cy, f32 r, u8 val);
	DLLEXPORT vx_status cuDrawEllipse(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
							f32 cx, f32 cy, f32 r1, f32 r2, u8 val);
	DLLEXPORT vx_status cuDrawRing(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
							f32 cx, f32 cy, f32 min_r, f32 max_r, f32 st_angle, f32 ed_angle, u8 val);
	DLLEXPORT vx_status cuDrawPolygon(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
							cuVector2df* poly_ptr, i32 count, u8 val);
}
#endif

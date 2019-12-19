#pragma once
#include "config.h"
#include "base.h"

#if defined(POR_WITH_OVX)
#include "VX/vx.h"

//filter2d
vx_status	_vxGaussian2d_u16(u16* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u16* dst_img_ptr, i32 dst_stride, i32 kernel_size);

#endif
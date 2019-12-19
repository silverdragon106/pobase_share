#include "nvx_filter.h"
#include "nvx_base.cuh"

vx_status cuGaussian2d_u16(u16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u16* dst_img_ptr, i32 dst_stride, i32 kernel_size)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr || kernel_size < 1)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	return VX_SUCCESS;
}

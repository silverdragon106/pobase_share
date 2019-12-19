#include "nvx_color.h"
#include "nvx_base.cuh"

__global__ void cuPaletteKernel_u16(u16* src_img_ptr, i32 w, i32 h, i32 src_stride, 
								u8* palette_ptr, i32 palette_size, u8* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u16* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		u16 index = *src_ptr;
		if (index < palette_size)
		{
			*dst_ptr = palette_ptr[index];
		}
	}
}

vx_status cuPalette_u16(u16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* palette_ptr, i32 palette_size, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!src_img_ptr || !palette_ptr || !dst_img_ptr || w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuPaletteKernel_u16 << <block, thread >> > (src_img_ptr, w, h, src_stride,
											palette_ptr, palette_size, dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

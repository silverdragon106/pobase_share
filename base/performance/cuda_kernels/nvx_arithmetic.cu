#include "nvx_arithmetic.h"
#include "nvx_base.cuh"

__global__ void cuAbs_i16_kernel(i16* src_img_ptr, i32 w, i32 h, i32 src_stride,
							i16* dst_img_ptr, i32 dst_stride)
{
    i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i16* scan_img_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		i16* scan_dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		*scan_dst_ptr = (i16)_cuAbs(*scan_img_ptr);
    }
}

vx_status cuAbs_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride, i16* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!src_img_ptr || w*h <= 0 || !dst_img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
    cuAbs_i16_kernel <<<block, thread>>>(src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

__global__ void cuMul_u8u8u16_kernel(u8* img1_ptr, i32 w, i32 h, i32 img1_stride,
								u8* img2_ptr, i32 img2_stride, u16* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		u8* scan_img1_ptr = img1_ptr + IMUL(ty, img1_stride) + tx;
		u8* scan_img2_ptr = img2_ptr + IMUL(ty, img2_stride) + tx;
		u16* scan_dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		*scan_dst_ptr = min((*scan_img1_ptr) * (*scan_img2_ptr), 0xFFFF);
	}
}

vx_status cuMul_u8u8u16(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u16* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!img1_ptr || !img2_ptr || w*h <= 0 || !dst_img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
    cuMul_u8u8u16_kernel <<<block, thread>>>(img1_ptr, w, h, img1_stride, img2_ptr, img2_stride,
										dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

__global__ void cuAddConstKernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
								u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		i32 pixel = min(*src_ptr + value, 0xFF);
		*dst_ptr = (u8)pixel;
	}
}

vx_status cuAddConst_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuAddConstKernel_u8 << <block, thread >> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuAddConstKernel_i16(i16* src_img_ptr, i32 w, i32 h, i32 src_stride,
								i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		i16* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		i16* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = max(min(*src_ptr + value, 0x7FFF), -0x7FFF);
		*dst_ptr = (i16)pixel;
	}
}

vx_status cuAddConst_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuAddConstKernel_i16 << <block, thread >> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuMulConstf32Kernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
								u8* dst_img_ptr, i32 dst_stride, f32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{

		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = min((i32)(*src_ptr * value), 0xFF);
		*dst_ptr = (u8)pixel;
	}
}

vx_status cuMulConstf32_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, f32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMulConstf32Kernel_u8 << <block, thread >> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuMulConsti32Kernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
								u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w&& ty < h)
	{
		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = min(IMUL(*src_ptr, value), 0xFF);
		*dst_ptr = (u8)pixel;
	}
}

vx_status cuMulConsti32_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMulConsti32Kernel_u8 << <block, thread>> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuMulConstf32Kernel_i16(i16* src_img_ptr, i32 w, i32 h, i32 src_stride,
									i16* dst_img_ptr, i32 dst_stride, f32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		i16* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		i16* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = min((i32)(*src_ptr * value), 0x7FFF);
		pixel = max(pixel, -0x7FFF);
		*dst_ptr = (i16)pixel;
	}
}

vx_status cuMulConstf32_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
						i16* dst_img_ptr, i32 dst_stride, f32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMulConstf32Kernel_i16 << <block, thread >> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuMulConsti32Kernel_i16(i16* src_img_ptr, i32 w, i32 h, i32 src_stride,
									i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		i16* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		i16* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = min(IMUL(*src_ptr, value), 0x7FFF);
		pixel = max(pixel, -0x7FFF);
		*dst_ptr = (i16)pixel;
	}
}

vx_status cuMulConsti32_i16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW ), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMulConsti32Kernel_i16 << <block, thread>> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuMin_u8u8u8_kernel(u8* img1_ptr, i32 w, i32 h, i32 img1_stride,
							u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		u8* scan_img1_ptr = img1_ptr + IMUL(ty, img1_stride) + tx;
		u8* scan_img2_ptr = img2_ptr + IMUL(ty, img2_stride) + tx;
		u8* scan_dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		*scan_dst_ptr = min(*scan_img1_ptr, *scan_img2_ptr);
	}
}

vx_status cuMin_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!img1_ptr || !img2_ptr || w * h <= 0 || !dst_img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMin_u8u8u8_kernel << <block, thread >> > (img1_ptr, w, h, img1_stride, img2_ptr, img2_stride,
											dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

__global__ void cuMax_u8u8u8_kernel(u8* img1_ptr, i32 w, i32 h, i32 img1_stride,
							u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		u8* scan_img1_ptr = img1_ptr + IMUL(ty, img1_stride) + tx;
		u8* scan_img2_ptr = img2_ptr + IMUL(ty, img2_stride) + tx;
		u8* scan_dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		*scan_dst_ptr = min(*scan_img1_ptr, *scan_img2_ptr);
	}
}

vx_status cuMax_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!img1_ptr || !img2_ptr || w * h <= 0 || !dst_img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMax_u8u8u8_kernel << <block, thread >> > (img1_ptr, w, h, img1_stride, img2_ptr, img2_stride,
											dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

__global__ void cuCutKernel_u16(i16* src_img_ptr, i32 w, i32 h, i32 src_stride,
						u8* mask_img_ptr, i32 mask_stride, i32 threshold,
						i16* dst_img_ptr, i32 dst_stride, i32* valid_pixels)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i16* scan_img_ptr;
		i16* scan_dst_ptr;
		u8* scan_mask_ptr;

		scan_img_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		scan_dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		scan_mask_ptr = mask_img_ptr + IMUL(ty, mask_stride) + tx;

		i32 value = *scan_img_ptr;
		if (value > threshold && *scan_mask_ptr != 0)
		{
			atomicAdd(valid_pixels, 1);
			*scan_dst_ptr = value;
		}
		else
		{
			*scan_dst_ptr = 0;
		}
	}
}

vx_status cuCut_u16(i16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
				u8* mask_img_ptr, i32 mask_stride, i32 threshold,
				i16* dst_img_ptr, i32 dst_stride, i32* valid_pixels)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));

	*valid_pixels = 0;
	i32* cu_valid_pixels;
	cudaMalloc(&cu_valid_pixels, sizeof(i32));
	cudaMemset(cu_valid_pixels, 0, sizeof(i32));
	cuCutKernel_u16 << <block, thread>> > (src_img_ptr, w, h, src_stride, mask_img_ptr, mask_stride,
										threshold, dst_img_ptr, dst_stride, cu_valid_pixels);
	cudaMemcpy(valid_pixels, cu_valid_pixels, sizeof(i32), cudaMemcpyDeviceToHost);
	cudaFree(cu_valid_pixels);
	return VX_SUCCESS;
}

__global__ void cuClipMinKernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
								u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = max(*src_ptr, value);
		*dst_ptr = pixel;
	}
}

vx_status cuClipMin_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW ), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuClipMinKernel_u8 << <block, thread>> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuClipMaxKernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
								u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = min(*src_ptr, value);
		*dst_ptr = pixel;
	}
}

vx_status cuClipMax_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuClipMaxKernel_u8 << <block, thread >> > (src_img_ptr, w, h, src_stride, dst_img_ptr, dst_stride, value);
	return VX_SUCCESS;
}

__global__ void cuSubtractExKernel_u8(u8* img1_ptr, i32 w, i32 h, i32 img1_stride, f32 alpha,
								u8* img2_ptr, i32 img2_stride, f32 beta, u8* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* src1_ptr = img1_ptr + IMUL(ty, img1_stride) + tx;
		u8* src2_ptr = img2_ptr + IMUL(ty, img2_stride)+ tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		i32 pixel = (i32)(alpha * (*src1_ptr) - beta * (*src2_ptr));
		pixel = max(pixel, 0);
		*dst_ptr = pixel;
	}
}

vx_status cuSubtractEx_u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride, f32 alpha,
					u8* img2_ptr, i32 img2_stride, f32 beta, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuSubtractExKernel_u8 << <block, thread >> > (img1_ptr, w, h, img1_stride, alpha, 
												img2_ptr, img2_stride, beta, dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

__global__ void cuMaskKernel_u8(u8* src_img_ptr, i32 w, i32 h, i32 src_stride,
							u8* mask_img_ptr, i32 mask_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* src_ptr = src_img_ptr + IMUL(ty, src_stride) + tx;
		u8* mask_ptr = mask_img_ptr + IMUL(ty, mask_stride) + tx;
		u8* dst_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		*dst_ptr = (*mask_ptr > 0) ? (*src_ptr) : 0;
	}
}

vx_status cuMask_u8_u8(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* mask_img_ptr, i32 mask_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuMaskKernel_u8 << <block, thread >> > (src_img_ptr, w, h, src_stride,
										mask_img_ptr, mask_stride, dst_img_ptr, dst_stride);
	return VX_SUCCESS;
}

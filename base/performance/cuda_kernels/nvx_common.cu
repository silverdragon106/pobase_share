#include "nvx_common.h"
#include "nvx_base.cuh"

__global__ void cuHistogram256Kernel(i32* cu_hist, u8* img_ptr, i32 w, i32 h, i32 img_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i32 pos = IMUL(ty, img_stride) + tx;
		u8 v = img_ptr[pos];

		atomicAdd(&cu_hist[v], 1);
	}
}

__global__ void cuHistogram256MaskKernel(i32* cu_hist, u8* img_ptr, u8* mask_ptr, i32 w, i32 h, i32 img_stride, i32 mask_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i32 pos2 = IMUL(ty, mask_stride) + tx;
		u8 mask = mask_ptr[pos2];
		if (mask != 0)
		{
			i32 pos1 = IMUL(ty, img_stride) + tx;
			u8 v = img_ptr[pos1];

			atomicAdd(&cu_hist[v], 1);
		}
	}
}

__global__ void cuHistogram256BorderKernel(i32* cu_hist, i32* cu_border_hist, u8* img_ptr, i32 w, i32 h, i32 img_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i32 pos = IMUL(ty, img_stride) + tx;
		u8 v = img_ptr[pos];
		if (v == cuEdgePixel)
		{
			atomicAdd(&cu_border_hist[v], 1);
		}
		atomicAdd(&cu_hist[v], 1);
	}
}

__global__ void cuHistogram256MaskBorderKernel(i32* cu_hist, i32* cu_border_hist,
					u8* src_img_ptr, u8* mask_ptr, i32 w, i32 h, i32 img_stride, i32 mask_stride)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	if (tx < w && ty < h)
	{
		i32 pos2 = IMUL(ty, mask_stride) + tx;
		u8 mask = mask_ptr[pos2];
		if (mask != 0)
		{
			i32 pos1 = IMUL(ty, img_stride) + tx;
			u8 v = src_img_ptr[pos1];
			if (mask == cuEdgePixel)
			{
				atomicAdd(&cu_border_hist[v], 1);
			}
			atomicAdd(&cu_hist[v], 1);
		}
	}
}

vx_status cuHistogram(u8* src_img_ptr, vx_rectangle_t rect, i32 src_stride, i32* hist_ptr)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!src_img_ptr || w*h <= 0 || !hist_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	
	i32* cu_hist;
	cudaMalloc(&cu_hist, 256 * sizeof(i32));
	cudaMemset(cu_hist, 0, 256 * sizeof(i32));
	cuHistogram256Kernel << <block, thread >> > (cu_hist, src_img_ptr, w, h, src_stride);
	cudaMemcpy(hist_ptr, cu_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
	cudaFree(cu_hist);
	return VX_SUCCESS;
}

vx_status cuHistogramEx(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect,
					i32 src_stride, i32 mask_stride, i32* hist_ptr, i32* border_hist_ptr)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!img_ptr || w*h <= 0 || !hist_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));

	if (mask_img_ptr)
	{
		if (border_hist_ptr)
		{
			i32* cu_hist;
			i32* cu_border_hist;
			cudaMalloc(&cu_hist, 256 * sizeof(i32));
			cudaMemset(cu_hist, 0, 256 * sizeof(i32));
			cudaMalloc(&cu_border_hist, 256 * sizeof(i32));
			cudaMemset(cu_border_hist, 0, 256 * sizeof(i32));
			cuHistogram256MaskBorderKernel<<<block, thread >>>(cu_hist, cu_border_hist,
														img_ptr, mask_img_ptr, w, h, src_stride, mask_stride);
			cudaMemcpy(hist_ptr, cu_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaMemcpy(border_hist_ptr, cu_border_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaFree(cu_hist);
			cudaFree(cu_border_hist);
		}
		else
		{
			i32* cu_hist;
			cudaMalloc(&cu_hist, 256 * sizeof(i32));
			cudaMemset(cu_hist, 0, 256 * sizeof(i32));
			cuHistogram256MaskKernel << <block, thread >> > (cu_hist, img_ptr, mask_img_ptr, w, h, src_stride, mask_stride);
			cudaMemcpy(hist_ptr, cu_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaFree(cu_hist);
		}
	}
	else
	{
		if (border_hist_ptr)
		{
			i32* cu_hist;
			i32* cu_border_hist;
			cudaMalloc(&cu_hist, 256 * sizeof(i32));
			cudaMemset(cu_hist, 0, 256 * sizeof(i32));
			cudaMalloc(&cu_border_hist, 256 * sizeof(i32));
			cudaMemset(cu_border_hist, 0, 256 * sizeof(i32));
			cuHistogram256BorderKernel << <block, thread >> > (cu_hist, cu_border_hist, img_ptr, w, h, src_stride);
			cudaMemcpy(hist_ptr, cu_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaMemcpy(border_hist_ptr, cu_border_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaFree(cu_hist);
			cudaFree(cu_border_hist);
		}
		else
		{
			i32* cu_hist;
			cudaMalloc(&cu_hist, 256 * sizeof(i32));
			cudaMemset(cu_hist, 0, 256 * sizeof(i32));
			cuHistogram256Kernel << <block, thread >> > (cu_hist, img_ptr, w, h, src_stride);
			cudaMemcpy(hist_ptr, cu_hist, 256 * sizeof(i32), cudaMemcpyDeviceToHost);
			cudaFree(cu_hist);
		}
	}
	return VX_SUCCESS;
}

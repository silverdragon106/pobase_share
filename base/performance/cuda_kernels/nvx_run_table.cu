#include "nvx_run_table.h"
#include "nvx_base.cuh"

#if defined(POR_WITH_CUDA)

// Build RunTable from run tables of each line.
//__global__ void cuCombineRunMaxtrix(i32 w, i32 h, u16* run2_matrix, i32* pixels_ptr, i32* run_count_ptr,
//								i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count)
//{
//	i32 line_run_count;
//	i32 tmp_pixels = 0, tmp_run_count = 0;
//	i32 w2 = w + 2;
//	u16* tmp_run_ptr = run2_ptr;
//	for (i32 i = 0; i < h; i++)
//	{
//		tmp_pixels += pixels_ptr[i];
//		line_run_count = run_count_ptr[i];
//		tmp_run_count += line_run_count;
//		pxy_ptr[i + 1] = tmp_run_count;
//		memcpy(tmp_run_ptr, run2_matrix + IMUL(i, w2), sizeof(u16)*line_run_count);
//		tmp_run_ptr += line_run_count;
//	}
//
//	*pixels = tmp_pixels;
//	*run_count = tmp_run_count;
//	pxy_ptr[0] = 0;
//}

__global__ void cuCombinePxyAddressKernel(i32 h, i32* pixels_ptr, i32* run_count_ptr,
								i32* pxy_ptr, i32* pixels, i32* run_count)
{
	i32 index = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index != 0)
	{
		return;
	}

	*pxy_ptr = 0; pxy_ptr++;
	i32 sum_pixels = 0, sum_run_count = 0;

	for (i32 i = 0; i < h; i++, pxy_ptr++, pixels_ptr++, run_count_ptr++)
	{
		sum_pixels += *pixels_ptr;
		sum_run_count += *run_count_ptr;
		*pxy_ptr = sum_run_count;
	}

	*pixels = sum_pixels;
	*run_count = sum_run_count;
}

__global__ void cuCombineRunTableKernel(i32 w, i32 h, u16* run2_matrix, i32* run_count_ptr,
									i32* pxy_ptr, u16* run2_ptr)
{
	i32 line = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (line < 0 || line >= h)
	{
		return;
	}
	
	i32 w2 = w + 2;
	u16* line_run2_ptr = run2_matrix + IMUL(line, w2);
	u16* dst_run2_ptr = run2_ptr + pxy_ptr[line];
	memcpy(dst_run2_ptr, line_run2_ptr, sizeof(u16)*run_count_ptr[line]);
}

/*
@ pxy_ptr		: Array with size of image height, i.e. int[height].
				  pxy_ptr[i] indicates beginning index in run2_ptr
@ run2_matrix	: Duple Array with size of image height * image width
				  This value is designed for shared memory for all threads, while
				  each thread processes single line.
*/
__global__ void cuImage2RuntableKernel(u8* img_ptr, i32 w, i32 h, i32 img_stride,
									u16* run2_matrix, i32* pixels_ptr, i32* run_count_ptr)
{
	i32 line = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (line < 0 || line >= h)
	{
		return;
	}

	i32 seg_count = 0;
	i32 tot_pixel_count = 0;
	i32 x = 0, w2 = w + 2;
	
	i32 st_index = 0;			// start index of segment with continuous white tot_pixel_count
	i32 seg_pixel_count = 0;	// pixel count of segment with continuous white tot_pixel_count
	u8* line_ptr = img_ptr + IMUL(line, img_stride);
	u16* line_run2_ptr = run2_matrix + IMUL(line, w2);

	for (x = 0; x < w; x++, line_ptr++)
	{
		if (*line_ptr > 0)
		{
			seg_pixel_count++;
			tot_pixel_count++;
			if (seg_pixel_count == 1) // if a new segment
			{
				st_index = x;
			}
		}
		else if (seg_pixel_count > 0) // if end of a segment
		{
			line_run2_ptr[0] = st_index;
			line_run2_ptr[1] = seg_pixel_count;
			line_run2_ptr += 2;
			seg_pixel_count = 0;
			seg_count++;
		}
	}
	if (seg_pixel_count > 0) // if a segment ends with white pixel
	{
		line_run2_ptr[0] = st_index;
		line_run2_ptr[1] = seg_pixel_count;
		line_run2_ptr += 2;
		seg_pixel_count = 0;
		seg_count++;
	}

	pixels_ptr[line] = tot_pixel_count;
	run_count_ptr[line] = IMUL(seg_count, 2); // Size in u16, so muliply 2.
}

vx_status cuImage2Runtable(u8* img_ptr, i32 w, i32 h, i32 img_stride, 
						i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count)
{
	if (!img_ptr || w*h <= 0 || !pxy_ptr || !run2_ptr || !pixels || !run_count)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	
	u16* run2_matrix;
	i32* pixels_ptr;
	i32* run_count_ptr;

	cudaMalloc(&run2_matrix, sizeof(u16) * h * (w + 2));
	cudaMemset(&run2_matrix, 0, sizeof(u16) * h * (w + 2));
	cudaMalloc(&pixels_ptr, sizeof(i32) * h);
	cudaMemset(&pixels_ptr, 0, sizeof(i32) * h);
	cudaMalloc(&run_count_ptr, sizeof(i32) * h);
	cudaMemset(&run_count_ptr, 0, sizeof(i32) * h);
	
	dim3 thread(CUDA_MAX_THREAD);
	dim3 block(_cuDivUp(h, CUDA_MAX_THREAD));
	cuImage2RuntableKernel<<<block, thread>>> (img_ptr, w, h, img_stride, run2_matrix, pixels_ptr, run_count_ptr);
	cuCombinePxyAddressKernel<<<1, 1>>> (h, pixels_ptr, run_count_ptr, pxy_ptr, pixels, run_count);
	cuCombineRunTableKernel<<<block, thread>>> (w, h, run2_matrix, run_count_ptr, pxy_ptr, run2_ptr);

	cudaFree(run2_matrix);
	cudaFree(pixels_ptr);
	cudaFree(run_count_ptr);
	return VX_SUCCESS;
}

// Invert
__global__ void cuImage2RuntableInvertKernel(u8* img_ptr, int w, int h, i32 img_stride,
										u16* run2_matrix, i32* pixels_ptr, i32* run_count_ptr)
{
	i32 line = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (line < 0 || line >= h)
	{
		return;
	}

	i32 x = 0, w2 = w + 2;
	i32 seg_count = 0;
	i32 tot_pixel_count = 0;

	i32 st_index = 0;			// start index of segment with continuous white tot_pixel_count
	i32 seg_pixel_count = 0;	// pixel count of segment with continuous white tot_pixel_count
	u8* line_ptr = img_ptr + IMUL(line, img_stride);
	u16* line_run2_ptr = run2_matrix + IMUL(line, w2);

	for (x = 0; x < w; x++, line_ptr++)
	{
		if (*line_ptr == 0)
		{
			seg_pixel_count++;
			tot_pixel_count++;
			if (seg_pixel_count == 1) // if a new segment
			{
				st_index = x;
			}
		}
		else if (seg_pixel_count > 0) // if end of a segment
		{
			line_run2_ptr[0] = st_index;
			line_run2_ptr[1] = seg_pixel_count;
			line_run2_ptr += 2;

			seg_count++;
			seg_pixel_count = 0;
		}
	}
	if (seg_pixel_count > 0) // if a segment ends with white pixel
	{
		line_run2_ptr[0] = st_index;
		line_run2_ptr[1] = seg_pixel_count;
		line_run2_ptr += 2;

		seg_count++;
		seg_pixel_count = 0;
	}

	pixels_ptr[line] = tot_pixel_count;
	run_count_ptr[line] = IMUL(seg_count, 2); // Size in u16, so muliply 2.
}

vx_status cuImage2RuntableInvert(u8* img_ptr, int w, int h, i32 img_stride,
							i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count)
{
	if (!img_ptr || w*h <= 0 || !pxy_ptr || !run2_ptr || !pixels || !run_count)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	u16* run2_matrix;
	i32* pixels_ptr;
	i32* run_count_ptr;

	cudaMalloc(&run2_matrix, sizeof(u16) * h * (w + 2));
	cudaMemset(&run2_matrix, 0, sizeof(u16) * h * (w + 2));
	cudaMalloc(&pixels_ptr, sizeof(i32) * h);
	cudaMemset(&pixels_ptr, 0, sizeof(i32) * h);
	cudaMalloc(&run_count_ptr, sizeof(i32) * h);
	cudaMemset(&run_count_ptr, 0, sizeof(i32) * h);

	dim3 thread(CUDA_MAX_THREAD);
	dim3 block(_cuDivUp(h, CUDA_MAX_THREAD));
	cuImage2RuntableInvertKernel << <block, thread >> > (img_ptr, w, h, img_stride, run2_matrix, pixels_ptr, run_count_ptr);
	cuCombinePxyAddressKernel << <1, 1 >> > (h, pixels_ptr, run_count_ptr, pxy_ptr, pixels, run_count);
	cuCombineRunTableKernel << <block, thread >> > (w, h, run2_matrix, run_count_ptr, pxy_ptr, run2_ptr);

	cudaFree(run2_matrix);
	cudaFree(pixels_ptr);
	cudaFree(run_count_ptr);
	return VX_SUCCESS;
}

// Mask
__global__ void cuImage2RuntableWithMaskKernel(u8* img_ptr, i32 w, i32 h, i32 img_stride,
							u8* mask_ptr, i32 mask_stride, u16* run2_matrix, i32* pixels_ptr, i32* run_count_ptr)
{
	i32 line = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (line < 0 || line >= h)
	{
		return;
	}

	i32 x = 0, w2 = w + 2;
	i32 seg_count = 0;
	i32 tot_pixel_count = 0;

	i32 st_index = 0;			// start index of segment with continuous white tot_pixel_count
	i32 seg_pixel_count = 0;	// pixel count of segment with continuous white tot_pixel_count
	u8* line_ptr = img_ptr + IMUL(line, img_stride);
	u8* mask_line_ptr = mask_ptr + IMUL(line, mask_stride);
	u16* line_run2_ptr = run2_matrix + IMUL(line, w2);

	for (x = 0; x < w; x++, line_ptr++, mask_line_ptr)
	{
		if (*line_ptr > 0 && *mask_line_ptr > 0)
		{
			seg_pixel_count++;
			tot_pixel_count++;
			if (seg_pixel_count == 1) // if a new segment
			{
				st_index = x;
			}
		}
		else if (seg_pixel_count > 0) // if end of a segment
		{
			line_run2_ptr[0] = st_index;
			line_run2_ptr[1] = seg_pixel_count;
			line_run2_ptr += 2;

			seg_count++;
			seg_pixel_count = 0;
		}
	}
	if (seg_pixel_count > 0) // if a segment ends with white pixel
	{
		line_run2_ptr[0] = st_index;
		line_run2_ptr[1] = seg_pixel_count;
		line_run2_ptr += 2;

		seg_count++;
		seg_pixel_count = 0;
	}

	pixels_ptr[line] = tot_pixel_count;
	run_count_ptr[line] = IMUL(seg_count, 2); // Size in u16, so muliply 2.
}

vx_status cuImage2RuntableWithMask(u8* img_ptr, int w, int h, i32 img_stride,
								u8* mask_ptr, i32 mask_stride,
								i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count)
{
	if (!img_ptr || w*h <= 0 || !mask_ptr || !pxy_ptr || !run2_ptr || !pixels || !run_count)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	u16* run2_matrix;
	i32* pixels_ptr;
	i32* run_count_ptr;

	cudaMalloc(&run2_matrix, sizeof(u16) * h * (w + 2));
	cudaMemset(&run2_matrix, 0, sizeof(u16) * h * (w + 2));
	cudaMalloc(&pixels_ptr, sizeof(i32) * h);
	cudaMemset(&pixels_ptr, 0, sizeof(i32) * h);
	cudaMalloc(&run_count_ptr, sizeof(i32) * h);
	cudaMemset(&run_count_ptr, 0, sizeof(i32) * h);

	dim3 thread(CUDA_MAX_THREAD);
	dim3 block(_cuDivUp(h, CUDA_MAX_THREAD));
	cuImage2RuntableWithMaskKernel << <block, thread>> > (img_ptr, w, h, img_stride,
											mask_ptr, mask_stride, run2_matrix, pixels_ptr, run_count_ptr);
	cuCombinePxyAddressKernel << <1, 1 >> > (h, pixels_ptr, run_count_ptr, pxy_ptr, pixels, run_count);
	cuCombineRunTableKernel << <block, thread >> > (w, h, run2_matrix, run_count_ptr, pxy_ptr, run2_ptr);


	cudaFree(run2_matrix);
	cudaFree(pixels_ptr);
	cudaFree(run_count_ptr);
	return VX_SUCCESS;
}

// Mask Value
__global__ void cuImage2RuntableWithMaskValKernel(u8* img_ptr, i32 w, i32 h, i32 img_stride, i32 mask_val,
												u16* run2_matrix, i32* pixels_ptr, i32* run_count_ptr)
{
	i32 line = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (line < 0 || line >= h)
	{
		return;
	}

	i32 x = 0, w2 = w + 2;
	i32 seg_count = 0;
	i32 tot_pixel_count = 0;

	i32 st_index = 0;			// start index of segment with continuous white tot_pixel_count
	i32 seg_pixel_count = 0;	// pixel count of segment with continuous white tot_pixel_count
	u8* line_ptr = img_ptr + IMUL(line, img_stride);
	u16* line_run2_ptr = run2_matrix + IMUL(line, w2);

	for (x = 0; x < w; x++, line_ptr++)
	{
		if ((*line_ptr & mask_val) > 0)
		{
			seg_pixel_count++;
			tot_pixel_count++;
			if (seg_pixel_count == 1) // if a new segment
			{
				st_index = x;
			}
		}
		else if (seg_pixel_count > 0) // if end of a segment
		{
			line_run2_ptr[0] = st_index;
			line_run2_ptr[1] = seg_pixel_count;
			line_run2_ptr += 2;

			seg_count++;
			seg_pixel_count = 0;
		}
	}
	if (seg_pixel_count > 0) // if a segment ends with white pixel
	{
		line_run2_ptr[0] = st_index;
		line_run2_ptr[1] = seg_pixel_count;
		line_run2_ptr += 2;

		seg_count++;
		seg_pixel_count = 0;
	}

	pixels_ptr[line] = tot_pixel_count;
	run_count_ptr[line] = IMUL(seg_count, 2); // Size in u16, so muliply 2.
}

vx_status cuImage2RuntableWithMaskVal(u8* img_ptr, i32 w, i32 h, i32 img_stride, i32 mask_val,
									i32* pxy_ptr, u16* run2_ptr, i32* pixels, i32* run_count)
{
	if (!img_ptr || w*h <= 0 || !pxy_ptr || !run2_ptr || !pixels || !run_count)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	u16* run2_matrix;
	i32* pixels_ptr;
	i32* run_count_ptr;

	cudaMalloc(&run2_matrix, sizeof(u16) * h * (w + 2));
	cudaMemset(&run2_matrix, 0, sizeof(u16) * h * (w + 2));
	cudaMalloc(&pixels_ptr, sizeof(i32) * h);
	cudaMemset(&pixels_ptr, 0, sizeof(i32) * h);
	cudaMalloc(&run_count_ptr, sizeof(i32) * h);
	cudaMemset(&run_count_ptr, 0, sizeof(i32) * h);

	dim3 thread(CUDA_MAX_THREAD);
	dim3 block(_cuDivUp(h, CUDA_MAX_THREAD));
	cuImage2RuntableWithMaskValKernel << <block, thread>> > (img_ptr, w, h, img_stride,
												mask_val, run2_matrix, pixels_ptr, run_count_ptr);
	cuCombinePxyAddressKernel << <1, 1 >> > (h, pixels_ptr, run_count_ptr, pxy_ptr, pixels, run_count);
	cuCombineRunTableKernel << <block, thread >> > (w, h, run2_matrix, run_count_ptr, pxy_ptr, run2_ptr);

	cudaFree(run2_matrix);
	cudaFree(pixels_ptr);
	cudaFree(run_count_ptr);
	return VX_SUCCESS;
}

// Runtable to Image Functions
__global__ void cuRuntable2ImageKernel(u8* dst_img_ptr, i32 w, i32 h, i32 img_stride,
									i32* pxy_ptr, u16* run2_ptr, i32 val)
{
	i32 line = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (line < h)
	{
		i32 pos = IMUL(IMUL(blockIdx.x, blockDim.x) + threadIdx.x, 2);
		i32 st_pos = pxy_ptr[line] + pos;
		i32 ed_pos = pxy_ptr[line + 1];
		if (st_pos < ed_pos)
		{
			u8* line_ptr = dst_img_ptr + IMUL(line, img_stride);
			memset(line_ptr + run2_ptr[st_pos], val, run2_ptr[st_pos + 1]);
		}
	}
}

vx_status cuRuntable2Image(u8* dst_img_ptr, i32 w, i32 h, i32 img_stride,
						i32* pxy_ptr, u16* run2_ptr, i32 val)
{
	if (!dst_img_ptr || w*h <= 0 || !pxy_ptr || !run2_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	cudaMemset(dst_img_ptr, 0, img_stride*h);

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp((w + 1) / 2, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuRuntable2ImageKernel << <block, thread>> > (dst_img_ptr, w, h, img_stride, pxy_ptr, run2_ptr, val);
	return VX_SUCCESS;
}
#endif

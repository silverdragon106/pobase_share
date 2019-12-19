#include "nvx_draw.h"
#include "nvx_base.cuh"

__global__ void cuDrawCircleKernel(u8* dst_img_ptr, i32 w, i32 h, i32 dst_stride,
								f32 cx, f32 cy, f32 r2, u8 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		f32 dx = tx - cx;
		f32 dy = ty - cy;
		u8* pixel_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		if (dx*dx + dy*dy <= r2)
		{
			*pixel_ptr = value;
		}
		else
		{
			*pixel_ptr = 0;
		}
	}
}

vx_status cuDrawCircle(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
					f32 cx, f32 cy, f32 r, u8 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!dst_img_ptr || w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuDrawCircleKernel << <block, thread>> > (dst_img_ptr, w, h, dst_stride, cx, cy, r*r, value);
	return VX_SUCCESS;
}

__global__ void cuDrawRingKernel(u8* dst_img_ptr, i32 w, i32 h, i32 dst_stride,
							f32 cx, f32 cy, f32 inner_r2, f32 outer_r2, f32 st_angle, f32 angle_len, u8 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		u8* pixel_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;

		f32 dx = tx - cx;
		f32 dy = ty - cy;
		if ((dx*dx + dy*dy < inner_r2) || (dx*dx + dy*dy > outer_r2))
		{
			*pixel_ptr = 0;
			return;
		}

		f32 angle = atan2f(dy, dx);
		angle = angle < 0 ? CUDA_PO_PI2 + angle : angle;
		angle = angle - st_angle;
		angle = angle < 0 ? CUDA_PO_PI2 + angle : angle;
		if (angle < angle_len)
		{
			*pixel_ptr = value;
		}
		else
		{
			*pixel_ptr = 0;
		}
	}
}

vx_status cuDrawRing(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
					f32 cx, f32 cy, f32 r1, f32 r2, f32 st_angle, f32 ed_angle, u8 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!dst_img_ptr || w*h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	f32 inner_r = min(r1, r2);
	f32 outer_r = max(r1, r2);
	
	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuDrawRingKernel << <block, thread >> > (dst_img_ptr, w, h, dst_stride, cx, cy,
			inner_r*inner_r, outer_r*outer_r, st_angle, ed_angle - st_angle, value);
	return VX_SUCCESS;
}

__global__ void cuDrawEllipseKernel(u8* dst_img_ptr, i32 w, i32 h, i32 dst_stride,
									f32 cx, f32 cy, f32 r12, f32 r22, f32 cov_r2, u8 value)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx < w && ty < h)
	{
		f32 dx = tx - cx;
		f32 dy = ty - cy;
		u8* pixel_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		if (dx*dx*r22 + dy*dy*r12 <= cov_r2)
		{
			*pixel_ptr = value;
		}
		else
		{
			*pixel_ptr = 0;
		}
	}
}

vx_status cuDrawEllipse(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
						f32 cx, f32 cy, f32 r1, f32 r2, u8 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!dst_img_ptr || w * h <= 0)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	f32 r12 = r1 * r1;
	f32 r22 = r2 * r2;
	f32 cov_r2 = r12 * r22;
	dim3 thread(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	dim3 block(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuDrawEllipseKernel << <block, thread>> > (dst_img_ptr, w, h, dst_stride, cx, cy, r12, r22, cov_r2, value);
	return VX_SUCCESS;
}

__global__ void cuDrawPolygonPreprocessKernel(cuVector2df* poly_ptr, i32 poly_pt_count, f32* cu_poly_mul_ptr, f32* cu_poly_con_ptr)
{
	//see more detail... http://alienryderflex.com/polygon/
	i32 i, ni;
	f32 dx, dy, dtmp, x1, x2, y1, y2;
	i = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i < poly_pt_count)
	{
		ni = (i + 1) % poly_pt_count;
		x1 = poly_ptr[i].x;
		y1 = poly_ptr[i].y;
		x2 = poly_ptr[ni].x;
		y2 = poly_ptr[ni].y;

		if (y1 == y2)
		{
			cu_poly_mul_ptr[i] = 0;
			cu_poly_con_ptr[i] = x1;
		}
		else
		{
			dx = x2 - x1;
			dy = y2 - y1;
			dtmp = dx / dy;
			cu_poly_mul_ptr[i] = dtmp;
			cu_poly_con_ptr[i] = x1 - y1*dtmp;
		}
	}
}

__global__ void cuDrawPolygonKernel(u8* dst_img_ptr, i32 w, i32 h, i32 dst_stride,
								cuVector2df* poly_ptr, i32 poly_pt_count, u8 value,
								f32* cu_poly_mul_ptr, f32* cu_poly_con_ptr)
{
	i32 tx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	i32 ty = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	i32 i, ni;

	if (tx < w && ty < h)
	{
		f32 y1, y2;
		bool is_in_polygon = false;
		u8* scan_img_ptr = dst_img_ptr + IMUL(ty, dst_stride) + tx;
		for (i = 0; i < poly_pt_count; i++)
		{
			ni = (i + 1) % poly_pt_count;
			y1 = poly_ptr[i].y;
			y2 = poly_ptr[ni].y;
			if ((y1 < ty && y2 >= ty) || (y2 < ty && y1 >= ty))
			{
				is_in_polygon ^= (ty*cu_poly_mul_ptr[i] + cu_poly_con_ptr[i] < tx);
			}
		}

		//check point inside polygon
		if (is_in_polygon)
		{
			*scan_img_ptr = value;
		}
		else
		{
			*scan_img_ptr = 0;
		}
	}
}

vx_status cuDrawPolygon(u8* dst_img_ptr, vx_rectangle_t rect, i32 dst_stride,
					cuVector2df* poly_ptr, i32 poly_pt_count, u8 value)
{
	i32 w = rect.end_x - rect.start_x;
	i32 h = rect.end_y - rect.start_y;
	if (!dst_img_ptr || w*h <= 0 || !poly_ptr || poly_pt_count < 3)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	f32* cu_poly_mul_ptr;
	f32* cu_poly_con_ptr;
	cuVector2df* cu_poly_ptr;

	cudaMalloc(&cu_poly_mul_ptr, poly_pt_count * sizeof(f32));
	cudaMalloc(&cu_poly_con_ptr, poly_pt_count * sizeof(f32));
	cudaMalloc(&cu_poly_ptr, poly_pt_count * sizeof(cuVector2df));
	cudaMemcpy(cu_poly_ptr, poly_ptr, poly_pt_count * sizeof(cuVector2df), cudaMemcpyHostToDevice);
	
	dim3 thread(CUDA_MAX_THREAD);
	dim3 block(_cuDivUp(poly_pt_count, CUDA_MAX_THREAD));
	cuDrawPolygonPreprocessKernel << <block, thread>> > (cu_poly_ptr, poly_pt_count, cu_poly_mul_ptr, cu_poly_con_ptr);

	thread = dim3(CUDA_THREAD_MAXW, CUDA_THREAD_MAXH);
	block = dim3(_cuDivUp(w, CUDA_THREAD_MAXW), _cuDivUp(h, CUDA_THREAD_MAXH));
	cuDrawPolygonKernel << <block, thread>> > (dst_img_ptr, w, h, dst_stride,
						cu_poly_ptr, poly_pt_count, value, cu_poly_mul_ptr, cu_poly_con_ptr);

	cudaFree(cu_poly_mul_ptr);
	cudaFree(cu_poly_con_ptr);
	cudaFree(cu_poly_ptr);
	return VX_SUCCESS;
}

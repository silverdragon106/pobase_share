#include "vx_api_filter.h"
#include "proc/image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_OVX)

vx_status _vxGaussian2d_u16(u16* src_img_ptr, vx_rectangle_t rect, i32 src_stride,
					u16* dst_img_ptr, i32 dst_stride, i32 kernel_size)
{
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);
	if (!src_img_ptr || !dst_img_ptr || w * h <= 0 || kernel_size < 3)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	cv::Mat cv_src_img(h, w, CV_16UC1, src_img_ptr, sizeof(u16)*src_stride);
	cv::Mat cv_dst_img(h, w, CV_16UC1, dst_img_ptr, sizeof(u16)*dst_stride);
	cv::GaussianBlur(cv_src_img, cv_dst_img, cv::Size(kernel_size, kernel_size), 0.5f);
	return VX_SUCCESS;
}
#endif
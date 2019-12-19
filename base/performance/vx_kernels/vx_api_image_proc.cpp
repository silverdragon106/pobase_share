#include "vx_api_image_proc.h"
#include "proc/image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_OVX)

vx_status _vxAbs_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride, i16* dst_img_ptr, i32 dst_stride)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	i16* scan_img_ptr;
	i16* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = std::abs(scan_img_ptr[x]);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxAddConst_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
						u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_min(scan_img_ptr[x] + value, 0xFF);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxAddConst_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
						i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	i16* scan_img_ptr;
	i16* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_max(po::_min(scan_img_ptr[x] + value, 0x7FFF), -0x7FFF);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxMulConsti32_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
						u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_min(scan_img_ptr[x]*value, 0xFF);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxMulConstf32_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
						 u8* dst_img_ptr, i32 dst_stride, f32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_min(scan_img_ptr[x] * value, 0xFF);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxMulConsti32_i16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
						i16* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	i16* scan_img_ptr;
	i16* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_max(po::_min(scan_img_ptr[x] * value, 0x7FFF), -0x7FFF);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxMin_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	cv::Mat cv_img1(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::Mat cv_img2(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::Mat cv_min_img(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::min(cv_img1, cv_img2, cv_min_img);
	return VX_SUCCESS;
}

vx_status _vxMax_u8u8u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	cv::Mat cv_img1(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::Mat cv_img2(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::Mat cv_max_img(h, w, CV_8UC1, img1_ptr, img1_stride);
	cv::max(cv_img1, cv_img2, cv_max_img);
	return VX_SUCCESS;
}

vx_status _vxMul_u8u8u16(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride,
					u8* img2_ptr, i32 img2_stride, u16* dst_img_ptr, i32 dst_stride)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img1_ptr;
	u8* scan_img2_ptr;
	u16* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img1_ptr = img1_ptr + y * img1_stride;
		scan_img2_ptr = img2_ptr + y * img2_stride;
		scan_dst_ptr = dst_img_ptr + y * dst_stride;
		for (x = 0; x < w; x++)
		{
			*scan_dst_ptr = (*scan_img1_ptr) * (*scan_img2_ptr);
			scan_img1_ptr++; scan_img2_ptr++; scan_dst_ptr++;
		}
	}
	return VX_SUCCESS;
}

vx_status _vxCut_u16(i16* img_ptr, vx_rectangle_t rect, i32 src_stride,
				u8* mask_img_ptr, i32 mask_stride, i32 threshold,
				i16* dst_img_ptr, i32 dst_stride, i32& valid_pixels)
{
	i32 x, y, value;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	i16* scan_img_ptr;
	i16* scan_dst_ptr;
	u8* scan_mask_ptr;

	valid_pixels = 0;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		scan_mask_ptr = mask_img_ptr+ y*mask_stride;
		for (x = 0; x < w; x++)
		{
			value = scan_img_ptr[x];
			if (value >= threshold && scan_mask_ptr[x] != 0)
			{
				valid_pixels++;
				scan_dst_ptr[x] = value;
			}
			else
			{
				scan_dst_ptr[x] = 0;
			}
		}
	}
	return VX_SUCCESS;
}

vx_status _vxClipMin_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_max(scan_img_ptr[x], value);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxClipMax_u8(u8* img_ptr, vx_rectangle_t rect, i32 src_stride,
					u8* dst_img_ptr, i32 dst_stride, i32 value)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_min(scan_img_ptr[x], value);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxSubtractEx_u8(u8* img1_ptr, vx_rectangle_t rect, i32 img1_stride, f32 alpha,
					u8* img2_ptr, i32 img2_stride, f32 beta, u8* dst_img_ptr, i32 dst_stride)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img1_ptr;
	u8* scan_img2_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img1_ptr = img1_ptr + y*img1_stride;
		scan_img2_ptr = img2_ptr + y*img2_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = po::_max(scan_img1_ptr[x] * alpha - scan_img2_ptr[x] * beta, 0);
		}
	}
	return VX_SUCCESS;
}

vx_status _vxMask_u8_u8(u8* img_ptr, vx_rectangle_t rect, i32 img_stride,
				u8* mask_ptr, i32 mask_stride, u8* dst_img_ptr, i32 dst_stride)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);
	memset(dst_img_ptr, 0, h*dst_stride);

	u8* scan_img_ptr;
	u8* scan_mask_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y * img_stride;
		scan_mask_ptr = mask_ptr + y * mask_stride;
		scan_dst_ptr = dst_img_ptr + y * dst_stride;
		for (x = 0; x < w; x++)
		{
			if (scan_mask_ptr[x])
			{
				scan_dst_ptr[x] = scan_img_ptr[x];
			}
		}
	}
	return VX_SUCCESS;
}

vx_status _vxPalette_u16(u16* img_ptr, vx_rectangle_t rect, i32 img_stride,
					u8* palette_ptr, i32 palette_size, u8* dst_img_ptr, i32 dst_stride)
{
	i32 x, y;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u16* scan_src_ptr;
	u8* scan_dst_ptr;
	for (y = 0; y < h; y++)
	{
		scan_src_ptr = img_ptr + y*img_stride;
		scan_dst_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			scan_dst_ptr[x] = palette_ptr[scan_src_ptr[x]];
		}
	}
	return VX_SUCCESS;
}

vx_status __vxHistogram(u8* img_ptr, vx_rectangle_t rect, i32 src_stride, i32* hist_ptr)
{
	i32 x, y, img_pixel;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		for (x = 0; x < w; x++)
		{
			img_pixel = *scan_img_ptr;
			hist_ptr[img_pixel]++;
			scan_img_ptr++;
		}
	}
	return VX_SUCCESS;
}

vx_status __vxHistogram(u8* img_ptr, vx_rectangle_t rect, i32 src_stride, i32* hist_ptr, i32* border_hist_ptr)
{
	i32 x, y, img_pixel;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		for (x = 0; x < w; x++)
		{
			img_pixel = *scan_img_ptr;

			if (img_pixel == kPOEdgePixel)
			{
				border_hist_ptr[img_pixel]++;
			}
			hist_ptr[img_pixel]++;
			scan_img_ptr++;
		}
	}
	return VX_SUCCESS;
}

vx_status __vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect, i32 src_stride,
					i32 mask_stride, i32* hist_ptr)
{
	i32 x, y, img_pixel, mask_pixel;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_mask_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_mask_ptr = mask_img_ptr + y*mask_stride;
		for (x = 0; x < w; x++)
		{
			img_pixel = *scan_img_ptr;
			mask_pixel = *scan_mask_ptr;

			if (mask_pixel > 0)
			{
				hist_ptr[img_pixel]++;
			}
			scan_img_ptr++;
			scan_mask_ptr++;
		}
	}
	return VX_SUCCESS;
}

vx_status __vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect, i32 src_stride, i32 mask_stride,
					i32* hist_ptr, i32* border_hist_ptr)
{
	i32 x, y, img_pixel, mask_pixel;
	i32 w = RECT_WIDTH(rect);
	i32 h = RECT_HEIGHT(rect);

	u8* scan_img_ptr;
	u8* scan_mask_ptr;
	for (y = 0; y < h; y++)
	{
		scan_img_ptr = img_ptr + y*src_stride;
		scan_mask_ptr = mask_img_ptr + y*mask_stride;
		for (x = 0; x < w; x++)
		{
			img_pixel = *scan_img_ptr;
			mask_pixel = *scan_mask_ptr;

			if (mask_pixel > 0)
			{
				hist_ptr[img_pixel]++;
				if (mask_pixel == kPOEdgePixel)
				{
					border_hist_ptr[img_pixel]++;
				}
			}
			scan_img_ptr++;
			scan_mask_ptr++;
		}
	}
	return VX_SUCCESS;
}

vx_status _vxHistogram(u8* img_ptr, u8* mask_img_ptr, vx_rectangle_t rect, i32 src_stride, i32 mask_stride,
					i32* hist_ptr, i32* border_hist_ptr)
{
	if (!img_ptr || !hist_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	if (mask_img_ptr)
	{
		if (border_hist_ptr)
		{
			return __vxHistogram(img_ptr, mask_img_ptr, rect, src_stride, mask_stride, hist_ptr, border_hist_ptr);
		}
		else
		{
			return __vxHistogram(img_ptr, mask_img_ptr, rect, src_stride, mask_stride, hist_ptr);
		}
	}
	else
	{
		if (border_hist_ptr)
		{
			return __vxHistogram(img_ptr, rect, src_stride, hist_ptr, border_hist_ptr);
		}
		else
		{
			return __vxHistogram(img_ptr, rect, src_stride, hist_ptr);
		}
	}
	return VX_FAILURE;
}

i32 _vxCalcThreshold(i32* hist_ptr, i32 hist_min, i32 hist_max)
{
	return CImageProc::calcThreshold(hist_ptr, hist_min, hist_max);
}

i32	_vxConnectComponents(u8* img_ptr, vx_rectangle_t src_rect, i32 src_stride,
					u8* mask_img_ptr, vx_rectangle_t mask_rect, i32 mask_stride,
					u16* label_img_ptr, vx_rectangle_t dst_rect, i32 dst_stride)
{
	if (!img_ptr || !label_img_ptr)
	{
		return 0;
	}

	i32 cc_count = 0;
	i32 w = RECT_WIDTH(src_rect);
	i32 h = RECT_HEIGHT(src_rect);
	cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr, src_stride);
	cv::Mat cv_label_img(h, w, CV_16UC1, label_img_ptr, sizeof(u16)*dst_stride);

	if (mask_img_ptr)
	{
		cv::Mat cv_dst_img;
		cv::Mat cv_mask_img(mask_rect.end_y, mask_rect.end_x, CV_8UC1, mask_img_ptr, mask_stride);
		cv_src_img.copyTo(cv_dst_img, cv_mask_img);
		cc_count = cv::connectedComponents(cv_dst_img, cv_label_img, 8, CV_16U);
	}
	else
	{
		cc_count = cv::connectedComponents(cv_src_img, cv_label_img, 8, CV_16U);
	}
	return cc_count;
}

bool _vxIsInvertedBackground(i32 th, i32* hist_ptr, i32* border_hist_ptr)
{
	return CImageProc::isInvertedBackground(th, hist_ptr, border_hist_ptr);
}
#endif
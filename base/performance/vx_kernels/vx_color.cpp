#include "vx_color.h"
#include "vx_kernel_types.h"

#if defined(POR_WITH_OVX)
#include "vx_api_image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_CUDA)
#include "performance/cuda_kernels/nvx_color.h"
#endif

//////////////////////////////////////////////////////////////////////////
inline void rgb2hsv(i32 r, i32 g, i32 b, i32& h, i32& s, i32& v)
{
	i32 rgb_min = po::_min(po::_min(r, g), b);
	i32 rgb_max = po::_max(po::_max(r, g), b);

	v = rgb_max;
	if (v == 0)
	{
		h = s = 0;
		return;
	}

	s = 255 * (rgb_max - rgb_min) / v;
	if (s == 0)
	{
		h = 0;
		return;
	}

	if (rgb_max == r)
	{
		h = 0 + 42.5f * (g - b) / (rgb_max - rgb_min);
	}
	else if (rgb_max == g)
	{
		h = 85 + 42.5f * (b - r) / (rgb_max - rgb_min);
	}
	else
	{
		h = 170 + 42.5f * (r - g) / (rgb_max - rgb_min);
	}
	if (h < 0)
	{
		h += 255;
	}
}

void convertRGBToHSVImage(u8* src_img_ptr, vx_rectangle_t src_rect, i32 src_stride,
				u8* dst_img_ptr, i32 dst_stride)
{
	i32 x, y, r, g, b, h, s, v;
	u8* scn_img_ptr;
	u8* new_img_ptr;
	i32 width = RECT_WIDTH(src_rect);
	i32 height = RECT_HEIGHT(src_rect);

	for (y = 0; y < height; y++)
	{
		scn_img_ptr = src_img_ptr + y*src_stride;
		new_img_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < width; x++)
		{
			r = scn_img_ptr[0];
			g = scn_img_ptr[1];
			b = scn_img_ptr[2];
			rgb2hsv(r, g, b, h, s, v);
			new_img_ptr[0] = h;
			new_img_ptr[1] = s;
			new_img_ptr[2] = v;

			scn_img_ptr += kPORGBChannels;
			new_img_ptr += kPORGBChannels;
		}
	}
}

vx_status convertToHSVKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kConvertToHSVKernelInImage];
	vx_image dst_image = (vx_image)parameters[kConvertToHSVKernelOutImage];

	if (count == kConvertToHSVKernelParamCount)
	{
		status = VX_SUCCESS;
		if (OvxHelper::getFormat(src_image) != VX_DF_IMAGE_RGB ||
			OvxHelper::getFormat(dst_image) != VX_DF_IMAGE_RGB)
		{
			return VX_ERROR_INVALID_PARAMETERS;
		}

		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_IMAGE(dst_image, VX_WRITE_ONLY);
		if (status == VX_SUCCESS)
		{
			convertRGBToHSVImage((u8*)data_src_image, rect_src_image, addr_src_image.stride_y,
							(u8*)data_dst_image, addr_dst_image.stride_y);
		}
		UNMAP_IMAGE(src_image);
		UNMAP_IMAGE(dst_image);
	}
	return status;
}

vx_status convertToHSVKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertToHSVKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertToHSVKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kConvertToHSVKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kConvertToHSVKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kConvertToHSVKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kConvertToHSVKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
void convertToAvgChannel(u8* src_img_ptr, i32 src_channel, vx_rectangle_t src_rect, i32 src_stride,
						u8* dst_img_ptr, i32 dst_stride)
{
	i32 x, y, k, sum;
	if (!CPOBase::isPositive(src_channel))
	{
		return;
	}

	u8* scn_img_ptr;
	u8* new_img_ptr;
	i32 w = RECT_WIDTH(src_rect);
	i32 h = RECT_HEIGHT(src_rect);
	for (y = 0; y < h; y++)
	{
		scn_img_ptr = src_img_ptr + y*src_stride;
		new_img_ptr = dst_img_ptr + y*dst_stride;
		for (x = 0; x < w; x++)
		{
			sum = 0;
			for (k = 0; k < src_channel; k++)
			{
				sum += *scn_img_ptr; scn_img_ptr++;
			}
			new_img_ptr[x] = sum / k;
		}
	}
}

vx_status convertToAvgKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kConvertToHSVKernelInImage];
	vx_image dst_image = (vx_image)parameters[kConvertToHSVKernelOutImage];

	if (count == kConvertToHSVKernelParamCount)
	{
		status = VX_SUCCESS;
		i32 channels = kPOAnyChannels;
		switch (OvxHelper::getFormat(src_image))
		{
			case VX_DF_IMAGE_RGB:
			case VX_DF_IMAGE_YUV4:
			{
				channels = kPORGBChannels; break;
			}
			default:
			{
				return VX_ERROR_INVALID_PARAMETERS;
			}
		}

		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_IMAGE(dst_image, VX_WRITE_ONLY);
		if (status == VX_SUCCESS)
		{
			convertToAvgChannel((u8*)data_src_image, channels, rect_src_image, addr_src_image.stride_y,
							(u8*)data_dst_image, addr_dst_image.stride_y);
		}
		UNMAP_IMAGE(src_image);
		UNMAP_IMAGE(dst_image);
	}
	return status;
}

vx_status convertToAvgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertToAvgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertToAvgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kConvertToHSVKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kConvertToHSVKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kConvertToHSVKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kConvertToHSVKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kConvertToHSVKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status paletteImgKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kPaletteImageKernelInImage];
	vx_array in_palette = (vx_array)parameters[kPaletteImageKernelInPalette];
	vx_image dst_image = (vx_image)parameters[kPaletteImageKernelOutImage];

	if (count == kPaletteImageKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(src_image, VX_READ_ONLY);
		NVX_MAP_ARRAY(in_palette, VX_READ_ONLY);
		NVX_MAP_IMAGE(dst_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			switch (addr_src_image.stride_x)
			{
				case 2:
				{
					status |= cuPalette_u16((u16*)data_src_image, rect_src_image, SCAN_WIDTH(src_image),
										(u8*)data_in_palette, (i32)num_in_palette,
										(u8*)data_dst_image, SCAN_WIDTH(dst_image));
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(src_image);
		NVX_UNMAP_ARRAY(in_palette);
		NVX_UNMAP_IMAGE(dst_image);
#else
		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_ARRAY(in_palette, VX_READ_ONLY);
		MAP_IMAGE(dst_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			switch (addr_src_image.stride_x)
			{
				case 2:
				{
					status |= _vxPalette_u16((u16*)data_src_image, rect_src_image, SCAN_WIDTH(src_image),
										(u8*)data_in_palette, (i32)num_in_palette,
										(u8*)data_dst_image, SCAN_WIDTH(dst_image));
					break;
				}
			}
		}
		UNMAP_IMAGE(src_image);
		UNMAP_ARRAY(in_palette);
		UNMAP_IMAGE(dst_image);
#endif
		status |= vxSetImageValidRectangle(dst_image, &rect_src_image);
	}
	return status;
}

vx_status paletteImgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status paletteImgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status paletteImgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kPaletteImageKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kPaletteImageKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kPaletteImageKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kPaletteImageKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kPaletteImageKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kPaletteImageKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kPaletteImageKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}
#endif
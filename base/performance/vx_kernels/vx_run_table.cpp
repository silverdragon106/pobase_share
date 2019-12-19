#include "vx_run_table.h"
#include "vx_kernel_types.h"

#if defined(POR_WITH_OVX)
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_CUDA)
#include "performance/cuda_kernels/nvx_run_table.h"
#endif

vx_status _vxImage2RunTableInvert(u8* img_ptr, i32 w, i32 h, i32 stride_w,
						i32* pxy_ptr, u16* run2_ptr, i32& pixels, i32& run_count)
{
	i32 x, y;
	u16 run_pixels = 0, run_st_pos = 0;
	u8* tmp_img_ptr;
	u16* tmp_run2_ptr;

	for (y = 0; y < h; y++)
	{
		tmp_img_ptr = img_ptr + y*stride_w;
		for (x = 0; x < w; x++, tmp_img_ptr++)
		{
			if (*tmp_img_ptr == 0)
			{
				pixels++; run_pixels++;
				if (run_pixels == 1)
				{
					run_st_pos = x;
				}
			}
			else if (run_pixels > 0)
			{
				tmp_run2_ptr = run2_ptr + run_count;
				tmp_run2_ptr[0] = run_st_pos;
				tmp_run2_ptr[1] = run_pixels;
				run_count += 2;
				run_pixels = 0;
			}
		}
		if (run_pixels > 0)
		{
			tmp_run2_ptr = run2_ptr + run_count;
			tmp_run2_ptr[0] = run_st_pos;
			tmp_run2_ptr[1] = run_pixels;
			run_count += 2;
			run_pixels = 0;
		}
		pxy_ptr[y + 1] = run_count;
	}
	return VX_SUCCESS;
}

vx_status _vxImage2RunTableWithMask(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h,
						i32 stride_w, i32 mask_stride_w, i32* pxy_ptr, u16* run2_ptr, i32& pixels, i32& run_count)
{
	i32 x, y;
	u16 run_pixels = 0, run_st_pos = 0;
	u8* tmp_img_ptr;
	u8* tmp_mask_img_ptr;
	u16* tmp_run2_ptr;

	for (y = 0; y < h; y++)
	{
		tmp_img_ptr = img_ptr + y*stride_w;
		tmp_mask_img_ptr = mask_img_ptr + y*mask_stride_w;
		for (x = 0; x < w; x++, tmp_img_ptr++, tmp_mask_img_ptr++)
		{
			if (*tmp_img_ptr > 0 && *tmp_mask_img_ptr > 0)
			{
				pixels++; run_pixels++;
				if (run_pixels == 1)
				{
					run_st_pos = x;
				}
			}
			else if (run_pixels > 0)
			{
				tmp_run2_ptr = run2_ptr + run_count;
				tmp_run2_ptr[0] = run_st_pos;
				tmp_run2_ptr[1] = run_pixels;
				run_count += 2;
				run_pixels = 0;
			}
		}
		if (run_pixels > 0)
		{
			tmp_run2_ptr = run2_ptr + run_count;
			tmp_run2_ptr[0] = run_st_pos;
			tmp_run2_ptr[1] = run_pixels;
			run_count += 2;
			run_pixels = 0;
		}
		pxy_ptr[y + 1] = run_count;
	}
	return VX_SUCCESS;
}

vx_status _vxImage2RunTableWithMaskVal(u8* img_ptr, i32 w, i32 h, i32 stride_w, u8 mask_val, 
						i32* pxy_ptr, u16* run2_ptr, i32& pixels, i32& run_count)
{
	i32 x, y;
	u16 run_pixels = 0, run_st_pos = 0;
	u8* tmp_img_ptr;
	u16* tmp_run2_ptr;

	for (y = 0; y < h; y++)
	{
		tmp_img_ptr = img_ptr + y*stride_w;
		for (x = 0; x < w; x++, tmp_img_ptr++)
		{
			if ((*tmp_img_ptr & mask_val) > 0)
			{
				pixels++; run_pixels++;
				if (run_pixels == 1)
				{
					run_st_pos = x;
				}
			}
			else if (run_pixels > 0)
			{
				tmp_run2_ptr = run2_ptr + run_count;
				tmp_run2_ptr[0] = run_st_pos;
				tmp_run2_ptr[1] = run_pixels;
				run_count += 2;
				run_pixels = 0;
			}
		}
		if (run_pixels > 0)
		{
			tmp_run2_ptr = run2_ptr + run_count;
			tmp_run2_ptr[0] = run_st_pos;
			tmp_run2_ptr[1] = run_pixels;
			run_count += 2;
			run_pixels = 0;
		}
		pxy_ptr[y + 1] = run_count;
	}
	return VX_SUCCESS;
}

vx_status _vxImage2RunTable(u8* img_ptr, i32 w, i32 h, i32 stride_w,
						i32* pxy_ptr, u16* run2_ptr, i32& pixels, i32& run_count)
{
	i32 x, y;
	u16 run_pixels = 0, run_st_pos = 0;
	u8* tmp_img_ptr;
	u16* tmp_run2_ptr;

	for (y = 0; y < h; y++)
	{
		tmp_img_ptr = img_ptr + y*stride_w;
		for (x = 0; x < w; x++, tmp_img_ptr++)
		{
			if (*tmp_img_ptr > 0)
			{
				pixels++; run_pixels++;
				if (run_pixels == 1)
				{
					run_st_pos = x;
				}
			}
			else if (run_pixels> 0)
			{
				tmp_run2_ptr = run2_ptr + run_count;
				tmp_run2_ptr[0] = run_st_pos;
				tmp_run2_ptr[1] = run_pixels;
				run_count += 2;
				run_pixels = 0;
			}
		}
		if (run_pixels > 0)
		{
			tmp_run2_ptr = run2_ptr + run_count;
			tmp_run2_ptr[0] = run_st_pos;
			tmp_run2_ptr[1] = run_pixels;
			run_count += 2;
			run_pixels = 0;
		}
		pxy_ptr[y + 1] = run_count;
	}
	return VX_SUCCESS;
}

vx_status convertImg2RunTableKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kConvertImg2RunTableKernelInImage];
	vx_image src_mask_image = (vx_image)parameters[kConvertImg2RunTableKernelInMaskImage];
	vx_array in_param = (vx_array)parameters[kConvertImg2RunTableKernelParam];
	vx_array out_array = (vx_array)parameters[kConvertImg2RunTableKernelOutArray];

	if (count == kConvertImg2RunTableKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_ARRAY(in_param, VX_READ_ONLY);

		if (status == VX_SUCCESS)
		{
			i32 w = RECT_WIDTH(rect_src_image);
			i32 h = RECT_HEIGHT(rect_src_image);
			i32 stride_w = SCAN_WIDTH(src_image);
			u8* img_ptr = (u8*)data_src_image;
			vxParamConvertImg2RunTable* param_ptr = (vxParamConvertImg2RunTable*)data_in_param;

			status |= OvxHelper::makeFullArray(out_array);
			NVX_MAP_ARRAY(out_array, VX_WRITE_ONLY);

			i32* pixels = (i32*)data_out_array;
			i32* run_count = (i32*)pixels + 1;
			i32* pxy_ptr = (i32*)run_count + 1;
			u16* run2_ptr = (u16*)((i32*)pxy_ptr + (param_ptr->height + 1));
		
			if (param_ptr->flag == kRunTableFlagNone)
			{
				status |= cuImage2Runtable(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, pixels, run_count);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagInvert))
			{
				status |= cuImage2RuntableInvert(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, pixels, run_count);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagMaskImg))
			{
				NVX_MAP_IMAGE(src_mask_image, VX_READ_ONLY);
				status |= cuImage2RuntableWithMask(img_ptr, w, h, stride_w,
								(u8*)data_src_mask_image, SCAN_WIDTH(src_mask_image),
								pxy_ptr, run2_ptr, pixels, run_count);
				NVX_UNMAP_IMAGE(src_mask_image);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagMaskVal))
			{
				status |= cuImage2RuntableWithMaskVal(img_ptr, w, h, stride_w, param_ptr->mask_val, 
								pxy_ptr, run2_ptr, pixels, run_count);
			}
			NVX_UNMAP_ARRAY(out_array);
		}
		NVX_UNMAP_IMAGE(src_image);
		UNMAP_ARRAY(in_param);
#else
		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_ARRAY(in_param, VX_READ_ONLY);

		if (status == VX_SUCCESS)
		{
			i32 w = RECT_WIDTH(rect_src_image);
			i32 h = RECT_HEIGHT(rect_src_image);
			i32 stride_w = SCAN_WIDTH(src_image);
			u8* img_ptr = (u8*)data_src_image;
			vxParamConvertImg2RunTable* param_ptr = (vxParamConvertImg2RunTable*)data_in_param;

			i32* pxy_ptr = po_new i32[param_ptr->height + 1];
			u16* run2_ptr = po_new u16[(param_ptr->width + 2) * param_ptr->height];
			i32 pixels = 0, run_count = 0;
			pxy_ptr[0] = 0;

			if (param_ptr->flag == kRunTableFlagNone)
			{
				status |= _vxImage2RunTable(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, pixels, run_count);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagInvert))
			{
				status |= _vxImage2RunTableInvert(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, pixels, run_count);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagMaskImg))
			{
				MAP_IMAGE(src_mask_image, VX_READ_ONLY);
				status |= _vxImage2RunTableWithMask(img_ptr, (u8*)data_src_mask_image, w, h,
								stride_w, SCAN_WIDTH(src_mask_image), pxy_ptr, run2_ptr, pixels, run_count);
				UNMAP_IMAGE(src_mask_image);
			}
			else if (CPOBase::bitCheck(param_ptr->flag, kRunTableFlagMaskVal))
			{
				status |= _vxImage2RunTableWithMaskVal(img_ptr, w, h, stride_w, param_ptr->mask_val, pxy_ptr, run2_ptr, pixels, run_count);
			}

			status |= OvxHelper::writeArray(out_array, &pixels, sizeof(pixels), 1);
			status |= OvxHelper::appendArray(out_array, &run_count, sizeof(run_count), 1);
			status |= OvxHelper::appendArray(out_array, pxy_ptr, sizeof(i32)*(param_ptr->height + 1), 1);
			status |= OvxHelper::appendArray(out_array, run2_ptr, sizeof(u32)*run_count, 1);
			POSAFE_DELETE_ARRAY(pxy_ptr);
			POSAFE_DELETE_ARRAY(run2_ptr);
		}
		UNMAP_IMAGE(src_image);
		UNMAP_ARRAY(in_param);
#endif
	}
	return status;
}

vx_status convertImg2RunTableKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertImg2RunTableKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertImg2RunTableKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kConvertImg2RunTableKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_size capacity = OvxHelper::getCapacity((vx_array)parameters[kConvertImg2RunTableKernelOutArray]);
	vx_int32 item_type = OvxHelper::getFormat((vx_array)parameters[kConvertImg2RunTableKernelOutArray]);

	/* output validation */
	status |= vxSetMetaFormatAttribute(metas[kConvertImg2RunTableKernelOutArray], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
	status |= vxSetMetaFormatAttribute(metas[kConvertImg2RunTableKernelOutArray], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status _vxRunTable2Image(u8*img_ptr, i32 w, i32 h, i32 stride_w, i32* pxy_ptr, u16* run2_ptr, u8 val)
{
	if (!img_ptr || w*h <= 0 || !pxy_ptr || !run2_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	i32 k, y, st_pos, ed_pos;
	u8* scan_imgy_ptr;
	for (y = 0; y < h; y++)
	{
		scan_imgy_ptr = img_ptr + y*stride_w;
		st_pos = pxy_ptr[y];
		ed_pos = pxy_ptr[y + 1];
		for (k = st_pos; k < ed_pos; k += 2)
		{
			memset(scan_imgy_ptr + run2_ptr[k], val, run2_ptr[k + 1]);
		}
	}
	return VX_SUCCESS;
}

vx_status convertRunTable2ImgKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_array in_array = (vx_array)parameters[kConvertRunTable2ImgKernelInArray];
	vx_array in_param = (vx_array)parameters[kConvertRunTable2ImgKernelParam];
	vx_image out_image = (vx_image)parameters[kConvertRunTable2ImgKernelOutImage];

	if (count == kConvertRunTable2ImgKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		MAP_ARRAY(in_param, VX_READ_ONLY);
		NVX_MAP_ARRAY(in_array, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		i32 w, h, val;
		if (status == VX_SUCCESS)
		{
			vxParamConvertRunTable2Img* param_ptr = (vxParamConvertRunTable2Img*)data_in_param;
			w = param_ptr->width;
			h = param_ptr->height;
			val = param_ptr->value;
			i32 stride_w = SCAN_WIDTH(out_image);
			u8* img_ptr = (u8*)data_out_image;
			i32* pxy_ptr = (i32*)((u8*)data_in_array + sizeof(i32) * 2);
			u16* run2_ptr = (u16*)((u8*)pxy_ptr + sizeof(i32)*(h + 1));

			status |= cuRuntable2Image(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, val);
		}
		UNMAP_ARRAY(in_param);
		NVX_UNMAP_ARRAY(in_array);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_ARRAY(in_array, VX_READ_ONLY);
		MAP_ARRAY(in_param, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		i32 w, h, val;
		if (status == VX_SUCCESS)
		{
			vxParamConvertRunTable2Img* param_ptr = (vxParamConvertRunTable2Img*)data_in_param;
			w = param_ptr->width;
			h = param_ptr->height;
			val = param_ptr->value;
			i32 stride_w = SCAN_WIDTH(out_image);
			u8* img_ptr = (u8*)data_out_image;
			i32* pxy_ptr = (i32*)((u8*)data_in_array + sizeof(i32) * 2);
			u16* run2_ptr = (u16*)((u8*)pxy_ptr + sizeof(i32)*(h + 1));

			memset(img_ptr, 0, stride_w*h);
			status |= _vxRunTable2Image(img_ptr, w, h, stride_w, pxy_ptr, run2_ptr, val);
		}
		UNMAP_ARRAY(in_array);
		UNMAP_ARRAY(in_param);
		UNMAP_IMAGE(out_image);
#endif
		status |= OvxHelper::setValidRectOnly(out_image, w, h);
	}
	return status;
}

vx_status convertRunTable2ImgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertRunTable2ImgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status convertRunTable2ImgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kConvertRunTable2ImgKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kConvertRunTable2ImgKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kConvertRunTable2ImgKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kConvertRunTable2ImgKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kConvertRunTable2ImgKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kConvertRunTable2ImgKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kConvertRunTable2ImgKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}
#endif

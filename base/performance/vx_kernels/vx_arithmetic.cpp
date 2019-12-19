#include "vx_arithmetic.h"

#if defined(POR_WITH_OVX)

#include "vx_api_image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_CUDA)
#include "performance/cuda_kernels/nvx_arithmetic.h"
#endif

vx_status absKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kAbsKernelParamInImage];
	vx_image out_image = (vx_image)parameters[kAbsKernelParamOutImage];
	
	if (count == kAbsKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 2:
				{
					status |= cuAbs_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(i16*)data_out_image, SCAN_WIDTH(out_image));
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 2:
				{
					status |= _vxAbs_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(i16*)data_out_image, SCAN_WIDTH(out_image));
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status absKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status absKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status absKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kAbsKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kAbsKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kAbsKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kAbsKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kAbsKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kAbsKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kAbsKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status addKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image1 = (vx_image)parameters[kAddKernelParamInImage1];
	vx_image in_image2 = (vx_image)parameters[kAddKernelParamInImage2];
	vx_image out_image = (vx_image)parameters[kAddKernelParamOutImage];

	if (count == kAddKernelParamCount)
	{
		status = VX_SUCCESS;  //TODO: Not Implemented...
	}
	return status;
}

vx_status addKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status addKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status addKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kAddKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kAddKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kAddKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kAddKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kAddKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kAddKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kAddKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status mulKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image1 = (vx_image)parameters[kMulKernelParamInImage1];
	vx_image in_image2 = (vx_image)parameters[kMulKernelParamInImage2];
	vx_image out_image = (vx_image)parameters[kMulKernelParamOutImage];

	if (count == kMulKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image1, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_image2, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS &&
			addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 && addr_out_image.stride_x == 2)
		{
			status |= cuMul_u8u8u16((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u16*)data_out_image, SCAN_WIDTH(out_image));
		}
		NVX_UNMAP_IMAGE(in_image1);
		NVX_UNMAP_IMAGE(in_image2);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image1, VX_READ_ONLY);
		MAP_IMAGE(in_image2, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS &&
			addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 && addr_out_image.stride_x == 2)
		{
			status |= _vxMul_u8u8u16((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u16*)data_out_image, SCAN_WIDTH(out_image));
		}
		UNMAP_IMAGE(in_image1);
		UNMAP_IMAGE(in_image2);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image1);
	}
	return status;
}

vx_status mulKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status mulKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status mulKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kMulKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kMulKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kMulKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kMulKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kMulKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kMulKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kMulKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status addConstKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kAddConstKernelParamInImage];
	vx_image out_image = (vx_image)parameters[kAddConstKernelParamOutImage];
	vx_scalar in_add_const = (vx_scalar)parameters[kAddConstKernelParamInAddConst];

	if (count == kAddConstKernelParamCount)
	{
		i32 value = 0;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&value, in_add_const);

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= cuAddConst_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
				case 2:
				{
					status |= cuAddConst_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(i16*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= _vxAddConst_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
						(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
				case 2:
				{
					status |= _vxAddConst_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
						(i16*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status addConstKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status addConstKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status addConstKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kAddConstKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kAddConstKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kAddConstKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kAddConstKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kAddConstKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kAddConstKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kAddConstKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status mulConstKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kMulConstKernelParamInImage];
	vx_image out_image = (vx_image)parameters[kMulConstKernelParamOutImage];
	vx_scalar in_mul_const = (vx_scalar)parameters[kMulConstKernelParamInMulConst];

	if (count == kMulConstKernelParamCount)
	{
		status = VX_SUCCESS;
		i32 const_type = OvxHelper::getFormat(in_mul_const);

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					switch (const_type)
					{
						case VX_TYPE_FLOAT32:
						{
							f32 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= cuMulConstf32_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(u8*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
						case VX_TYPE_UINT8:
						{
							u8 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= cuMulConsti32_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(u8*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
					}
					break;
				}
				case 2:
				{
					switch (const_type)
					{
						case VX_TYPE_FLOAT32:
						{
							f32 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= cuMulConstf32_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(i16*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
						case VX_TYPE_UINT8:
						{
							u8 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= cuMulConsti32_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(i16*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
					}
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					switch (const_type)
					{
						case VX_TYPE_FLOAT32:
						{
							f32 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= _vxMulConstf32_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
										(u8*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
						case VX_TYPE_UINT8:
						{
							u8 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= _vxMulConsti32_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(u8*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
					}
					break;
				}
				case 2:
				{
					switch (const_type)
					{
						case VX_TYPE_UINT8:
						{
							u8 value = 0;
							status |= OvxHelper::readScalar(&value, in_mul_const);
							status |= _vxMulConsti32_i16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
											(i16*)data_out_image, SCAN_WIDTH(out_image), value);
							break;
						}
					}
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status mulConstKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status mulConstKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status mulConstKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kMulConstKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 w = OvxHelper::getWidth((vx_image)parameters[kMulConstKernelParamOutImage]);
	vx_uint32 h = OvxHelper::getHeight((vx_image)parameters[kMulConstKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kMulConstKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kMulConstKernelParamOutImage], VX_IMAGE_WIDTH, &w, sizeof(w));
	status |= vxSetMetaFormatAttribute(metas[kMulConstKernelParamOutImage], VX_IMAGE_HEIGHT, &h, sizeof(h));
	status |= vxSetMetaFormatAttribute(metas[kMulConstKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status minKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image1 = (vx_image)parameters[kMinKernelParamInImage1];
	vx_image in_image2 = (vx_image)parameters[kMinKernelParamInImage2];
	vx_image out_image = (vx_image)parameters[kMinKernelParamOutImage];

	if (count == kMinKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image1, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_image2, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			if (addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 &&
				addr_out_image.stride_x == 1)
			{
				status |= cuMin_u8u8u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
			}
		}
		NVX_UNMAP_IMAGE(in_image1);
		NVX_UNMAP_IMAGE(in_image2);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image1, VX_READ_ONLY);
		MAP_IMAGE(in_image2, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			if (addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 &&
				addr_out_image.stride_x == 1)
			{
				status |= _vxMin_u8u8u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
			}
		}
		UNMAP_IMAGE(in_image1);
		UNMAP_IMAGE(in_image2);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image1);
	}
	return status;
}

vx_status minKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status minKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status minKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kMinKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 w = OvxHelper::getWidth((vx_image)parameters[kMinKernelParamOutImage]);
	vx_uint32 h = OvxHelper::getHeight((vx_image)parameters[kMinKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kMinKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kMinKernelParamOutImage], VX_IMAGE_WIDTH, &w, sizeof(w));
	status |= vxSetMetaFormatAttribute(metas[kMinKernelParamOutImage], VX_IMAGE_HEIGHT, &h, sizeof(h));
	status |= vxSetMetaFormatAttribute(metas[kMinKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status maxKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image1 = (vx_image)parameters[kMaxKernelParamInImage1];
	vx_image in_image2 = (vx_image)parameters[kMaxKernelParamInImage2];
	vx_image out_image = (vx_image)parameters[kMaxKernelParamOutImage];

	if (count == kMaxKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image1, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_image2, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			if (addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 &&
				addr_out_image.stride_x == 1)
			{
				status |= cuMax_u8u8u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
			}
		}
		NVX_UNMAP_IMAGE(in_image1);
		NVX_UNMAP_IMAGE(in_image2);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image1, VX_READ_ONLY);
		MAP_IMAGE(in_image2, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS)
		{
			if (addr_in_image1.stride_x == 1 && addr_in_image2.stride_x == 1 &&
				addr_out_image.stride_x == 1)
			{
				status |= _vxMax_u8u8u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1),
								(u8*)data_in_image2, SCAN_WIDTH(in_image2),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
			}
		}
		UNMAP_IMAGE(in_image1);
		UNMAP_IMAGE(in_image2);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image1);
	}
	return status;
}

vx_status maxKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status maxKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status maxKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kMaxKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 w = OvxHelper::getWidth((vx_image)parameters[kMaxKernelParamOutImage]);
	vx_uint32 h = OvxHelper::getHeight((vx_image)parameters[kMaxKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kMaxKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kMaxKernelParamOutImage], VX_IMAGE_WIDTH, &w, sizeof(w));
	status |= vxSetMetaFormatAttribute(metas[kMaxKernelParamOutImage], VX_IMAGE_HEIGHT, &h, sizeof(h));
	status |= vxSetMetaFormatAttribute(metas[kMaxKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status clipMinKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kClipMinKernelParamInImage];
	vx_image out_image = (vx_image)parameters[kClipMinKernelParamOutImage];
	vx_scalar in_clip_min = (vx_scalar)parameters[kClipMinKernelParamInValue];

	if (count == kClipMinKernelParamCount)
	{
		i32 value = 0;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&value, in_clip_min);

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= cuClipMin_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(out_image);
#else

		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= _vxClipMin_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status clipMinKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status clipMinKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status clipMinKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kClipMinKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kClipMinKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kClipMinKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kClipMinKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kClipMinKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kClipMinKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kClipMinKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

vx_status clipMaxKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kClipMaxKernelParamInImage];
	vx_image out_image = (vx_image)parameters[kClipMaxKernelParamOutImage];
	vx_scalar in_clip_max = (vx_scalar)parameters[kClipMaxKernelParamInValue];

	if (count == kClipMinKernelParamCount)
	{
		i32 value = 0;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&value, in_clip_max);

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= cuClipMax_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 1:
				{
					status |= _vxClipMax_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_out_image, SCAN_WIDTH(out_image), value);
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status clipMaxKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status clipMaxKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status clipMaxKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kClipMaxKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kClipMaxKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kClipMaxKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kClipMaxKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kClipMaxKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kClipMaxKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kClipMaxKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////

vx_status cutKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kCutKernelParamInImage];
	vx_image in_mask_image = (vx_image)parameters[kCutKernelParamInMaskImage];
	vx_scalar in_threshold = (vx_scalar)parameters[kCutKernelParamThreshold];
	vx_image out_image = (vx_image)parameters[kCutKernelParamOutImage];
	vx_scalar out_valid_pixels = (vx_scalar)parameters[kCutKernelParamOutPixelCount];

	if (count == kCutKernelParamCount)
	{
		i32 valid_pixels = 0;
		i32 threshold;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&threshold, in_threshold);
		
#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_mask_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 2:
				{
					status |= cuCut_u16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_in_mask_image, SCAN_WIDTH(in_mask_image), threshold,
									(i16*)data_out_image, SCAN_WIDTH(out_image), &valid_pixels);
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(in_mask_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(in_mask_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 2:
				{
					status |= _vxCut_u16((i16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
									(u8*)data_in_mask_image, SCAN_WIDTH(in_mask_image), threshold,
									(i16*)data_out_image, SCAN_WIDTH(out_image), valid_pixels);
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(in_mask_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
		status |= OvxHelper::writeScalar(out_valid_pixels, &valid_pixels);
	}
	return status;
}

vx_status cutKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status cutKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status cutKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kCutKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kCutKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kCutKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kCutKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kCutKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kCutKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kCutKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));

	vx_uint32 datatype = OvxHelper::getFormat((vx_scalar)parameters[kCutKernelParamOutPixelCount]);
	status |= vxSetMetaFormatAttribute(metas[kCutKernelParamOutPixelCount], VX_SCALAR_TYPE, &datatype, sizeof(datatype));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status subtractExKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image1 = (vx_image)parameters[kSubtractExKernelParamInImage1];
	vx_image in_image2 = (vx_image)parameters[kSubtractExKernelParamInImage2];
	vx_image out_image = (vx_image)parameters[kSubtractExKernelParamOutImage];
	vx_scalar in_alpha = (vx_scalar)parameters[kSubtractExKernelParamInAlpha];
	vx_scalar in_beta = (vx_scalar)parameters[kSubtractExKernelParamInBeta];

	if (count == kSubtractExKernelParamCount)
	{
		f32 alpha, beta;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&alpha, in_alpha);
		status |= OvxHelper::readScalar(&beta, in_beta);

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image1, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_image2, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && 
			addr_in_image1.stride_x == addr_out_image.stride_x &&
			addr_in_image2.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image1.stride_x)
			{
				case 1:
				{
					status |= cuSubtractEx_u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1), alpha,
									(u8*)data_in_image2, SCAN_WIDTH(in_image2), beta,
									(u8*)data_out_image, SCAN_WIDTH(out_image));
					break;
				}
			}
		}
		NVX_UNMAP_IMAGE(in_image1);
		NVX_UNMAP_IMAGE(in_image2);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image1, VX_READ_ONLY);
		MAP_IMAGE(in_image2, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS &&
			addr_in_image1.stride_x == addr_out_image.stride_x &&
			addr_in_image2.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image1.stride_x)
			{
				case 1:
				{
					status |= _vxSubtractEx_u8((u8*)data_in_image1, rect_in_image1, SCAN_WIDTH(in_image1), alpha,
									(u8*)data_in_image2, SCAN_WIDTH(in_image2), beta, 
									(u8*)data_out_image, SCAN_WIDTH(out_image));
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image1);
		UNMAP_IMAGE(in_image2);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image1);
	}
	return status;
}

vx_status subtractExKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status subtractExKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status subtractExKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kSubtractExKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kSubtractExKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kSubtractExKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kSubtractExKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kSubtractExKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kSubtractExKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kSubtractExKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status maskKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kMaskKernelParamInImage];
	vx_image in_mask_image = (vx_image)parameters[kMaskKernelParamInMaskImage];
	vx_image out_image = (vx_image)parameters[kMaskKernelParamOutImage];

	if (count == kMaskKernelParamCount)
	{
		status = VX_SUCCESS;

#if defined(POR_WITH_CUDA)
		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(in_mask_image, VX_READ_ONLY);
		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (addr_in_image.stride_x == 1 && addr_in_mask_image.stride_x == 1)
		{
			status |= cuMask_u8_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
								(u8*)data_in_mask_image, SCAN_WIDTH(in_mask_image),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
		}
		else
		{
			status = VX_ERROR_INVALID_PARAMETERS;
		}
		NVX_UNMAP_IMAGE(in_image);
		NVX_UNMAP_IMAGE(in_mask_image);
		NVX_UNMAP_IMAGE(out_image);
#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(in_mask_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (addr_in_image.stride_x == 1 && addr_in_mask_image.stride_x == 1)
		{
			status |= _vxMask_u8_u8((u8*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
								(u8*)data_in_mask_image, SCAN_WIDTH(in_mask_image),
								(u8*)data_out_image, SCAN_WIDTH(out_image));
		}
		else
		{
			status = VX_ERROR_INVALID_PARAMETERS;
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(in_mask_image);
		UNMAP_IMAGE(out_image);
#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);
	}
	return status;
}

vx_status maskKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status maskKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status maskKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kMaskKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kMaskKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kMaskKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kMaskKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kMaskKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kMaskKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kMaskKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}
#endif
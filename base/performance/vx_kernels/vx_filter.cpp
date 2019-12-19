#include "vx_filter.h"

#if defined(POR_WITH_OVX)
#include "vx_api_filter.h"
#include "performance/openvx_pool/ovx_base.h"

#if defined(POR_WITH_CUDA)
#include "performance/cuda_kernels/nvx_filter.h"
#endif

//#define POR_TESTMODE
#if defined(POR_TESTMODE)
#include "proc/image_proc.h"
#endif

vx_status gaussian2dKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image in_image = (vx_image)parameters[kGaussian2dKernelParamInImage];
	vx_scalar in_kernel_size = (vx_scalar)parameters[kGaussian2dKernelParamKernelSize];
	vx_image out_image = (vx_image)parameters[kGaussian2dKernelParamOutImage];
	
	if (count == kGaussian2dKernelParamCount)
	{
		i32 kernel_size = 1;
		status = VX_SUCCESS;
		status |= OvxHelper::readScalar(&kernel_size, in_kernel_size);

//#if defined(POR_WITH_CUDA)
//		NVX_MAP_IMAGE(in_image, VX_READ_ONLY);
//		NVX_MAP_IMAGE(out_image, VX_WRITE_ONLY);
//
//		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
//		{
//			switch (addr_in_image.stride_x)
//			{
//				case 2:
//				{
//					status |= cuGaussian2d_u16((u16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
//									(u16*)data_out_image, SCAN_WIDTH(out_image), kernel_size);
//					break;
//				}
//			}
//		}
//		NVX_UNMAP_IMAGE(in_image);
//		NVX_UNMAP_IMAGE(out_image);
//#else
		MAP_IMAGE(in_image, VX_READ_ONLY);
		MAP_IMAGE(out_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && addr_in_image.stride_x == addr_out_image.stride_x)
		{
			switch (addr_in_image.stride_x)
			{
				case 2:
				{
					status |= _vxGaussian2d_u16((u16*)data_in_image, rect_in_image, SCAN_WIDTH(in_image),
										(u16*)data_out_image, SCAN_WIDTH(out_image), kernel_size);
					break;
				}
			}
		}
		UNMAP_IMAGE(in_image);
		UNMAP_IMAGE(out_image);
//#endif
		status |= vxSetImageValidRectangle(out_image, &rect_in_image);

#if defined(POR_TESTMODE)
		CImageProc::saveImgOpenVx(PO_DEBUG_PATH"vx_src.bmp", in_image);
		CImageProc::saveImgOpenVx(PO_DEBUG_PATH"vx_out.bmp", out_image);
#endif
	}
	return status;
}

vx_status gaussian2dKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status gaussian2dKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status gaussian2dKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kGaussian2dKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kGaussian2dKernelParamOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kGaussian2dKernelParamOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kGaussian2dKernelParamOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kGaussian2dKernelParamOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kGaussian2dKernelParamOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kGaussian2dKernelParamOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}
#endif
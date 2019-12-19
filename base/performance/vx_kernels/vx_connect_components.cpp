#include "vx_connect_components.h"
#if defined(POR_WITH_OVX)

#include "vx_api_image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

vx_status connectComponentsKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	i32 ret_count = 0;
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kConnectComponentsKernelInImage];
	vx_image src_mask_image = (vx_image)parameters[kConnectComponentsKernelInMaskImage];
	vx_image dst_image = (vx_image)parameters[kConnectComponentsKernelOutImage];
	vx_scalar cc_count = (vx_scalar)parameters[kConnectComponentsKernelOutCount];

	if (count == kConnectComponentsKernelParamCount)
	{
		status = VX_SUCCESS;

		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_IMAGE(src_mask_image, VX_READ_ONLY);
		MAP_IMAGE(dst_image, VX_WRITE_ONLY);
		
		if (status == VX_SUCCESS)
		{
			ret_count = _vxConnectComponents((u8*)data_src_image, rect_src_image, SCAN_WIDTH(src_image),
									(u8*)data_src_mask_image, rect_src_mask_image, SCAN_WIDTH(src_mask_image),
									(u16*)data_dst_image, rect_dst_image, SCAN_WIDTH(dst_image));
		}
		
		UNMAP_IMAGE(src_image);
		UNMAP_IMAGE(src_mask_image);
		UNMAP_IMAGE(dst_image);
		VX_CHKRET_O(OvxHelper::setValidRectOnly(dst_image, RECT_WIDTH(rect_src_image), RECT_HEIGHT(rect_src_image)));
		VX_CHKRET_O(OvxHelper::writeScalar(cc_count, &ret_count));
	}
	return status;
}

vx_status connectComponentsKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status connectComponentsKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status connectComponentsKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kConnectComponentsKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* input validation */

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kConnectComponentsKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kConnectComponentsKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kConnectComponentsKernelOutImage]);
	vx_int32 count_format = OvxHelper::getFormat((vx_scalar)parameters[kConnectComponentsKernelOutCount]);

	status |= vxSetMetaFormatAttribute(metas[kConnectComponentsKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kConnectComponentsKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kConnectComponentsKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	status |= vxSetMetaFormatAttribute(metas[kConnectComponentsKernelOutCount], VX_SCALAR_TYPE, &count_format, sizeof(count_format));
	return status;
}
#endif
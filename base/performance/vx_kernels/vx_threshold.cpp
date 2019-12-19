#include "vx_threshold.h"
#if defined(POR_WITH_OVX)

#include "vx_api_image_proc.h"
#include "performance/openvx_pool/ovx_base.h"

vx_status autoThresholdKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_image src_image = (vx_image)parameters[kAutoThresholdKernelInImage];
	vx_image src_mask_image = (vx_image)parameters[kAutoThresholdKernelInMaskImage];
	vx_threshold out_threshold = (vx_threshold)parameters[kAutoThresholdKernelOutThreshold];

	i32 ret_threshold = 0;
	if (count == kAutoThresholdKernelParamCount)
	{
		status = VX_SUCCESS;

		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_IMAGE(src_mask_image, VX_READ_ONLY);

		if (status == VX_SUCCESS)
		{
			i32 hist[256];
			memset(hist, 0, sizeof(i32) * 256);
			_vxHistogram((u8*)data_src_image, (u8*)data_src_mask_image, rect_src_image, SCAN_WIDTH(src_image),
					SCAN_WIDTH(src_mask_image), hist, NULL);
			ret_threshold = _vxCalcThreshold(hist, 1, 255);
		}
		UNMAP_IMAGE(src_image);
		UNMAP_IMAGE(src_mask_image);
		VX_CHKRET_O(OvxHelper::writeThreshold(out_threshold, ret_threshold));
	}
	return status;
}

vx_status autoThresholdKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status autoThresholdKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status autoThresholdKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kAutoThresholdKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_int32 threshold_type = 0;
	vxQueryThreshold((vx_threshold)parameters[kAutoThresholdKernelOutThreshold], VX_THRESHOLD_TYPE, &threshold_type, sizeof(threshold_type));
	status |= vxSetMetaFormatAttribute(metas[kAutoThresholdKernelOutThreshold], VX_THRESHOLD_TYPE, &threshold_type, sizeof(threshold_type));
	return status;

}
#endif
#include "vx_caliper.h"

#if defined(POR_WITH_OVX)
#include "performance/openvx_pool/ovx_base.h"
#include "proc/find_caliper.h"

vx_status findCaliperKernel(vx_node node, const vx_reference* parameters, vx_uint32 count)
{
	vx_status status = VX_SUCCESS;
	vx_image src_image = (vx_image)parameters[kCaliperFindKernelInImage];
	vx_scalar in_count = (vx_scalar)parameters[kCaliperFindKernelInCaliperCount];
	vx_array in_param = (vx_array)parameters[kCaliperFindKernelInCaliperParam];
	vx_array out_vec = (vx_array)parameters[kCaliperFindKernelOutCaliperVec];

	if (count == kCaliperFindKernelParamCount)
	{
		MAP_IMAGE(src_image, VX_READ_ONLY);
		MAP_ARRAY(in_param, VX_READ_ONLY);
		MAP_ARRAY(out_vec, VX_READ_AND_WRITE);

		if (status == VX_SUCCESS)
		{
			i32 i, count;
			OvxHelper::readScalar(&count, in_count);
			if (CPOBase::isPositive(count))
			{
				i32 w = RECT_WIDTH(rect_src_image);
				i32 h = RECT_HEIGHT(rect_src_image);
				i32 stride_w = SCAN_WIDTH(src_image);
				u8* img_ptr = (u8*)data_src_image;
				Caliper* caliper_ptr = (Caliper*)data_out_vec;
				CaliperParam* caliper_param_ptr = (CaliperParam*)data_in_param;

				for (i = 0; i < count; i++)
				{
					CFindCaliper::findEdgesInCaliper(img_ptr, w, h, stride_w, *(caliper_ptr + i), *(caliper_param_ptr));
				}
			}
		}

		UNMAP_IMAGE(src_image);
		UNMAP_ARRAY(in_param);
		UNMAP_ARRAY(out_vec);
	}
	return status;
}

vx_status findCaliperKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status findCaliperKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status findCaliperKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kCaliperFindKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_size capacity = OvxHelper::getCapacity((vx_array)parameters[kCaliperFindKernelOutCaliperVec]);
	vx_int32 item_type = OvxHelper::getFormat((vx_array)parameters[kCaliperFindKernelOutCaliperVec]);

	/* output validation */
	status |= vxSetMetaFormatAttribute(metas[kCaliperFindKernelOutCaliperVec], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
	status |= vxSetMetaFormatAttribute(metas[kCaliperFindKernelOutCaliperVec], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
	return status;
}
#endif
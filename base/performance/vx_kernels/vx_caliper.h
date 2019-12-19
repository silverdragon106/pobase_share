#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_caliper_find_kernel_parameter_e
{
	kCaliperFindKernelInImage,
	kCaliperFindKernelInCaliperCount,
	kCaliperFindKernelInCaliperParam,
	kCaliperFindKernelOutCaliperVec,
	kCaliperFindKernelParamCount
};

vx_status findCaliperKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status findCaliperKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status findCaliperKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status findCaliperKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif
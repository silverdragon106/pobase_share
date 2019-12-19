#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_auto_threshold_kernel_parameter_e
{
	kAutoThresholdKernelInImage,
	kAutoThresholdKernelInMaskImage,
	kAutoThresholdKernelOutThreshold,
	kAutoThresholdKernelParamCount
};

vx_status autoThresholdKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status autoThresholdKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status autoThresholdKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status autoThresholdKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif

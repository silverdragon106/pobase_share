#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_gaussian2d_kernel_parameter_e
{
	kGaussian2dKernelParamInImage,
	kGaussian2dKernelParamKernelSize,
	kGaussian2dKernelParamOutImage,

	kGaussian2dKernelParamCount
};

vx_status gaussian2dKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status gaussian2dKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status gaussian2dKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status gaussian2dKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif

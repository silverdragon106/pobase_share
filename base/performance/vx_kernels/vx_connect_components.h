#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_connect_components_kernel_parameter_e
{
	kConnectComponentsKernelInImage,
	kConnectComponentsKernelInMaskImage,
	kConnectComponentsKernelOutImage,
	kConnectComponentsKernelOutCount,
	kConnectComponentsKernelParamCount
};

vx_status connectComponentsKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status connectComponentsKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status connectComponentsKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status connectComponentsKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif
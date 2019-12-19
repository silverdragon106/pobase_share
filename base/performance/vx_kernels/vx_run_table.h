#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_convert_img_to_runtable_kernel_parameter_e
{
	kConvertImg2RunTableKernelInImage,
	kConvertImg2RunTableKernelInMaskImage,
	kConvertImg2RunTableKernelParam,
	kConvertImg2RunTableKernelOutArray,
	kConvertImg2RunTableKernelParamCount
};

vx_status convertImg2RunTableKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertImg2RunTableKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertImg2RunTableKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertImg2RunTableKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_convert_runtable_to_img_kernel_parameter_e
{
	kConvertRunTable2ImgKernelInArray,
	kConvertRunTable2ImgKernelParam,
	kConvertRunTable2ImgKernelOutImage,
	kConvertRunTable2ImgKernelParamCount
};

vx_status convertRunTable2ImgKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertRunTable2ImgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertRunTable2ImgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertRunTable2ImgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif
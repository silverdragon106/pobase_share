#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_convert_hsv_kernel_parameter_e
{
	kConvertToHSVKernelInImage,
	kConvertToHSVKernelOutImage,
	kConvertToHSVKernelParamCount
};

vx_status convertToHSVKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToHSVKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToHSVKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToHSVKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_convert_avg_kernel_parameter_e
{
	kConvertToAvgKernelInImage,
	kConvertToAvgKernelOutImage,
	kConvertToAvgKernelParamCount
};

vx_status convertToAvgKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToAvgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToAvgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status convertToAvgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_palette_img_kernel_parameter_e
{
	kPaletteImageKernelInImage,
	kPaletteImageKernelInPalette,
	kPaletteImageKernelOutImage,
	kPaletteImageKernelParamCount
};

vx_status paletteImgKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status paletteImgKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status paletteImgKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status paletteImgKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif
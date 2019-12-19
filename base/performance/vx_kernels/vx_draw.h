#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_draw_circle_kernel_parameter_e
{
	kDrawCircleKernelInParamArray,
	kDrawCircleKernelOutImage,
	kDrawCircleKernelParamCount
};

vx_status drawCircleKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawCircleKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawCircleKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawCircleKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_draw_ellipse_kernel_parameter_e
{
	kDrawEllipseKernelInParamArray,
	kDrawEllipseKernelOutImage,
	kDrawEllipseKernelParamCount
};

vx_status drawEllipseKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawEllipseKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawEllipseKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawEllipseKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_draw_polygon_kernel_parameter_e
{
	kDrawPolygonKernelInParamArray,
	kDrawPolygonKernelOutImage,
	kDrawPolygonKernelParamCount
};

vx_status drawPolygonKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawPolygonKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawPolygonKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawPolygonKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_draw_ring_kernel_parameter_e
{
	kDrawRingKernelInParamArray,
	kDrawRingKernelOutImage,
	kDrawRingKernelParamCount
};

vx_status drawRingKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawRingKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawRingKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status drawRingKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif
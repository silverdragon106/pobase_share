#pragma once
#include "config.h"
#include "vx_config.h"
#include "vx_kernel_types.h"

#if defined(POR_WITH_OVX)
#include "VX/vx.h"

/* User kernel names */
#define POVX_KERNEL_NAME_ABS					"vx.po.common.abs"
#define POVX_KERNEL_NAME_ADD					"vx.po.common.add"
#define POVX_KERNEL_NAME_MUL					"vx.po.common.mul"
#define POVX_KERNEL_NAME_ADD_CONST				"vx.po.common.add.const"
#define POVX_KERNEL_NAME_MUL_CONST				"vx.po.common.mul.const"
#define POVX_KERNEL_NAME_MIN					"vx.po.common.min"
#define POVX_KERNEL_NAME_MAX					"vx.po.common.max"
#define POVX_KERNEL_NAME_CUT					"vx.po.common.add.cut"
#define POVX_KERNEL_NAME_CLIP_MIN				"vx.po.common.clip.min"
#define POVX_KERNEL_NAME_CLIP_MAX				"vx.po.common.clip.max"
#define POVX_KERNEL_NAME_SUBTRACT_EX			"vx.po.common.sub.ex"
#define POVX_KERNEL_NAME_MASK					"vx.po.common.mask"

#define POVX_KERNEL_NAME_FILTER_GAUSSIAN2D		"vx.po.filter.gaussian2d"

#define POVX_KERNEL_NAME_AUTO_THRESHOLD			"vx.po.threshold.auto"

#define POVX_KERNEL_NAME_CONNECT_COMPONENNTS	"vx.po.connect.components"

#define POVX_KERNEL_NAME_CALIPER_FIND			"vx.po.caliper.find"

#define POVX_KERNEL_NAME_IMG_TO_RUNTABLE		"vx.po.imgproc.imgtoruntable"
#define POVX_KERNEL_NAME_RUNTABLE_TO_IMG		"vx.po.imgproc.runtabletoimg"

#define POVX_KERNEL_NAME_PALETTE				"vx.po.imgproc.palette"
#define POVX_KERNEL_NAME_CONVERT_TO_HSV			"vx.po.imgproc.cvt.hsv"
#define POVX_KERNEL_NAME_CONVERT_TO_AVG			"vx.po.imgproc.cvt.avg"

#define POVX_KERNEL_NAME_DRAW_CIRCLE			"vx.po.common.draw.circle"
#define POVX_KERNEL_NAME_DRAW_ELLIPSE			"vx.po.common.draw.ellipse"
#define POVX_KERNEL_NAME_DRAW_RING				"vx.po.common.draw.ring"
#define POVX_KERNEL_NAME_DRAW_POLYGON			"vx.po.common.draw.polygon"

/* User kernel enums */
#define POVX_LIBRARY_KERNEL						(0x001)
#define POVX_LIBRARY_TILED_KERNEL				(0x100)

enum vx_kernel_porbase_e
{
	/* diff image tiled kernel */
	POVX_KERNEL_ABS = VX_KERNEL_BASE(VX_ID_DEFAULT, POVX_LIBRARY_KERNEL) + 0x0,
	POVX_KERNEL_ADD,
	POVX_KERNEL_MUL,
	POVX_KERNEL_ADD_CONST,
	POVX_KERNEL_MUL_CONST,
	POVX_KERNEL_MIN,
	POVX_KERNEL_MAX,
	POVX_KERNEL_CUT,
	POVX_KERNEL_CLIP_MIN,
	POVX_KERNEL_CLIP_MAX,
	POVX_KERNEL_SUBTRACT_EX,
	POVX_KERNEL_MASK,

	POVX_KERNEL_FILTER_GAUSSIAN2D,

	POVX_KERNEL_AUTO_THRESHOLD,

	POVX_KERNEL_CONNECT_COMPONENNTS,

	POVX_KERNEL_CALIPER_FIND,

	POVX_KERNEL_IMG_TO_RUNTABLE,
	POVX_KERNEL_RUNTABLE_TO_IMG,

	POVX_KERNEL_PALETTE,
	POVX_KERNEL_CONVERT_TO_HSV,
	POVX_KERNEL_CONVERT_TO_AVG,

	POVX_KERNEL_DRAW_CIRCLE,
	POVX_KERNEL_DRAW_ELLIPSE,
	POVX_KERNEL_DRAW_RING,
	POVX_KERNEL_DRAW_POLYGON,
	
	POVX_KERNEL_EXTENSION
};

class OvxKernel
{
public:
	static bool			publishKernels(vx_context context);

#if defined(POVX_USE_ARITHMETIC)
	static vx_status	publishAbs(vx_context context);
	static vx_status	publishAddEx(vx_context context);
	static vx_status	publishMul(vx_context context);
	static vx_status	publishAddConst(vx_context context);
	static vx_status	publishMulConst(vx_context context);
	static vx_status	publishMin(vx_context context);
	static vx_status	publishMax(vx_context context);
	static vx_status	publishCut(vx_context context);
	static vx_status	publishClipMin(vx_context context);
	static vx_status	publishClipMax(vx_context context);
	static vx_status	publishSubtractEx(vx_context context);
	static vx_status	publishMask(vx_context context);
#endif
#if defined(POVX_USE_FILTER)
	static vx_status	publishGaussian2d(vx_context context);
#endif
#if defined(POVX_USE_THRESHOLD)
	static vx_status	publishAutoThreshold(vx_context context);
#endif
#if defined(POVX_USE_CONNECT_COMPONENNTS)
	static vx_status	publishConnectComponents(vx_context context);
#endif
#if defined(POVX_USE_CALIPER)
	static vx_status	publishCaliperFind(vx_context context);
#endif
#if defined(POVX_USE_RUNTABLE)
	static vx_status	publishConvertImgToRunTable(vx_context context);
	static vx_status	publishConvertRunTableToImg(vx_context context);
#endif
#if defined(POVX_USE_COLOR)
	static vx_status	publishConvertToHSV(vx_context context);
	static vx_status	publishConvertToAvg(vx_context context);
	static vx_status	publishPalette(vx_context context);
#endif

#if defined(POVX_USE_DRAW)
	static vx_status	publishDrawCircle(vx_context context);
	static vx_status	publishDrawEllipse(vx_context context);
	static vx_status	publishDrawRing(vx_context context);
	static vx_status	publishDrawPolygon(vx_context context);
#endif
};

class OvxCustomNode
{
public:
#if defined(POVX_USE_ARITHMETIC)
	static vx_node		vxAbsNode(vx_graph graph,
							vx_image in_image, vx_image out_image);
	static vx_node		vxAddExNode(vx_graph graph, 
							vx_image in_image1, vx_image in_image2, vx_image out_image);
	static vx_node		vxMulNode(vx_graph graph, 
							vx_image in_image1, vx_image in_image2, vx_image out_image);
	static vx_node		vxAddConstNode(vx_graph graph,
							vx_image in_image, vx_scalar in_add_const, vx_image out_image);
	static vx_node		vxMulConstNode(vx_graph graph,
							vx_image in_image, vx_scalar in_mul_const, vx_image out_image);
	static vx_node		vxMinNode(vx_graph graph,
							vx_image in_image1, vx_image in_image2, vx_image out_image);
	static vx_node		vxMaxNode(vx_graph graph,
							vx_image in_image1, vx_image in_image2, vx_image out_image);
	static vx_node		vxCutNode(vx_graph graph,
							vx_image in_image, vx_image in_src_mask, vx_scalar in_threshold,
							vx_image out_image, vx_scalar out_valid_pixels);
	static vx_node		vxClipMinNode(vx_graph graph,
							vx_image in_image, vx_scalar in_min_clip, vx_image out_image);
	static vx_node		vxClipMaxNode(vx_graph graph,
							vx_image in_image, vx_scalar in_max_clip, vx_image out_image);
	static vx_node		vxSubtractExNode(vx_graph graph, 
							vx_image in_image1, vx_scalar in_alpha,
							vx_image in_image2, vx_scalar in_beta, vx_image out_image);
	static vx_node		vxMaskNode(vx_graph graph, 
							vx_image in_image, vx_image in_mask_image, vx_image out_image);
#endif
#if defined(POVX_USE_FILTER)
	static vx_node		vxGaussian2dNode(vx_graph graph,
							vx_image in_image, vx_image out_image, vx_scalar in_kernel_size);
#endif
#if defined(POVX_USE_THRESHOLD)
	static vx_node		vxAutoThresholdNode(vx_graph graph,
							vx_image in_image, vx_image in_mask_image,
							vx_threshold out_threshold);
#endif
#if defined(POVX_USE_CONNECT_COMPONENNTS)
	static vx_node		vxConnectComponentsNode(vx_graph graph,
							vx_image in_image, vx_image in_mask_image,
							vx_image out_image, vx_scalar out_cc_count);
#endif
#if defined(POVX_USE_CALIPER)
	static vx_node		vxCaliperFindNode(vx_graph graph,
							vx_image src_image, vx_scalar in_count, vx_array in_param,
							vx_array out_vec);
#endif
#if defined(POVX_USE_RUNTABLE)
	static vx_node		vxConvertImgToRunTableNode(vx_graph graph, 
							vx_image src_image, vx_image src_mask_image, vx_array img2run_param,
							vx_array dst_runtable);
	static vx_node		vxConvertRunTableToImgNode(vx_graph graph,
							vx_array src_runtable, vx_array run2img_param, vx_image dst_image);
#endif
#if defined(POVX_USE_COLOR)
	static vx_node		vxPaletteImgNode(vx_graph graph,
							vx_image src_image, vx_array in_palette, vx_image dst_image);
	static vx_node		vxConvertToHSVNode(vx_graph graph,
							vx_image src_image, vx_image dst_image);
	static vx_node		vxConvertToAvgNode(vx_graph graph,
							vx_image src_image, vx_image dst_image);
#endif

#if defined(POVX_USE_DRAW)
	static vx_node		vxDrawCircleNode(vx_graph graph,
							vx_array in_draw_param, vx_image out_image);
	static vx_node		vxDrawEllipseNode(vx_graph graph, 
							vx_array in_draw_param, vx_image out_image);
	static vx_node		vxDrawRingNode(vx_graph graph,
							vx_array in_draw_param, vx_image out_image);
	static vx_node		vxDrawPolygonNode(vx_graph graph,
							vx_array in_draw_param, vx_image out_image);
#endif
};
#endif
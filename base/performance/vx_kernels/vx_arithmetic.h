#pragma once
#include "config.h"
#if defined(POR_WITH_OVX)

#include <VX/vx.h>

enum vx_abs_kernel_parameter_e
{
	kAbsKernelParamInImage,
	kAbsKernelParamOutImage,
	kAbsKernelParamCount
};

vx_status absKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status absKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status absKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status absKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_add_kernel_parameter_e
{
	kAddKernelParamInImage1,
	kAddKernelParamInImage2,
	kAddKernelParamOutImage,
	kAddKernelParamCount
};

vx_status addKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_mul_kernel_parameter_e
{
	kMulKernelParamInImage1,
	kMulKernelParamInImage2,
	kMulKernelParamOutImage,
	kMulKernelParamCount
};

vx_status mulKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_add_const_kernel_parameter_e
{
	kAddConstKernelParamInImage,
	kAddConstKernelParamInAddConst,
	kAddConstKernelParamOutImage,
	kAddConstKernelParamCount
};

vx_status addConstKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addConstKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addConstKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status addConstKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_mul_const_kernel_parameter_e
{
	kMulConstKernelParamInImage,
	kMulConstKernelParamInMulConst,
	kMulConstKernelParamOutImage,
	kMulConstKernelParamCount
};

vx_status mulConstKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulConstKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulConstKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status mulConstKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_min_kernel_parameter_e
{
	kMinKernelParamInImage1,
	kMinKernelParamInImage2,
	kMinKernelParamOutImage,
	kMinKernelParamCount
};

vx_status minKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status minKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status minKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status minKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_max_kernel_parameter_e
{
	kMaxKernelParamInImage1,
	kMaxKernelParamInImage2,
	kMaxKernelParamOutImage,
	kMaxKernelParamCount
};

vx_status maxKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maxKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maxKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maxKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_clip_min_kernel_parameter_e
{
	kClipMinKernelParamInImage,
	kClipMinKernelParamInValue,
	kClipMinKernelParamOutImage,
	kClipMinKernelParamCount
};

vx_status clipMinKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMinKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMinKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMinKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_clip_max_kernel_parameter_e
{
	kClipMaxKernelParamInImage,
	kClipMaxKernelParamInValue,
	kClipMaxKernelParamOutImage,
	kClipMaxKernelParamCount
};

vx_status clipMaxKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMaxKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMaxKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status clipMaxKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_cut_kernel_parameter_e
{
	kCutKernelParamInImage,
	kCutKernelParamInMaskImage,
	kCutKernelParamThreshold,
	kCutKernelParamOutImage,
	kCutKernelParamOutPixelCount,
	kCutKernelParamCount
};

vx_status cutKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status cutKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status cutKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status cutKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_subtract_ex_kernel_parameter_e
{
	kSubtractExKernelParamInImage1,
	kSubtractExKernelParamInAlpha,
	kSubtractExKernelParamInImage2,
	kSubtractExKernelParamInBeta,
	kSubtractExKernelParamOutImage,
	kSubtractExKernelParamCount
};

vx_status subtractExKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status subtractExKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status subtractExKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status subtractExKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);

enum vx_mask_kernel_parameter_e
{
	kMaskKernelParamInImage,
	kMaskKernelParamInMaskImage,
	kMaskKernelParamOutImage,
	kMaskKernelParamCount
};

vx_status maskKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maskKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maskKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num);
vx_status maskKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
#endif

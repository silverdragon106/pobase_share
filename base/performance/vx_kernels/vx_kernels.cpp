#include "vx_kernels.h"
#include "logger/logger.h"

#if defined(POR_WITH_OVX)
#if defined(POVX_USE_ARITHMETIC)
#include "performance/vx_kernels/vx_arithmetic.h"
#endif

#if defined(POVX_USE_FILTER)
#include "performance/vx_kernels/vx_filter.h"
#endif

#if defined(POVX_USE_DRAW)
#include "performance/vx_kernels/vx_draw.h"
#endif

#if defined(POVX_USE_CONNECT_COMPONENNTS)
#include "performance/vx_kernels/vx_connect_components.h"
#endif

#if defined(POVX_USE_CALIPER)
#include "performance/vx_kernels/vx_caliper.h"
#endif

#if defined(POVX_USE_THRESHOLD)
#include "performance/vx_kernels/vx_threshold.h"
#endif

#if defined(POVX_USE_RUNTABLE)
#include "performance/vx_kernels/vx_run_table.h"
#endif

#if defined(POVX_USE_COLOR)
#include "performance/vx_kernels/vx_color.h"
#endif
//////////////////////////////////////////////////////////////////////////

bool OvxKernel::publishKernels(vx_context context)
{
	//Common
	vx_status status = VX_SUCCESS;

#if defined(POVX_USE_ARITHMETIC)
	if ((status |= publishAbs(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishAbs failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishAddEx(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishAdd failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishMul(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishMul failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishAddConst(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishAddConst failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishMulConst(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishMulConst failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishMin(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishMin failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishMax(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishMax failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishCut(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishCut failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishClipMin(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishClipMin failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishClipMax(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishClipMax failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishSubtractEx(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishSubtractEx failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishMask(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishMask failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_FILTER)
	if ((status |= publishGaussian2d(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishGaussian2d failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_THRESHOLD)
	if ((status |= publishAutoThreshold(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishAutoThreshold failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_CONNECT_COMPONENNTS)
	if ((status |= publishConnectComponents(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishConnectComponents failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_CALIPER)
	if ((status |= publishCaliperFind(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishCaliperFind failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_RUNTABLE)
	if ((status |= publishConvertImgToRunTable(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishImageToRunTable failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishConvertRunTableToImg(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishRunTableToImage failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
#if defined(POVX_USE_COLOR)
	if ((status |= publishConvertToHSV(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishConvertToHSV failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishConvertToAvg(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishConvertToAvg failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishPalette(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishPalette failed", LOG_SCOPE_OVX);
		return false;
	}
#endif

#if defined(POVX_USE_DRAW)
	if ((status |= publishDrawCircle(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishDrawCircle failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishDrawEllipse(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishDrawEllipse failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishDrawRing(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishDrawRing failed", LOG_SCOPE_OVX);
		return false;
	}
	if ((status |= publishDrawPolygon(context)) != VX_SUCCESS)
	{
		printlog_lvs2("_publishDrawPolygon failed", LOG_SCOPE_OVX);
		return false;
	}
#endif
	return true;
}

#if defined(POVX_USE_ARITHMETIC)
vx_status OvxKernel::publishAbs(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_ABS,
							POVX_KERNEL_ABS,
							absKernel,					/* process-local function pointer of this kernel to be invoked */
							kAbsKernelParamCount,		/* parameter count of this kernel */
							absKernelValidator,			/* parameter validator callback */
							absKernelInitialize,		/* kernel initialization function */
							absKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kAbsKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAbs publish failed, kAbsKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAbsKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAbs publish failed, kAbsKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAbs publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishAddEx(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
									POVX_KERNEL_NAME_ADD,
									POVX_KERNEL_ADD,
									addKernel,				/* process-local function pointer of this kernel to be invoked */
									kAddKernelParamCount,	/* parameter count of this kernel */
									addKernelValidator,		/* parameter validator callback */
									addKernelInitialize,	/* kernel initialization function */
									addKernelDeinitialize	/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kAddKernelParamInImage1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAdd publish failed, kAddKernelParamInImage1", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAddKernelParamInImage2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAdd publish failed, kAddKernelParamInImage2", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAddKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAdd publish failed, kAddKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAdd publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishMul(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_MUL,
								POVX_KERNEL_MUL,
								mulKernel,				/* process-local function pointer of this kernel to be invoked */
								kMulKernelParamCount,	/* parameter count of this kernel */
								mulKernelValidator,		/* parameter validator callback */
								mulKernelInitialize,	/* kernel initialization function */
								mulKernelDeinitialize	/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kMulKernelParamInImage1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMul publish failed, kMulKernelParamInImage1", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMulKernelParamInImage2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMul publish failed, kMulKernelParamInImage2", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMulKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMul publish failed, kMulKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMul publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishAddConst(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_ADD_CONST,
							POVX_KERNEL_ADD_CONST,
							addConstKernel,					/* process-local function pointer of this kernel to be invoked */
							kAddConstKernelParamCount,		/* parameter count of this kernel */
							addConstKernelValidator,		/* parameter validator callback */
							addConstKernelInitialize,		/* kernel initialization function */
							addConstKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kAddConstKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAddConst publish failed, kAddConstKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAddConstKernelParamInAddConst, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAddConst publish failed, kAddConstKernelParamInAddConst", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAddConstKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAddConst publish failed, kAddConstKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAddConst publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishMulConst(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_MUL_CONST,
								POVX_KERNEL_MUL_CONST,
								mulConstKernel,					/* process-local function pointer of this kernel to be invoked */
								kMulConstKernelParamCount,		/* parameter count of this kernel */
								mulConstKernelValidator,		/* parameter validator callback */
								mulConstKernelInitialize,		/* kernel initialization function */
								mulConstKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kMulConstKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMulConst publish failed, kMulConstKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMulConstKernelParamInMulConst, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMulConst publish failed, kMulConstKernelParamInMulConst", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMulConstKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMulConst publish failed, kMulConstKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMulConst publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishMin(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
									POVX_KERNEL_NAME_MIN,
									POVX_KERNEL_MIN,
									minKernel,				/* process-local function pointer of this kernel to be invoked */
									kMinKernelParamCount,	/* parameter count of this kernel */
									minKernelValidator,		/* parameter validator callback */
									minKernelInitialize,	/* kernel initialization function */
									minKernelDeinitialize	/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kMinKernelParamInImage1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMin publish failed, kMinKernelParamInImage1", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMinKernelParamInImage2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMin publish failed, kMinKernelParamInImage2", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMinKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMin publish failed, kMinKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMin publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishMax(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
									POVX_KERNEL_NAME_MAX,
									POVX_KERNEL_MAX,
									maxKernel,				/* process-local function pointer of this kernel to be invoked */
									kMaxKernelParamCount,	/* parameter count of this kernel */
									maxKernelValidator,		/* parameter validator callback */
									maxKernelInitialize,	/* kernel initialization function */
									maxKernelDeinitialize	/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kMaxKernelParamInImage1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMax publish failed, kMaxKernelParamInImage1", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMaxKernelParamInImage2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMax publish failed, kMaxKernelParamInImage2", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMaxKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMax publish failed, kMaxKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMax publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishCut(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CUT,
							POVX_KERNEL_CUT,
							cutKernel,					/* process-local function pointer of this kernel to be invoked */
							kCutKernelParamCount,		/* parameter count of this kernel */
							cutKernelValidator,			/* parameter validator callback */
							cutKernelInitialize,		/* kernel initialization function */
							cutKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kCutKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, kCutKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCutKernelParamInMaskImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, kCutKernelParamInMaskImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCutKernelParamThreshold, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, kCutKernelParamThreshold", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCutKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, kCutKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCutKernelParamOutPixelCount, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, kCutKernelParamOutPixelCount", LOG_SCOPE_OVX);
			return status;
		}
		
		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("_publishCut publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishClipMin(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CLIP_MIN,
							POVX_KERNEL_CLIP_MIN,
							clipMinKernel,					/* process-local function pointer of this kernel to be invoked */
							kClipMinKernelParamCount,		/* parameter count of this kernel */
							clipMinKernelValidator,			/* parameter validator callback */
							clipMinKernelInitialize,		/* kernel initialization function */
							clipMinKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kClipMinKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMin publish failed, kClipMinKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kClipMinKernelParamInValue, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMin publish failed, kClipMinKernelParamInValue", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kClipMinKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMin publish failed, kClipMinKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMin publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishClipMax(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CLIP_MAX,
							POVX_KERNEL_CLIP_MAX,
							clipMaxKernel,					/* process-local function pointer of this kernel to be invoked */
							kClipMaxKernelParamCount,		/* parameter count of this kernel */
							clipMaxKernelValidator,			/* parameter validator callback */
							clipMaxKernelInitialize,		/* kernel initialization function */
							clipMaxKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kClipMaxKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMax publish failed, kClipMaxKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kClipMaxKernelParamInValue, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMax publish failed, kClipMaxKernelParamInValue", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kClipMaxKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMax publish failed, kClipMaxKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishClipMax publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishSubtractEx(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_SUBTRACT_EX,
							POVX_KERNEL_SUBTRACT_EX,
							subtractExKernel,				/* process-local function pointer of this kernel to be invoked */
							kSubtractExKernelParamCount,	/* parameter count of this kernel */
							subtractExKernelValidator,		/* parameter validator callback */
							subtractExKernelInitialize,		/* kernel initialization function */
							subtractExKernelDeinitialize	/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kSubtractExKernelParamInImage1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, kSubtractExKernelParamInImage1", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kSubtractExKernelParamInAlpha, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, kSubtractExKernelParamInAlpha", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kSubtractExKernelParamInImage2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, kSubtractExKernelParamInImage2", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kSubtractExKernelParamInBeta, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, kSubtractExKernelParamInBeta", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kSubtractExKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, kSubtractExKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishSubtractEx publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishMask(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_MASK,
							POVX_KERNEL_MASK,
							maskKernel,				/* process-local function pointer of this kernel to be invoked */
							kMaskKernelParamCount,	/* parameter count of this kernel */
							maskKernelValidator,	/* parameter validator callback */
							maskKernelInitialize,	/* kernel initialization function */
							maskKernelDeinitialize	/* kernel deinitialization function */
						);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kMaskKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMask publish failed, kMaskKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMaskKernelParamInMaskImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMask publish failed, kMaskKernelParamInMaskImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kMaskKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMask publish failed, kMaskKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishMask publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_FILTER)
vx_status OvxKernel::publishGaussian2d(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_FILTER_GAUSSIAN2D,
								POVX_KERNEL_FILTER_GAUSSIAN2D,
								gaussian2dKernel,				/* process-local function pointer of this kernel to be invoked */
								kGaussian2dKernelParamCount,	/* parameter count of this kernel */
								gaussian2dKernelValidator,		/* parameter validator callback */
								gaussian2dKernelInitialize,		/* kernel initialization function */
								gaussian2dKernelDeinitialize	/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kGaussian2dKernelParamInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishGaussian2d publish failed, kGaussian2dKernelParamInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kGaussian2dKernelParamKernelSize, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishGaussian2d publish failed, kGaussian2dKernelParamKernelSize", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kGaussian2dKernelParamOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishGaussian2d publish failed, kGaussian2dKernelParamOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishGaussian2d publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_DRAW)
vx_status OvxKernel::publishDrawCircle(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_DRAW_CIRCLE,
								POVX_KERNEL_DRAW_CIRCLE,
								drawCircleKernel,					/* process-local function pointer of this kernel to be invoked */
								kDrawCircleKernelParamCount,		/* parameter count of this kernel */
								drawCircleKernelValidator,			/* parameter validator callback */
								drawCircleKernelInitialize,			/* kernel initialization function */
								drawCircleKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kDrawCircleKernelInParamArray, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawCircle publish failed, kDrawCircleKernelInParamArray", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kDrawCircleKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawCircle publish failed, kDrawCircleKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}
		
		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawCircle publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishDrawEllipse(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_DRAW_ELLIPSE,
								POVX_KERNEL_DRAW_ELLIPSE,
								drawEllipseKernel,					/* process-local function pointer of this kernel to be invoked */
								kDrawEllipseKernelParamCount,		/* parameter count of this kernel */
								drawEllipseKernelValidator,			/* parameter validator callback */
								drawEllipseKernelInitialize,		/* kernel initialization function */
								drawEllipseKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kDrawEllipseKernelInParamArray, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawEllipse publish failed, kDrawEllipseKernelInParamArray", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kDrawEllipseKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawEllipse publish failed, kDrawEllipseKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawEllipse publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishDrawRing(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_DRAW_RING,
								POVX_KERNEL_DRAW_RING,
								drawRingKernel,					/* process-local function pointer of this kernel to be invoked */
								kDrawRingKernelParamCount,		/* parameter count of this kernel */
								drawRingKernelValidator,			/* parameter validator callback */
								drawRingKernelInitialize,		/* kernel initialization function */
								drawRingKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kDrawRingKernelInParamArray, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawRing publish failed, kDrawRingKernelInParamArray", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kDrawRingKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawRing publish failed, kDrawRingKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawRing publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishDrawPolygon(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_DRAW_POLYGON,
								POVX_KERNEL_DRAW_POLYGON,
								drawPolygonKernel,					/* process-local function pointer of this kernel to be invoked */
								kDrawPolygonKernelParamCount,		/* parameter count of this kernel */
								drawPolygonKernelValidator,			/* parameter validator callback */
								drawPolygonKernelInitialize,		/* kernel initialization function */
								drawPolygonKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kDrawPolygonKernelInParamArray, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawPolygon publish failed, kDrawPolygonKernelInParamArray", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kDrawPolygonKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawPolygon publish failed, kDrawPolygonKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishDrawPolygon publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_THRESHOLD)
vx_status OvxKernel::publishAutoThreshold(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
								POVX_KERNEL_NAME_AUTO_THRESHOLD,
								POVX_KERNEL_AUTO_THRESHOLD,
								autoThresholdKernel,				/* process-local function pointer of this kernel to be invoked */
								kAutoThresholdKernelParamCount,		/* parameter count of this kernel */
								autoThresholdKernelValidator,		/* parameter validator callback */
								autoThresholdKernelInitialize, 		/* kernel initialization function */
								autoThresholdKernelDeinitialize		/* kernel deinitialization function */
								);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kAutoThresholdKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAutoThreshold publish failed, kAutoThresholdKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAutoThresholdKernelInMaskImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAutoThreshold publish failed, kAutoThresholdKernelInMaskImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kAutoThresholdKernelOutThreshold, VX_OUTPUT, VX_TYPE_THRESHOLD, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAutoThreshold publish failed, kAutoThresholdKernelOutThreshold", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishAutoThreshold publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_CONNECT_COMPONENNTS)
vx_status OvxKernel::publishConnectComponents(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CONNECT_COMPONENNTS,
							POVX_KERNEL_CONNECT_COMPONENNTS,
							connectComponentsKernel,				/* process-local function pointer of this kernel to be invoked */
							kConnectComponentsKernelParamCount,		/* parameter count of this kernel */
							connectComponentsKernelValidator,		/* parameter validator callback */
							connectComponentsKernelInitialize, 		/* kernel initialization function */
							connectComponentsKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kConnectComponentsKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConnectComponents publish failed, kConnectComponentsKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConnectComponentsKernelInMaskImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConnectComponents publish failed, kConnectComponentsKernelInMaskImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConnectComponentsKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConnectComponents publish failed, kConnectComponentsKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConnectComponentsKernelOutCount, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConnectComponents publish failed, kConnectComponentsKernelOutCount", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConnectComponents publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_CALIPER)
vx_status OvxKernel::publishCaliperFind(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CALIPER_FIND,
							POVX_KERNEL_CALIPER_FIND,
							findCaliperKernel,					/* process-local function pointer of this kernel to be invoked */
							kCaliperFindKernelParamCount,		/* parameter count of this kernel */
							findCaliperKernelValidator,			/* parameter validator callback */
							findCaliperKernelInitialize, 		/* kernel initialization function */
							findCaliperKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kCaliperFindKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishCaliperFind publish failed, kCaliperFindKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCaliperFindKernelInCaliperCount, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishCaliperFind publish failed, kCaliperFindKernelInCaliperCount", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCaliperFindKernelInCaliperParam, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishCaliperFind publish failed, kCaliperFindKernelInCaliperParam", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kCaliperFindKernelOutCaliperVec, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishCaliperFind publish failed, kCaliperFindKernelOutCaliperVec", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishCaliperFind publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_RUNTABLE)
vx_status OvxKernel::publishConvertImgToRunTable(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_IMG_TO_RUNTABLE,
							POVX_KERNEL_IMG_TO_RUNTABLE,
							convertImg2RunTableKernel,					/* process-local function pointer of this kernel to be invoked */
							kConvertImg2RunTableKernelParamCount,		/* parameter count of this kernel */
							convertImg2RunTableKernelValidator,			/* parameter validator callback */
							convertImg2RunTableKernelInitialize, 		/* kernel initialization function */
							convertImg2RunTableKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kConvertImg2RunTableKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertImgToRunTable publish failed, kConvertImg2RunTableKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertImg2RunTableKernelInMaskImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertImgToRunTable publish failed, kConvertImg2RunTableKernelInMaskImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertImg2RunTableKernelParam, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertImgToRunTable publish failed, kConvertImg2RunTableKernelParam", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertImg2RunTableKernelOutArray, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertImgToRunTable publish failed, kConvertImg2RunTableKernelOutArray", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertImgToRunTable publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishConvertRunTableToImg(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_RUNTABLE_TO_IMG,
							POVX_KERNEL_RUNTABLE_TO_IMG,
							convertRunTable2ImgKernel,					/* process-local function pointer of this kernel to be invoked */
							kConvertRunTable2ImgKernelParamCount,		/* parameter count of this kernel */
							convertRunTable2ImgKernelValidator,			/* parameter validator callback */
							convertRunTable2ImgKernelInitialize, 		/* kernel initialization function */
							convertRunTable2ImgKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kConvertRunTable2ImgKernelInArray, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertRunTableToImg publish failed, kConvertRunTable2ImgKernelInArray", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertRunTable2ImgKernelParam, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertRunTableToImg publish failed, kConvertRunTable2ImgKernelParam", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertRunTable2ImgKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertRunTableToImg publish failed, kConvertRunTable2ImgKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertRunTableToImg publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

#if defined(POVX_USE_COLOR)
vx_status OvxKernel::publishConvertToHSV(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CONVERT_TO_HSV,
							POVX_KERNEL_CONVERT_TO_HSV,
							convertToHSVKernel,					/* process-local function pointer of this kernel to be invoked */
							kConvertToHSVKernelParamCount,		/* parameter count of this kernel */
							convertToHSVKernelValidator,		/* parameter validator callback */
							convertToHSVKernelInitialize, 		/* kernel initialization function */
							convertToHSVKernelDeinitialize		/* kernel deinitialization function */
						);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kConvertToHSVKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToHSV publish failed, kConvertToHSVKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertToHSVKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToHSV publish failed, kConvertToHSVKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToHSV publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishConvertToAvg(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_CONVERT_TO_AVG,
							POVX_KERNEL_CONVERT_TO_AVG,
							convertToAvgKernel,					/* process-local function pointer of this kernel to be invoked */
							kConvertToAvgKernelParamCount,		/* parameter count of this kernel */
							convertToAvgKernelValidator,		/* parameter validator callback */
							convertToAvgKernelInitialize, 		/* kernel initialization function */
							convertToAvgKernelDeinitialize		/* kernel deinitialization function */
						);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kConvertToAvgKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToAvg publish failed, kConvertToAvgKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kConvertToAvgKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToAvg publish failed, kConvertToAvgKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishConvertToAvg publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}

vx_status OvxKernel::publishPalette(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddUserKernel(context,
							POVX_KERNEL_NAME_PALETTE,
							POVX_KERNEL_PALETTE,
							paletteImgKernel,					/* process-local function pointer of this kernel to be invoked */
							kPaletteImageKernelParamCount,		/* parameter count of this kernel */
							paletteImgKernelValidator,			/* parameter validator callback */
							paletteImgKernelInitialize, 		/* kernel initialization function */
							paletteImgKernelDeinitialize		/* kernel deinitialization function */
							);

	if (kernel)
	{
		status |= vxAddParameterToKernel(kernel, kPaletteImageKernelInImage, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishPalette publish failed, kPaletteImageKernelInImage", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kPaletteImageKernelInPalette, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishPalette publish failed, kPaletteImageKernelInPalette", LOG_SCOPE_OVX);
			return status;
		}
		status |= vxAddParameterToKernel(kernel, kPaletteImageKernelOutImage, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishPalette publish failed, kPaletteImageKernelOutImage", LOG_SCOPE_OVX);
			return status;
		}

		status |= vxFinalizeKernel(kernel);
		if (status != VX_SUCCESS)
		{
			vxRemoveKernel(kernel);
			printlog_lvs2("publishPalette publish failed, vxFinalizeKernel", LOG_SCOPE_OVX);
			return status;
		}
	}
	return status;
}
#endif

//////////////////////////////////////////////////////////////////////////
#if defined(POVX_USE_ARITHMETIC)
vx_node OvxCustomNode::vxAbsNode(vx_graph graph, vx_image in_image, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_ABS);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kAbsKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kAbsKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxAbsNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxAbsNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxAddExNode(vx_graph graph, vx_image in_image1, vx_image in_image2, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_ADD);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kAddKernelParamInImage1, (vx_reference)in_image1);
			status |= vxSetParameterByIndex(node, kAddKernelParamInImage2, (vx_reference)in_image2);
			status |= vxSetParameterByIndex(node, kAddKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxAddNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxAddNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxMulNode(vx_graph graph, vx_image in_image1, vx_image in_image2, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_MUL);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kMulKernelParamInImage1, (vx_reference)in_image1);
			status |= vxSetParameterByIndex(node, kMulKernelParamInImage2, (vx_reference)in_image2);
			status |= vxSetParameterByIndex(node, kMulKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxMulNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxMulNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxAddConstNode(vx_graph graph, vx_image in_image, vx_scalar in_add_const, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_ADD_CONST);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kAddConstKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kAddConstKernelParamInAddConst, (vx_reference)in_add_const);
			status |= vxSetParameterByIndex(node, kAddConstKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxAddConstNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxAddConstNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxMulConstNode(vx_graph graph, vx_image in_image, vx_scalar in_mul_const, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_MUL_CONST);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kMulConstKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kMulConstKernelParamInMulConst, (vx_reference)in_mul_const);
			status |= vxSetParameterByIndex(node, kMulConstKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxMulConstNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxMulConstNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxMinNode(vx_graph graph, vx_image in_image1, vx_image in_image2, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_MIN);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kMinKernelParamInImage1, (vx_reference)in_image1);
			status |= vxSetParameterByIndex(node, kMinKernelParamInImage2, (vx_reference)in_image2);
			status |= vxSetParameterByIndex(node, kMinKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxMinNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxMinNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxMaxNode(vx_graph graph, vx_image in_image1, vx_image in_image2, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_MAX);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kMaxKernelParamInImage1, (vx_reference)in_image1);
			status |= vxSetParameterByIndex(node, kMaxKernelParamInImage2, (vx_reference)in_image2);
			status |= vxSetParameterByIndex(node, kMaxKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxMaxNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxMaxNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxCutNode(vx_graph graph, vx_image in_image, vx_image in_src_mask, vx_scalar in_threshold,
				vx_image out_image, vx_scalar out_valid_pixels)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CUT);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kCutKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kCutKernelParamInMaskImage, (vx_reference)in_src_mask);
			status |= vxSetParameterByIndex(node, kCutKernelParamThreshold, (vx_reference)in_threshold);
			status |= vxSetParameterByIndex(node, kCutKernelParamOutImage, (vx_reference)out_image);
			status |= vxSetParameterByIndex(node, kCutKernelParamOutPixelCount, (vx_reference)out_valid_pixels);
			
			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxCutNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxCutNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxClipMinNode(vx_graph graph, vx_image in_image, vx_scalar in_min_clip, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CLIP_MIN);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kClipMinKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kClipMinKernelParamInValue, (vx_reference)in_min_clip);
			status |= vxSetParameterByIndex(node, kClipMinKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxClipMinNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxClipMinNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxClipMaxNode(vx_graph graph, vx_image in_image, vx_scalar in_max_clip, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CLIP_MAX);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kClipMaxKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kClipMaxKernelParamInValue, (vx_reference)in_max_clip);
			status |= vxSetParameterByIndex(node, kClipMaxKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxClipMaxNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxClipMaxNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxSubtractExNode(vx_graph graph, vx_image in_image1, vx_scalar in_alpha,
					vx_image in_image2, vx_scalar in_beta, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_SUBTRACT_EX);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kSubtractExKernelParamInImage1, (vx_reference)in_image1);
			status |= vxSetParameterByIndex(node, kSubtractExKernelParamInAlpha, (vx_reference)in_alpha);
			status |= vxSetParameterByIndex(node, kSubtractExKernelParamInImage2, (vx_reference)in_image2);
			status |= vxSetParameterByIndex(node, kSubtractExKernelParamInBeta, (vx_reference)in_beta);
			status |= vxSetParameterByIndex(node, kSubtractExKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxSubtractExNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxSubtractExNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxMaskNode(vx_graph graph, vx_image in_image, vx_image in_mask_image,
							vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_MASK);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kMaskKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kMaskKernelParamInMaskImage, (vx_reference)in_mask_image);
			status |= vxSetParameterByIndex(node, kMaskKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxMaskNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxMaskNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_FILTER)
vx_node OvxCustomNode::vxGaussian2dNode(vx_graph graph, vx_image in_image, vx_image out_image, vx_scalar in_kernel_size)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_FILTER_GAUSSIAN2D);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kGaussian2dKernelParamInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kGaussian2dKernelParamKernelSize, (vx_reference)in_kernel_size);
			status |= vxSetParameterByIndex(node, kGaussian2dKernelParamOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxGaussian2dNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxGaussian2dNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_DRAW)
vx_node OvxCustomNode::vxDrawCircleNode(vx_graph graph, vx_array in_draw_param, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_DRAW_CIRCLE);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kDrawCircleKernelInParamArray, (vx_reference)in_draw_param);
			status |= vxSetParameterByIndex(node, kDrawCircleKernelOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxDrawCircleNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxDrawCircleNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxDrawEllipseNode(vx_graph graph, vx_array in_draw_param, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_DRAW_ELLIPSE);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kDrawEllipseKernelInParamArray, (vx_reference)in_draw_param);
			status |= vxSetParameterByIndex(node, kDrawEllipseKernelOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxDrawEllipseNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxDrawEllipseNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxDrawRingNode(vx_graph graph, vx_array in_draw_param, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_DRAW_RING);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kDrawRingKernelInParamArray, (vx_reference)in_draw_param);
			status |= vxSetParameterByIndex(node, kDrawRingKernelOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxDrawRingNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxDrawRingNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxDrawPolygonNode(vx_graph graph, vx_array in_draw_param, vx_image out_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_DRAW_POLYGON);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kDrawPolygonKernelInParamArray, (vx_reference)in_draw_param);
			status |= vxSetParameterByIndex(node, kDrawPolygonKernelOutImage, (vx_reference)out_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxDrawPolygonNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxDrawPolygonNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_THRESHOLD)
vx_node OvxCustomNode::vxAutoThresholdNode(vx_graph graph, vx_image in_image, vx_image in_mask_image, vx_threshold out_threshold)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_AUTO_THRESHOLD);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kAutoThresholdKernelInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kAutoThresholdKernelInMaskImage, (vx_reference)in_mask_image);
			status |= vxSetParameterByIndex(node, kAutoThresholdKernelOutThreshold, (vx_reference)out_threshold);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxAutoThresholdNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxAutoThresholdNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_CONNECT_COMPONENNTS)
vx_node OvxCustomNode::vxConnectComponentsNode(vx_graph graph, vx_image in_image, vx_image in_mask_image, vx_image out_image, vx_scalar out_cc_count)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CONNECT_COMPONENNTS);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kConnectComponentsKernelInImage, (vx_reference)in_image);
			status |= vxSetParameterByIndex(node, kConnectComponentsKernelInMaskImage, (vx_reference)in_mask_image);
			status |= vxSetParameterByIndex(node, kConnectComponentsKernelOutImage, (vx_reference)out_image);
			status |= vxSetParameterByIndex(node, kConnectComponentsKernelOutCount, (vx_reference)out_cc_count);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxConnectComponentsNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxConnectComponentsNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_CALIPER)
vx_node OvxCustomNode::vxCaliperFindNode(vx_graph graph, vx_image src_image, vx_scalar in_count, vx_array in_param, vx_array out_vec)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CALIPER_FIND);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kCaliperFindKernelInImage, (vx_reference)src_image);
			status |= vxSetParameterByIndex(node, kCaliperFindKernelInCaliperCount, (vx_reference)in_count);
			status |= vxSetParameterByIndex(node, kCaliperFindKernelInCaliperParam, (vx_reference)in_param);
			status |= vxSetParameterByIndex(node, kCaliperFindKernelOutCaliperVec, (vx_reference)out_vec);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxCaliperFindNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxCaliperFindNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_RUNTABLE)
vx_node OvxCustomNode::vxConvertImgToRunTableNode(vx_graph graph, vx_image src_image, vx_image src_mask_image, vx_array img2run_param, vx_array dst_runtable)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_IMG_TO_RUNTABLE);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kConvertImg2RunTableKernelInImage, (vx_reference)src_image);
			status |= vxSetParameterByIndex(node, kConvertImg2RunTableKernelInMaskImage, (vx_reference)src_mask_image);
			status |= vxSetParameterByIndex(node, kConvertImg2RunTableKernelParam, (vx_reference)img2run_param);
			status |= vxSetParameterByIndex(node, kConvertImg2RunTableKernelOutArray, (vx_reference)dst_runtable);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxConvertImgToRunTableNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxConvertImgToRunTableNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxConvertRunTableToImgNode(vx_graph graph, vx_array src_runtable, vx_array run2img_param, vx_image dst_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_RUNTABLE_TO_IMG);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kConvertRunTable2ImgKernelInArray, (vx_reference)src_runtable);
			status |= vxSetParameterByIndex(node, kConvertRunTable2ImgKernelParam, (vx_reference)run2img_param);
			status |= vxSetParameterByIndex(node, kConvertRunTable2ImgKernelOutImage, (vx_reference)dst_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxConvertRunTableToImgNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxConvertRunTableToImgNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif

#if defined(POVX_USE_COLOR)
vx_node OvxCustomNode::vxConvertToHSVNode(vx_graph graph, vx_image src_image, vx_image dst_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CONVERT_TO_HSV);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kConvertToHSVKernelInImage, (vx_reference)src_image);
			status |= vxSetParameterByIndex(node, kConvertToHSVKernelOutImage, (vx_reference)dst_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxConvertToHSVNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxConvertToHSVNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxConvertToAvgNode(vx_graph graph, vx_image src_image, vx_image dst_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_CONVERT_TO_AVG);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kConvertToAvgKernelInImage, (vx_reference)src_image);
			status |= vxSetParameterByIndex(node, kConvertToAvgKernelOutImage, (vx_reference)dst_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxConvertToAvgNode invalid parameter!");
				return NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxConvertToAvgNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}

vx_node OvxCustomNode::vxPaletteImgNode(vx_graph graph, vx_image src_image, vx_array in_palette, vx_image dst_image)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, POVX_KERNEL_NAME_PALETTE);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_status status = VX_SUCCESS;
			status |= vxSetParameterByIndex(node, kPaletteImageKernelInImage, (vx_reference)src_image);
			status |= vxSetParameterByIndex(node, kPaletteImageKernelInPalette, (vx_reference)in_palette);
			status |= vxSetParameterByIndex(node, kPaletteImageKernelOutImage, (vx_reference)dst_image);

			if (status != VX_SUCCESS)
			{
				/* InvalidParameter Error. */
				vxReleaseNode(&node);
				vxReleaseKernel(&kernel);
				vxAddLogEntry(NULL, VX_FAILURE, "vxPaletteImgNode invalid parameter!");
				node = NULL;
				kernel = NULL;
			}
		}
		else
		{
			vxAddLogEntry(NULL, VX_FAILURE, "vxPaletteImgNode create failed!");
			vxReleaseKernel(&kernel);
		}
	}
	return node;
}
#endif
#endif

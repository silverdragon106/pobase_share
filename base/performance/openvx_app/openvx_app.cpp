#include "openvx_app.h"
#include "data_helper.h"
#include "base.h"
#include "logger/logger.h"

#ifdef POR_WITH_OVX
CBaseOpenVxApp::CBaseOpenVxApp()
{
	m_vx_context = NULL;
}

CBaseOpenVxApp::~CBaseOpenVxApp()
{
	exitOpenVX();
}

bool CBaseOpenVxApp::initOpenVX(const postring& openvx_kernel_dll)
{
	vx_status status = VX_SUCCESS;

	// create OpenVX context.
	m_vx_context = vxCreateContext();

	if ((status |= vxGetStatus((vx_reference)m_vx_context)) != VX_SUCCESS)
	{
		return false;
	}
	if ((status |= vxDirective((vx_reference)m_vx_context, VX_DIRECTIVE_ENABLE_PERFORMANCE)) != VX_SUCCESS)
	{
		return false;
	}

	// Register log callback
	vxRegisterLogCallback(m_vx_context, CBaseOpenVxApp::vxError, vx_true_e);
	vxAddLogEntry((vx_reference)NULL, VX_SUCCESS, "vxPublishKernels");

	// load kernels
	status |= vxLoadKernels(m_vx_context, openvx_kernel_dll.c_str());
	status |= CDataHelper::initInstance(m_vx_context);
	return (status == VX_SUCCESS);
}

bool CBaseOpenVxApp::exitOpenVX()
{
	if (m_vx_context != NULL)
	{
		vxReleaseContext(&m_vx_context);
	}
	return true;
}

void CBaseOpenVxApp::vxError(vx_context context, vx_reference ref, vx_status statux, const vx_char string[])
{
	printlog_lv2(QString("vxError: %1").arg(string));
}

#endif

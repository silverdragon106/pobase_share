#include "ovx_context.h"

#if defined(POR_WITH_OVX)
OvxContext::OvxContext()
{
	m_is_inited = false;
	m_context = NULL;
}

OvxContext::~OvxContext()
{
	destroy();
}

void OvxContext::printCaps()
{
}

bool OvxContext::create()
{
	if (!m_is_inited)
	{
		m_is_inited = true;
		m_context = vxCreateContext();

		vxDirective((vx_reference)m_context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
		vxRegisterLogCallback(m_context, onLogCallback, vx_true_e);
	}
	return (m_context != NULL);
}

bool OvxContext::destroy()
{
	if (m_is_inited)
	{
		m_is_inited = false;
		POVX_RELEASE(m_context);
	}
	return true;
}

void OvxContext::onLogCallback(vx_context, vx_reference, vx_status, const vx_char string[])
{
	printlog_lv1(QString("[vx_log]: %1").arg(string));
}
#endif
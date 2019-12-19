#pragma once
#include "ovx_base.h"
#include "ovx_object.h"

#if defined(POR_WITH_OVX)

class OvxContext : public OvxObject
{
public:
	OvxContext();
	virtual ~OvxContext();
	
	bool				create();
	bool				destroy();
	void				printCaps();
	
	/* typecast operators */
	operator vx_context() { return m_context; }

	/* methods */
	vx_context			getVxContext() { return m_context; }

	/* static methods */
	static void			onLogCallback(vx_context context, vx_reference ref, vx_status, const vx_char string[]);
	
private:
	bool				m_is_inited;
	vx_context			m_context;
};
#endif

#pragma once
#include "types.h"

#ifdef POR_WITH_OVX

#if !defined(POR_COMPILE_ON5728)
    #include "VX/vx.h"
    #include "VX/vxu.h"
#else
    #include "TI/tivx.h"
#endif

class CBaseOpenVxApp
{
public:
	CBaseOpenVxApp();
	~CBaseOpenVxApp();

	virtual bool				initOpenVX(const postring& openvx_kernel_dll);
	virtual bool				exitOpenVX();
		
	inline vx_context			getVxContext() { return m_vx_context; };

        static void VX_CALLBACK                 vxError(vx_context context, vx_reference ref, vx_status statux, const vx_char string[]);

public:
        vx_context                              m_vx_context;
};

#endif

#pragma once

#include "define.h"
#include "struct.h"
#include <QString>

class CPOBaseApp;
class CBaseAppUpdate
{
public:
	CBaseAppUpdate();
	virtual ~CBaseAppUpdate();

	void						initAppUpdate();
	CPOBaseApp*					getBaseApp();

	bool						isUpdateNow(i32 conn);
	bool						isUpdateReady(i32 conn);
	bool						checkUpdateReady(i32 conn);

	bool						deviceUpdate();
	bool						deviceUpdateInternal();
	void						deviceUpdateCancel();
	void						deviceUpdateConfirm(i32 confirm_delay_ms);
	i32							deviceUpdateStream(i32 conn, i32 pid, u8* buffer_ptr, i32& buffer_size);

public:
	UpdateInfo					m_app_update; //used temporary
};
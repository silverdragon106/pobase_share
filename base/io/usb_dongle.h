#pragma once

#include "define.h"

#if defined(POR_WITH_DONGLE)
extern "C"
{
#include "ss_lm_runtime.h"
}
#endif

#include <QThread>

//////////////////////////////////////////////////////////////////////////
// USB dongle license class.
// - using Elite5 library.
//////////////////////////////////////////////////////////////////////////
class CUSBDongle : public QThread
{
	Q_OBJECT
public:
	CUSBDongle();
	virtual ~CUSBDongle();

	bool						initInstance();
	bool						exitInstance();
	bool						logIn(i32 index, u32 lic_id);
	bool						logOut();

	bool						readData();
	bool						writeData();

	bool						encryptData();
	bool						decryptData();

	bool						isAvailable();
	postring					getGenAuthentication();

protected:
	void						run()	Q_DECL_OVERRIDE;

signals:
	void						licenseError();

public:
	bool						m_is_inited;
	std::atomic<bool>			m_is_thread_cancel;

#if defined(POR_WITH_DONGLE)
	SLM_HANDLE_INDEX			m_hslm;
#endif
	postring					m_auth_str_id;
	postring					m_auth_str_password;
};

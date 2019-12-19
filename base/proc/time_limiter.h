#pragma once
#include "define.h"

enum TimeLimitModeTypes
{
	kTimeLimitException = 0,
	kTimeLimitReturn = 1,

	kTimeLimitModeCount
};

class CTimeLimiter
{
public:
	CTimeLimiter(i32 mode);
	~CTimeLimiter();

	void					start(i32 timeout_ms);
	void					stop();

	bool					check();
	bool					test(i32& broken_ms);

	void					setEnabled(bool is_enabled);

private:
	bool					m_is_started;
	bool					m_is_timeout;
	bool					m_is_enabled;

	i32						m_timeout_mode;
	u64						m_timeout_ms;
	i32						m_broken_ms;
};

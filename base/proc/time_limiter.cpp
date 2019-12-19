#include "time_limiter.h"
#include "logger/logger.h"
#include "base.h"

CTimeLimiter::CTimeLimiter(i32 mode)
{
	m_is_started = false;
	m_is_timeout = false;
	m_is_enabled = true;

	m_timeout_mode = mode;
	m_timeout_ms = 0;
	m_broken_ms = 0;
}

CTimeLimiter::~CTimeLimiter()
{
}

void CTimeLimiter::start(i32 timeout_ms)
{
	if (m_is_enabled)
	{
		m_is_started = true;
		m_is_timeout = false;
		m_timeout_ms = sys_cur_time + timeout_ms;
		m_broken_ms = 0;
	}
}

void CTimeLimiter::stop()
{
	if (m_is_enabled)
	{
		m_is_started = false;
		m_is_timeout = false;
		m_timeout_ms = 0;
		m_broken_ms = 0;
	}
}

bool CTimeLimiter::check()
{
#if defined(POR_DEBUG)
	return true;
#endif

	if (!m_is_started || !m_is_enabled)
	{
		return true;
	}

	//check timeout
	u64 cur_time_ms = sys_cur_time;
	if (m_timeout_ms >= cur_time_ms)
	{
		return true;
	}

	//if check timeout after first-check, return timeout statue only
	if (m_is_timeout)
	{
		return false;
	}

	//in first check time after time-breaked
	m_broken_ms = cur_time_ms - m_timeout_ms;
	m_is_timeout = true;

	if (m_timeout_mode == kTimeLimitException)
	{
		throw (kTimeLimitException);
	}
	return false;
}

bool CTimeLimiter::test(i32& broken_ms)
{
	if (!m_is_started || !m_is_enabled)
	{
		return true;
	}

	broken_ms = m_broken_ms;
	return !m_is_timeout;
}

void CTimeLimiter::setEnabled(bool is_enabled)
{
	m_is_enabled = is_enabled;
}
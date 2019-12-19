#include "lock_guide.h"

//#ifdef USE_INHERIT_LOCK
CLockGuard::CLockGuard()
{
}

CLockGuard::CLockGuard(const CLockGuard& other)
{
}

CLockGuard::~CLockGuard()
{
}

void CLockGuard::operator=(const CLockGuard& other)
{
}

void CLockGuard::lock()
{
#if defined(USE_INHERIT_LOCK)
	m_mutex.lock();
#endif
}

void CLockGuard::unlock()
{
#if defined(USE_INHERIT_LOCK)
	m_mutex.unlock();
#endif
}

bool CLockGuard::trylock()
{
#if defined(USE_INHERIT_LOCK)
	return m_mutex.try_lock();
#else
	return true;
#endif
}
//#endif
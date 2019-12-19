#pragma once
#include <mutex>
#include "config.h"

#if defined(POR_DEVICE)
	#define USE_INHERIT_LOCK
#endif

#define mutex_guard(x)				std::lock_guard<std::mutex> lock(x);
#define exlock_guard(x)				std::lock_guard<std::recursive_mutex> lock(x);
#define exlock_guard_ptr(x)			std::lock_guard<std::recursive_mutex> lock(*x);

#if defined(USE_INHERIT_LOCK)
	#define lock_guard()			std::lock_guard<std::recursive_mutex> lock(m_mutex);
	#define anlock_guard(x)			std::lock_guard<std::recursive_mutex> anlock##x(x.m_mutex);
	#define anlock_guard_ptr(x)		std::lock_guard<std::recursive_mutex> anplock##x(x->m_mutex);
#else

	#define lock_guard()			
	#define anlock_guard(x)			
	#define anlock_guard_ptr(x)		
#endif

class CLockGuard
{
public:
	CLockGuard();
	CLockGuard(const CLockGuard& other);
	~CLockGuard();

	void							operator=(const CLockGuard& other);

	void							lock();
	void							unlock();
	bool							trylock();

public:
#if defined(USE_INHERIT_LOCK)
	std::recursive_mutex			m_mutex;
#endif
};

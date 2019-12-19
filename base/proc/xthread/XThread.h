#pragma once

#include "pthread.h"
#include <atomic>

class XThread
{
public:
	XThread();
	virtual ~XThread();

	void					ThreadStart();
	void					ThreadStop();
	bool					IsRunning();
	bool					HasToStop();

	void					OnThreadCallback();

protected:
	virtual void			ThreadRun() = 0;

	pthread_t				m_tid;
	std::atomic<bool>		m_bRunning;
	std::atomic<bool>		m_bThreadCancel;
};


#pragma once
#include "pthread.h"

class XMutex
{
public:
	XMutex();
	~XMutex();

	void		Lock();
	void		Unlock();
private:
	pthread_mutex_t m_mutex;
};


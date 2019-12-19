#include "XMutex.h"


XMutex::XMutex()
{
	pthread_mutex_init(&m_mutex, 0);
}

XMutex::~XMutex()
{
	if (m_mutex)
	{
		pthread_mutex_destroy(&m_mutex);
	}
}

void XMutex::Lock()
{
	if (m_mutex)
	{
		pthread_mutex_lock(&m_mutex);
	}
}

void XMutex::Unlock()
{
	if (m_mutex)
	{
		pthread_mutex_unlock(&m_mutex);
	}
}

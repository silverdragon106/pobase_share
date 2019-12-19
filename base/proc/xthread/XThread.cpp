#include "XThread.h"
#include "types.h"
#include <QThread>

#ifdef POR_WINDOWS
#pragma comment(lib, "pthreadVC2.lib")
#endif

void* thread_callback(void *param)
{
	((XThread*)param)->OnThreadCallback();
	return 0;
}

XThread::XThread()
{
	m_bRunning = false;
	m_bThreadCancel = false;

	memset(&m_tid, 0, sizeof(pthread_t));
}

XThread::~XThread()
{
	ThreadStop();
}

void XThread::OnThreadCallback()
{
	m_bRunning = true;
	ThreadRun();
	m_bRunning = false;
}

void XThread::ThreadStart()
{
	if (IsRunning())
	{
		return;
	}
	m_bThreadCancel = false;
	if (!pthread_create(&m_tid, NULL, thread_callback, this))
	{
	}
}

void XThread::ThreadStop()
{
	if (IsRunning())
	{
		m_bThreadCancel = true;
		while (IsRunning())
		{
			QThread::msleep(10);
		}
	}
	pthread_join(m_tid, NULL);
}

bool XThread::IsRunning()
{
	return m_bRunning;
}

bool XThread::HasToStop()
{
	return m_bThreadCancel;
}

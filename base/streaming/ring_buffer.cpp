#include "ring_buffer.h"
#include "base.h"
#include <QMutexLocker>
//#include <assert.h>

CRingBuffer::CRingBuffer()
{
	m_pStorage = NULL;
	m_nLen = 0;
	m_pStart = NULL;
	m_pEnd = NULL;
}

CRingBuffer::~CRingBuffer()
{
	Delete();
}

bool CRingBuffer::Create(int nMaxLen)
{
	m_pStorage = new u8[nMaxLen];
	m_nLen = nMaxLen;
	m_pStart = m_pStorage;
	m_pEnd = m_pStorage;

	return m_pStorage != NULL;
}

void CRingBuffer::Delete()
{
	QMutexLocker l(&m_mutex);

	m_nLen = 0;
	m_pStart = NULL;
	m_pEnd = NULL;

	m_pointers.clear();
	m_lengths.clear();
	POSAFE_DELETE_ARRAY(m_pStorage);
}

u8* CRingBuffer::Malloc(int nLen)
{
	if (!m_pStorage || nLen <= 0 || nLen > m_nLen)
		return NULL;

	u8* p = NULL;
	int llen = 0;
	int rlen = m_nLen;

	QMutexLocker l(&m_mutex);
	if (m_pStart <= m_pEnd)
	{
		llen = m_pStart - m_pStorage;
		rlen = m_nLen - (m_pEnd - m_pStorage);
	}
	else
	{
		rlen = m_pStart - m_pEnd;
	}
	// check remain size;
	if (rlen >= nLen)
	{
		p = m_pEnd;
		m_pEnd += nLen;
	}
	else if (nLen <= m_nLen)
	{
		if (nLen > llen)
		{
			simplelog("[Warning] Failed to malloc because of override to ringbuffer.");
		}
		else
		{
			p = m_pStorage;
			m_pEnd = m_pStorage + nLen;
		}
	}
	if (p)
	{
		m_pointers.append(p);
		m_lengths.append(nLen);
	}
	return p;
}

// No need to free.
u8* CRingBuffer::Malloc2(int nLen)
{
	if (!m_pStorage || nLen <= 0 || nLen > m_nLen)
		return NULL;

	u8* p = NULL;
	int llen = 0;
	int rlen = m_nLen;

	QMutexLocker l(&m_mutex);
	if (m_pStart <= m_pEnd)
	{
		llen = m_pStart - m_pStorage;
		rlen = m_nLen - (m_pEnd - m_pStorage);
	}
	else
	{
		rlen = m_pStart - m_pEnd;
	}
	// check remain size;
	if (rlen >= nLen)
	{
		p = m_pEnd;
		m_pStart = m_pEnd;
		m_pEnd += nLen;
	}
	else if (nLen <= m_nLen)
	{
		if (nLen > llen)
		{
			simplelog("[Warning] Failed to malloc because of override to ringbuffer.");
		}
		else
		{
			p = m_pStorage;
			m_pStart = m_pStorage;
			m_pEnd = m_pStorage + nLen;
		}
	}
	
	return p;
}

void CRingBuffer::Free2(u8* pBuf, int nLen)
{
	// check length.
	Free(pBuf);
}

void CRingBuffer::Free(u8* pBuf)
{
	QMutexLocker l(&m_mutex);
	int i = m_pointers.indexOf(pBuf);
	if (i != -1)
	{
		m_pointers.removeAt(i);
		m_lengths.removeAt(i);

		if (m_pointers.size() == 0)
		{
			m_pStart = m_pStorage;
			m_pEnd = m_pStorage;
		}
		else
		{
			m_pStart = m_pointers.first();
			m_pEnd = m_pointers.last() + m_lengths.last();
		}
	}
	if (m_pointers.size() > 0)
	{
		int a = 0;
	}
}

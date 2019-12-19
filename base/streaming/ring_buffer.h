#pragma once

#include "types.h"
#include <QMutex>
#include <QList>
//////////////////////////////////////////////////////////////////////////
// This class used under the first malloc memory is free first.
//////////////////////////////////////////////////////////////////////////
class CRingBuffer
{
public:
	CRingBuffer();
	~CRingBuffer();

	bool	Create(int nMaxLen);
	void	Delete();
	u8*		Malloc(int nLen);
	u8*		Malloc2(int nLen);

	void	Free2(u8* pBuf, int nLen);
	void	Free(u8* pBuf);
public:
	u8*						m_pStorage;
	int						m_nLen;
	u8*						m_pStart;
	u8*						m_pEnd;

	QList<u8*>				m_pointers;
	QList<int>				m_lengths;
	QMutex					m_mutex;
};


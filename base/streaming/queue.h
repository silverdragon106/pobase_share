#pragma once
#include "types.h"
#include <QMutex>
#include <QWaitCondition>
#include <QList>
#include "logger.h"

template<class T>
class CQueue
{
	const int kMaxQueueSize = 20;
public:
	typedef QList<T> Container;

	CQueue() :m_nWaitingReaders(0){}

	/* thread-safe functions */
	int				getSize();
	bool			put(const T& d);
	bool			pop(T* p, uint nTime = 0);
	bool			get(T* p, uint nTime = 0);
	void			remove(T p);
	void			clear();

	void			lock();
	void			unlock();

	/* thread-unsafe functions */
	Container&		getBuffer() { return m_buffer; }

public:
	QMutex			m_mutex;
	QWaitCondition	m_bufferIsNotEmpty;
	Container		m_buffer;
	short			m_nWaitingReaders;
};

#include "queue-inl.h"

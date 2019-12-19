#pragma once

template<class T>
int CQueue<T>::getSize()
{
	QMutexLocker locker(&m_mutex);
	return m_buffer.size();
}

template<class T>
bool CQueue<T>::put(const T& d)
{
	bool b = false;
	QMutexLocker locker(&m_mutex);
	if (m_buffer.size() < kMaxQueueSize)
	{
		m_buffer.push_back(d);
		b = true;
	}
	if (m_nWaitingReaders)
		m_bufferIsNotEmpty.wakeOne();
	return b;
}

template<class T>
bool CQueue<T>::pop(T* p, uint nTime)
{
	QMutexLocker locker(&m_mutex);
	bool isInQueue = false;

	isInQueue = m_buffer.size() > 0 ? true : false;
	if (!isInQueue && nTime)
	{
		m_nWaitingReaders++;
		isInQueue = m_bufferIsNotEmpty.wait(&m_mutex, nTime);
		m_nWaitingReaders--;
	}

	if (m_buffer.size() > 0)
	{
		T d = m_buffer.front();
		m_buffer.pop_front();
		*p = d;
		return true;
	}
	return false;
}

template<class T>
bool CQueue<T>::get(T* p, uint nTime) 
{
	QMutexLocker locker(&m_mutex);
	bool isInQueue = false;

	isInQueue = m_buffer.size() > 0 ? true : false;
	if (!isInQueue && nTime)
	{
		m_nWaitingReaders++;
		isInQueue = m_bufferIsNotEmpty.wait(&m_mutex, nTime);
		m_nWaitingReaders--;
	}

	if (m_buffer.size() > 0)
	{
		T d = m_buffer.front();
		*p = d;
		return true;
	}
	return false;
}

template<class T>
void CQueue<T>::remove(T p) 
{
	QMutexLocker locker(&m_mutex);
	m_buffer.removeOne(p);
}

template<class T>
void CQueue<T>::clear() 
{
	QMutexLocker locker(&m_mutex);
	m_buffer.clear();
}

template<class T>
void CQueue<T>::lock()
{
	m_mutex.lock();
}

template<class T>
void CQueue<T>::unlock()
{
	m_mutex.unlock();
}

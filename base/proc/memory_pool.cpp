#include "memory_pool.h"
#include "base.h"

CPOMemPool::CPOMemPool()
{
	m_pool_size = 0;
	m_pool_data_ptr = NULL;
	m_pool_buffer_ptr = NULL;
}

CPOMemPool::~CPOMemPool()
{
	freeBuffer();
}

void CPOMemPool::popBuffer(void* pointer)
{
	if (m_pool_size <= 0)
	{
		return;
	}

	//check pointer range
	if (pointer < m_pool_data_ptr &&
		CPOBase::checkIndex((u8*)pointer - m_pool_buffer_ptr, m_pool_size))
	{
		m_pool_data_ptr = (u8*)pointer;
	}
}

void CPOMemPool::initBuffer(i32 size)
{
	if (size <= m_pool_size)
	{
		releaseBuffer();
		return;
	}

	freeBuffer();
	if (CPOBase::isPositive(size))
	{
		m_pool_buffer_ptr = po_new u8[size];
		m_pool_data_ptr = m_pool_buffer_ptr;
		m_pool_size = size;
	}
}

void CPOMemPool::mallocBuffer(i32 size)
{
	if (size > m_pool_size)
	{
		POSAFE_DELETE_ARRAY(m_pool_buffer_ptr);
		m_pool_buffer_ptr = po_new u8[size];
		m_pool_data_ptr = m_pool_buffer_ptr;
		m_pool_size = size;
	}
}

bool CPOMemPool::extendBuffer(i32 size)
{
	//check new buffer size
	if (!CPOBase::isPositive(size))
	{
		return false;
	}
	
	i32 used_size = m_pool_data_ptr - m_pool_buffer_ptr;
	u8* buffer_ptr = po_new u8[used_size + size];

	//copy current memory data
	if (m_pool_buffer_ptr != m_pool_data_ptr)
	{
		CPOBase::memCopy(buffer_ptr, m_pool_buffer_ptr, used_size);
		POSAFE_DELETE_ARRAY(m_pool_buffer_ptr);
	}

	//set new buffer
	m_pool_buffer_ptr = buffer_ptr;
	m_pool_data_ptr = m_pool_buffer_ptr + used_size;
	m_pool_size = used_size + size;
	return true;
}

void CPOMemPool::freeBuffer()
{
	m_pool_size = 0;
	m_pool_data_ptr = NULL;
	POSAFE_DELETE_ARRAY(m_pool_buffer_ptr);
}

void CPOMemPool::releaseBuffer()
{
	m_pool_data_ptr = m_pool_buffer_ptr;
}

bool CPOMemPool::isAvailableBuffer(i32 size)
{
	if (!CPOBase::isPositive(m_pool_size))
	{
		return false;
	}
	return (m_pool_buffer_ptr - m_pool_data_ptr + m_pool_size) >= size;
}

#pragma once
#include "struct.h"

#define PO_MEMPOOL_SIZE		20480000 //about 40MB

class CPOMemPool
{
public:
	CPOMemPool();
	virtual ~CPOMemPool();

	void					initBuffer(i32 size = PO_MEMPOOL_SIZE);
	void					freeBuffer();
	void					releaseBuffer();

	bool					isAvailableBuffer(i32 size);

	void					mallocBuffer(i32 size);
	bool					extendBuffer(i32 size);
	void					popBuffer(void* pointer);
	
private:
	i32						m_pool_size;
	u8*						m_pool_data_ptr;
	u8*						m_pool_buffer_ptr;

public:
	template <typename T> 
	void getBuffer(T*& buffer_ptr, i32 size, bool use_malloc = false)
	{
		buffer_ptr = NULL;
		i32 buffer_size = sizeof(T)*size;

		if (!isAvailableBuffer(buffer_size))
		{
			if (!use_malloc || !CPOMemPool::extendBuffer(buffer_size))
			{
				return;
			}
		}
		buffer_ptr = (T*)m_pool_data_ptr;
		m_pool_data_ptr += buffer_size;
	}

	template <typename T>
	void getZeroBuffer(T*& buffer_ptr, i32 size, bool use_malloc = false)
	{
		getBuffer(buffer_ptr, size, use_malloc);
		if (buffer_ptr)
		{
			memset(buffer_ptr, 0, size*sizeof(T));
		}
	}
};

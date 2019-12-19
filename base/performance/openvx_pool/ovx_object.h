#pragma once
#include "ovx_base.h"

#define OVX_GRAPH_CACHED_MS		10000
#define OVX_RESOURCE_CACHED_MS	5000
#define OVX_RESOURCE_MAX_RANGE	1.2f

/*
* OvxObject class 
*	is base class of all Ovx_* classes.
*/
class OvxObject
{
public:
	OvxObject();
	virtual ~OvxObject();

	virtual void	addRef();
	virtual void	decRef();

	static void		printGlobalObjectStats();

	inline i32		getObjectID() const { return m_object_id; };

private:
	void			generateObject();

public:
	i32				m_ref;
	i32				m_object_id;
};

/* the count of current living OvxObject instances */
extern std::atomic<i32> g_object_count;
/* the unique id of OvxObject instances */
extern std::atomic<i32> g_object_global_id;

///*
//* OvxObjectRef
//*	is reference count class for all Ovx_* classes.
//*/
//template <class T>
//class OvxRef
//{
//public:
//	/* null constructor */
//    OvxRef()                    {m_data=NULL; }
//	/* initialize constructor */
//	OvxRef(T* data);
//	/* copy constructor */
//    OvxRef(const OvxRef& obj);
//    template <class U>
//    OvxRef(const OvxRef<U>& obj);
//	/* destructor */
//	~OvxRef();
//	
//	/* value copy operator */
//    /* Note: can copy one type to another type. So it is a reason of serious but.
//     *  When use this operator, make sure that the code is valid.
//    */
//    template <class U>
//    OvxRef&	operator=(const OvxRef<U>& obj);
//    OvxRef&	operator=(const OvxRef& obj);
//
//    /* data access operator */
//	inline T*		getData() const { return m_data; };
//	inline bool		isNull() const { return m_data == NULL; };
//
//	inline T*		operator->() { return m_data; };
//	inline T&		operator*() { return *m_data; };
//
//	inline void		addRef() { if (m_data) { ((OvxObject*)m_data)->addRef(); } };
//	inline void		decRef() { if (m_data) { ((OvxObject*)m_data)->decRef(); m_data = NULL; } };
//
//public:
//	T *				m_data;
//};
//
///*
//* OvxObjectRef class definition.
//*/
//template <class T>
//OvxRef<T>::OvxRef(T* data)
//{
//	m_data = data;
//	addRef();
//}
//
//template <class T>
//template <class U>
//OvxRef<T>::OvxRef(const OvxRef<U>& obj)
//{
//	/* force to typecast to Call AddRef. */
//	//decRef();
//    ((OvxRef<U>*)&obj)->addRef();
//
//	/* copy data */
//	this->m_data = (T*)obj.m_data;
//}
//
//template <class T>
//OvxRef<T>::OvxRef(const OvxRef<T>& obj)
//{
//    /* force to typecast to Call AddRef. */
//    ((OvxRef<T>*)&obj)->addRef();
//
//    /* copy data */
//    this->m_data = obj.m_data;
//}
//
//template <class T>
//OvxRef<T>::~OvxRef()
//{
//	decRef();
//}
//
//template <class T>
//template <class U>
//OvxRef<T>& OvxRef<T>::operator=(const OvxRef<U>& obj)
//{
//	/* force to typecast to Call AddRef. */
//	decRef();
//    ((OvxRef<U>*)&obj)->addRef();
//
//	/* copy data */
//	this->m_data = obj.m_data;
//
//    return *this;
//}
//
//template <class T>
//OvxRef<T>& OvxRef<T>::operator=(const OvxRef<T>& obj)
//{
//    /* force to typecast to Call AddRef. */
//	decRef();
//    ((OvxRef<T>*)&obj)->addRef();
//
//    /* copy data */
//    this->m_data = obj.m_data;
//
//    return *this;
//}
//
//template <class T1, class U>
//bool operator==(const OvxRef<T1>& ref1, const OvxRef<U>& ref2)
//{
//    return (ref1.getData() == ref2.getData());
//}
//
//template <class T>
//bool operator==(const OvxRef<T>& ref1, const OvxRef<T>& ref2)
//{
//    return (ref1.getData() == ref2.getData());
//}

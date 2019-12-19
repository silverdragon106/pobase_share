#include "ovx_object.h"

std::atomic<i32> g_object_count(0);
std::atomic<i32> g_object_global_id(0);

OvxObject::OvxObject()
{
	m_ref = 0;
	g_object_count++;
	generateObject();
}

OvxObject::~OvxObject()
{
	g_object_count--;
}

void OvxObject::printGlobalObjectStats()
{
	printlog_lvs2(QString("Object Count: %1").arg((i32)g_object_count), LOG_SCOPE_OVX);
	printlog_lvs2(QString("Next Object ID: %1").arg((i32)g_object_global_id), LOG_SCOPE_OVX);
}

void OvxObject::generateObject()
{
	g_object_global_id++;
	m_object_id = g_object_global_id;
}

void OvxObject::addRef()
{
	m_ref++;
}

void OvxObject::decRef()
{
	--m_ref;
	if (m_ref <= 0)
	{
		delete this;
	}
}

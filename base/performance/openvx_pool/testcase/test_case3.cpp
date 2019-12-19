/**
* 시험경우3:
*	설명: ResourcePool 및 GraphPool 검사.
*	목적: ResourcePool 및 GraphPool 을 리용하여 자원의 재사용경우를 검사한다.
*	- 스레드1에서 ResourcePool을 리용하여 자원(화상,스칼라,배렬)을 창조한다.
*	- 스레드1에서 ResourcePool을 리용하여 가상자원(화상,스칼라,배렬)을 창조한다.
*	- 스레드1에서 GraphPool을 리용하여 그라프를 창조한다.
*	- 스레드1에서 GraphPool을 리용하여 그라프를 해방한다.
*	- 스레드2에서 GraphPool을 리용하여 그라프를 창조한다.
*	- 스레드2에서 GraphPool을 리용하여 그라프를 실행한다.
*	- 스레드2에서 GraphPool을 리용하여 그라프를 해방한다.
*/

#include <QApplication>
#include <QObject>
#include <QThread>
#include "base.h"
#include "ovx_object.h"
#include "ovx_context.h"
#include "ovx_graph.h"
#include "ovx_node.h"
#include "ovx_graph_pool.h"
#include "ovx_resource_pool.h"
#include "ovx_lock_manager.h"
#include "test_case3.h"

namespace test_case3
{
	OvxContextRef context;
	OvxGraphRef graph1;
	OvxGraphRef graph2;
	OvxGraphPool graph_pool;
    OvxResourcePoolRef resource_pool;

	/*
	* Definition of Task1 class.
	*/
	void Task1::doWork()
	{
        QThread::sleep(1);
        /* created and process graph1 */
		ovxLock("graph1");
		graph1 = graph_pool.fetchGraph(OvxGraphPool::Graph1);

		graph1->verify();
		graph1->process();
		ovxUnlock("graph1");

		ovxSignal("graph1_was_processed");

        /* delete graph1 on Task2. see Task2 */

		/* create graph2 */
		ovxLock("graph2");
		graph2 = graph_pool.fetchGraph(OvxGraphPool::Graph2);
		ovxUnlock("graph2");

		ovxSignal("graph2_was_created");
		/* process graph2 on Task2. seek Task2 */
		ovxWait("graph2_was_processed");

		/* delete graph2 */
		ovxLock("graph2");
		graph_pool.freeGraph(graph2);
		ovxUnlock("graph2");

		ovxSignal("graph2_was_deleted");

		emit finished();
	}

	void Task1::startOnThread(QThread* thread)
	{
		this->moveToThread(thread);
		thread->start();

		connect(thread, SIGNAL(started()), this, SLOT(doWork()));
		connect(this, SIGNAL(finished()), thread, SLOT(quit()));

		connect(thread, SIGNAL(finished()), this, SLOT(deleteLater()));
		connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	}

	/*
	* Definition of Task2 class.
	*/

	void Task2::doWork()
	{
		/* graph1 is created on Task1. see Task1 */
		ovxWait("graph1_was_processed");

		/* delete graph1 */
		ovxLock("graph1");
		graph_pool.freeGraph(graph1);
		ovxUnlock("graph1");

		ovxSignal("graph1_was_deleted");

		/* graph2 is created on Task1. see Task1*/
		ovxWait("graph2_was_created");
		{
			/* process graph2 */
			OvxLocker locker("graph2");
			graph2->verify();
			graph2->process();
		}

		ovxSignal("graph2_was_processed");

		/* delete on Task1. see Task1 */
		ovxWait("graph2_was_deleted");

		emit finished();
	}

	void Task2::startOnThread(QThread* thread)
	{
		this->moveToThread(thread);
		thread->start();

		connect(thread, SIGNAL(started()), this, SLOT(doWork()));
		connect(this, SIGNAL(finished()), thread, SLOT(quit()));

		connect(thread, SIGNAL(finished()), this, SLOT(deleteLater()));
		connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	}
}

extern QApplication* g_app;
void testCase3()
{
    test_case3::context = OvxContextRef(new OvxContext());
    test_case3::resource_pool = OvxResourcePoolRef(new OvxResourcePool());

    test_case3::graph_pool.create(test_case3::context, test_case3::resource_pool);

    test_case3::Task1* task1 = new test_case3::Task1();
    test_case3::Task2* task2 = new test_case3::Task2();

    QThread *thread1 = new QThread();
    QThread *thread2 = new QThread();

    task2->startOnThread(thread2);
    task1->startOnThread(thread1);


    QObject::connect(thread2, SIGNAL(finished()), g_app, SLOT(quit()));

    g_app->exec();

    test_case3::graph_pool.clear();
    test_case3::resource_pool->clear();
    test_case3::context = OvxContextRef();
}

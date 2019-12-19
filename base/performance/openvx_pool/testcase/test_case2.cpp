/**
* 시험경우2:
*	설명: 그라프 및 노드의 다중스레드 창조, 해방 및 실행
*	목적: 그라프 및 노드의 다중스레드 창조, 해방 및 실행이 원할하게 진행되는가를 검사하는것이다.
*	- 스레드1에서 그라프1을 창조한다.
*	- 스레드2에서 그라프1을 해방한다.
*	- 스레드1에서 그라프2을 창조한다.
*	- 스레드2에서 그라프2를 실행한다.
*	- 스레드1에서 그라프2를 해방한다.
*	- 성능검사를 진행한다.
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
#include "test_case2.h"

namespace test_case2
{
	OvxContextRef context;
	OvxGraphRef graph1;
	OvxGraphRef graph2;
	OvxResourcePool resource_pool;

    static const int width = 1280;
    static const int height = 1024;

	/*
	* Definition of Task1 class.
	*/
	void Task1::doWork()
	{
        QThread::sleep(1);
		/* graph 1 */
		ovxLock("graph1");

		graph1 = OvxGraphRef(new OvxGraph(context, "Graph1"));

		vx_image image1 = resource_pool.fetchImage(context, width, height, VX_DF_IMAGE_U8);
		vx_image image2 = resource_pool.fetchImage(context, width, height, VX_DF_IMAGE_U8);

        graph1->addNode(vxGaussian3x3Node((vx_graph)(*graph1), image1, image2), "GaussianNode");
		graph1->verify();
		graph1->process();
		ovxUnlock("graph1");

		ovxSignal("graph1_was_processed");
		/* delete graph1 on Task2. see Task2 */


		/* graph 2 */
		ovxLock("graph2");
        /* resource pool test */
        resource_pool.fetchImage(context, width, height, VX_DF_IMAGE_U8);

		graph2 = OvxGraphRef(new OvxGraph(context, "Graph2"));
        graph2->addNode(vxGaussian3x3Node((vx_graph)(*graph2), image1, image2), "GaussianNode");
		ovxUnlock("graph2");

		ovxSignal("graph2_was_created");

		/* process graph2 on Task2. see Task2 */

		/* delete graph2 */
        ovxWait("graph2_was_processed");

		ovxLock("graph2");
		graph2 = OvxGraphRef();
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
		/* graph1 is created on Task1 */
		ovxWait("graph1_was_processed");

		/* delete graph1 */
		ovxLock("graph1");
        if (graph1->isFinished())
        {
            graph1 = OvxGraphRef();
        }
		ovxUnlock("graph1");

		ovxSignal("graph1_was_deleted");

		/* graph2 is created on Task1*/
		ovxWait("graph2_was_created");

		/* process graph2 */
		ovxLock("graph2");
		graph2->verify();
		graph2->process();
		ovxUnlock("graph2");

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
void testCase2()
{
    test_case2::context = OvxContextRef(new OvxContext());

    test_case2::Task1* task1 = new test_case2::Task1();
    test_case2::Task2* task2 = new test_case2::Task2();

    QThread *thread1 = new QThread();
    QThread *thread2 = new QThread();

    task2->startOnThread(thread2);
    task1->startOnThread(thread1);

    QObject::connect(thread2, SIGNAL(finished()), g_app, SLOT(quit()));

    g_app->exec();
}

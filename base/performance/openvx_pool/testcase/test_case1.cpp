/**
* 시험경우1:
*	설명: 그라프 및 노드의 동적 창조 및 해방
*	목적: 그라프 및 노드, 자원의 창조 및 해방시 메모리 릭을 검사하는것이다.
*	- 노드창조
*	- 노드해방
*	- 그라프창조
*	- 그라프해방
*	- 그라프를 실행한다.
*	- 성능검사를 진행한다.
*/
#include "base.h"
#include "ovx_object.h"
#include "ovx_context.h"
#include "ovx_graph.h"
#include "ovx_node.h"
#include "ovx_resource_pool.h"

namespace test_case1
{
	const int width = 1280;
	const int height = 1024;

}

void testCase1()
{
    OvxResourcePool resource_pool;
    OvxContextRef context(new OvxContext());
    OvxGraph graph1(context, "Graph1");

    vx_image img1 = resource_pool.fetchImage(context, test_case1::width, test_case1::height, VX_DF_IMAGE_U8);
    vx_image img2 = resource_pool.fetchImage(context, test_case1::width, test_case1::height, VX_DF_IMAGE_U8);

    graph1.addNode(vxGaussian3x3Node(graph1.getVxGraph(), img1, img2), "vxGaussianNode");
    graph1.verify();
    graph1.process();
    graph1.finished();

    graph1.printPerf();
}

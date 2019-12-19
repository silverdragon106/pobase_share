#include <QtCore/QCoreApplication>

#include "ovx_base.h"
#include "ovx_types.h"
#include "ovx_context.h"
#include "ovx_graph.h"
#include "ovx_graph_pool.h"
#include "ovx_resource_pool.h"

/*
* Cusomize Graph
****************************** 2018-11-12
*/
class MyGraph : public OvxGraph
{
public:
	MyGraph(const OvxContextRef& context, const postring& name) : OvxGraph(context, name) {
		_input = NULL;
		_output = NULL;
		_output_data = NULL;
	}
	void init(OvxResourcePoolRef resource_pool, int width, int height) {
		vx_image image1 = resource_pool->fetchImage(this, width, height, VX_DF_IMAGE_U8);
		vx_image image2 = resource_pool->fetchImage(this, width, height, VX_DF_IMAGE_U8);
		addNode(vxGaussian3x3Node(getVxGraph(), image1, image2), "GaussianNode");

		_resource_pool = resource_pool;
		_input = image1;
		_output = image2;
	}
	void prepare(u8* img_data, int width, int height) {
		OvxHelper::writeImage(_input, img_data, width, height, 1);
	}
	void setOutput(u8* out_data, int width, int height) {
		_output_data = out_data;
	}
	void finish() {
		OvxHelper::readImage(_output_data, NULL, NULL, _output);
	}

	OvxResourcePoolRef _resource_pool;
	vx_image _input;
	vx_image _output;
	int _width;
	int _height;

	u8* _output_data;
};
/**
* MyGraphPool
*/
class MyGraphPool : public OvxGraphPool
{
public:
	static const int width = 1024;
	static const int height = 1024;
	static const int format = VX_DF_IMAGE_U8;
public:
	virtual	OvxGraphRef createGraph(int graph_type) {
		switch (graph_type)
		{
		case 0:
		{
			OvxGraphRef graph(new OvxGraph(getContext(), "Graph 1"));
			vx_image image1 = getResourcePool()->fetchImage(graph, width, height, format);
			vx_image image2 = getResourcePool()->fetchImage(graph, width, height, format);
			graph->addNode(vxGaussian3x3Node(graph->getVxGraph(), image1, image2), "GaussianNode");

			return graph;
		}
		case 1:
		{
			OvxGraphRef graph(new OvxGraph(getContext(), "Graph 2"));
			vx_image image1 = getResourcePool()->fetchImage(graph, width + 1, height + 1, format);
			vx_image image2 = getResourcePool()->fetchImage(graph, width + 1, height + 1, format);
			graph->addNode(vxGaussian3x3Node(graph->getVxGraph(), image1, image2), "GaussianNode");

			return graph;
		}
		/*
		****************************** 2018 - 11 - 12
		*/
		case 3: 
		{
			MyGraph* graph = new MyGraph(getContext(), "Graph 3");
			graph->init(getResourcePool(), width, height);

			/* 
			* 사용실례:
			*/
			/*
			u8* out_data = new u8[width*height];
			u8* in_data = new_u8[width*height];

			graph->setOutput(data, width, height);
			graph->prepare(u8, width, height);
			graph->process();
			graph->finished();

			// at this point, you can manipulate on out_data.
			// out_data is result data from MyGraph.
			*/

			return OvxGraphRef(graph);
		}
		}

		return OvxGraphRef();
	}
};

/**
* OvxTestApp
*/

class OvxTestApp
{
public:
	void	init() {
		_context = OvxContextRef(new OvxContext());
		_resource_pool = OvxResourcePoolRef(new OvxResourcePool());
		_graph_pool = OvxGraphPoolRef(new MyGraphPool());

		_resource_pool->create(10);
		_graph_pool->create(_context, _resource_pool, 4);
	}
	void	destroy() {
		_resource_pool = OvxResourcePoolRef();
		_graph_pool = OvxGraphPoolRef();
		_context = OvxContextRef();
	}

	void	run() {
		/*
		- GraphPool을 통한 그라프의 창조, 해방을 시험한다.
		*/
		{
			OvxGraphRef graph1 = _graph_pool->fetchGraph(0);
			_graph_pool->freeGraph(graph1);

			OvxGraphRef graph2 = _graph_pool->fetchGraph(1);
			_graph_pool->freeGraph(graph2);
		}
		/*
		- GraphPool을 통한 그라프의 재사용성을 시험한다.
		*/
		{
			OvxGraphRef graph1 = _graph_pool->fetchGraph(0);

			graph1->verify();
			graph1->process();
			graph1->finish();

			_graph_pool->freeGraph(graph1);

			graph1 = _graph_pool->fetchGraph(0);
			graph1->verify();		/*여기서는 이미 verify가 되여있었기때문에 시간이 소비하지 않고 그냥 성공할것이다. */
			graph1->process();
			graph1->finish();

			_graph_pool->freeGraph(graph1);
		}

		/*
		- GraphPool과 ResourcePool사이에 련동을 하면서 
		  그라프가 Pool에서 완전히 삭제되였을때 그라프가 사용하던 Resource들을 Free상태로 정확히 만들어지는가를 시험한다.
		*/
		{
			OvxGraphRef graph1 = _graph_pool->fetchGraph(0);
			OvxGraphRef graph2 = _graph_pool->fetchGraph(0);
			/* 
			- GraphPool의 QueueSize를 2개로 설정하고 실행하였기때문에 
			  그라프 2개 창조는 성공할것이다. 
			  기대출력값은 Free Graph Count : 1, Using Graph Count : 2
			*/
			printlog_lv3(" Expected Result");
			printlog_lv3("    Free Graph Count: 1");
			printlog_lv3("    Using Graph Count: 2");
			_graph_pool->printStats();

			/*
			- GraphPool의 QueueSize가 2이기떄문에 그라프창조오유가 나올것이다.
			  기대출력값은 "Failed to create graph. GraphPool Queue Size[2] is Full.
			*/
			_graph_pool->fetchGraph(0);
			_graph_pool->fetchGraph(0);
			_graph_pool->fetchGraph(1);

			_graph_pool->freeGraph(graph1);
			_graph_pool->freeGraph(graph2);

			/*
			- GraphPool의 QueueSize를 2개로 설정하고 실행하였기때문에
			그라프 2개 창조는 성공할것이다.
			기대출력값은 Free Resource Count : 2, Using Resource Count : -
			*/
			printlog_lv3(" Expected Result");
			printlog_lv3("    Free Resource Count: 2");
			printlog_lv3("    Using Resource Count: --");
			_resource_pool->printStats();

			graph1 = _graph_pool->fetchGraph(1);
			graph2 = _graph_pool->fetchGraph(1);

			_graph_pool->freeGraph(graph1);

			/*
			- GraphPool의 QueueSize를 4개로 설정하고 실행하였기때문에
			  그라프 2개 창조는 성공할것이다.
			  기대출력값은 Free Graph Count : 2, Using Graph Count : 1
			*/
			printlog_lv3(" Expected Result");
			printlog_lv3("    Free Graph Count: 1");
			printlog_lv3("    Using Graph Count: 3");
			_graph_pool->printStats();
		}
	}
public:
	OvxContextRef		_context;
	OvxGraphPoolRef		_graph_pool;
	OvxResourcePoolRef	_resource_pool;

};


/************************************************************************/
/* main                                                                 */
/************************************************************************/
int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	g_debug_logger.initInstance(PO_LOG_FILENAME, kLogModeDirect);
	g_debug_logger.setLogLevel(LOG_LV4);

	OvxObject::printGlobalObjectStats();
	{
		OvxTestApp test_app;

		test_app.init();
		test_app.run();
		test_app.destroy();
	}
	OvxObject::printGlobalObjectStats();

	return 0;
	//return a.exec();
}

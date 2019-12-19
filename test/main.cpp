#include "ssim_processor.h"
#include "proc/image_proc.h"
#include "logger/logger.h"

#ifdef POR_DEVICE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

int main(int argc, char *argv[])
{
	SSIMMap ssim_map1;
	SSIMMap ssim_map2;
	i32 w1, h1, w2, h2;
	u8* img_ptr1 = CImageProc::loadImgOpenCV("d:\\1.bmp", w1, h1);
	u8* img_ptr2 = CImageProc::loadImgOpenCV("d:\\2.bmp", w2, h2);

	CSSIMProcessor ssim_proc;
	g_debug_logger.initInstance("log.txt");
	g_time_logger.initInstance("log_time.txt");

	for (i32 i = 0; i < 10; i++)
	{
		keep_time(1);
		ssim_proc.makeSSIMMap(ssim_map1, img_ptr1, w1, h1);
		leave_time(1);
		
		keep_time(2);
		ssim_proc.makeSSIMMap(ssim_map2, img_ptr2, w2, h2);
		leave_time(2);

		keep_time(3);
		f32 similar = ssim_proc.checkStructSimilar(ssim_map1, ssim_map2, w1, h1);
		leave_time(3);
	}
	
	printlog_lv1(QString("make ssim map1(%1ms), make ssim map2(%2ms), similar(%3ms)")
						.arg(tm_time(1)->avg_tick_time)
						.arg(tm_time(2)->avg_tick_time)
						.arg(tm_time(3)->avg_tick_time));
	return 1;
}

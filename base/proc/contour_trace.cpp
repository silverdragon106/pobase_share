#include "contour_trace.h"
#include "base.h"
#include "proc/image_proc.h"

CContourTrace::CContourTrace()
{
}

CContourTrace::~CContourTrace()
{
}

i32 CContourTrace::getContourPixel(u8* edge_img_ptr, i32 w, i32 h, i32 fg_pixel, i32 pixel_count, 
								i32 min_contour_pixel, ContouruVector& contour_vec)
{
	i32 i, wh = w*h;
	u8 *tmp_edge_ptr = edge_img_ptr;
	POSAFE_CLEAR(contour_vec);

	/* 에지화상의 화소수를 계수한다. */
	if (pixel_count == 0)
	{
		for (i = 0; i < wh; i++, tmp_edge_ptr++)
		{
			if (*tmp_edge_ptr == fg_pixel)
			{
				continue;
			}
			pixel_count++;
		}
	}
	if (pixel_count <= 0)
	{
		return 0;
	}
	
	/* 초기화 및 패딩 */
	u8 back_pixel = 127;
	Pixelu* tmp_pixel_ptr = po_new Pixelu[pixel_count];

	i32 w1 = w + 2;
	i32 h1 = h + 2;
	u8* pad_img_ptr = po_new u8[w1*h1];
	CImageProc::makePaddingImage(pad_img_ptr, edge_img_ptr, w, h, back_pixel);

	/* 륜곽선을 추출한다. */
	i32 x, y;
	i32 found_pixel = 0;
	u8* tmp_pad_ptr = pad_img_ptr;

	for (y = 0; y < h1; y++)
	{
		for (x = 0; x < w1; x++)
		{
			if (*tmp_pad_ptr == fg_pixel)
			{
				Contouru contour(tmp_pixel_ptr, pixel_count);
				traceContour(pad_img_ptr, w1, x, y, fg_pixel, back_pixel, kSearchModeBack, &contour);
				if (!contour.isClosedContour())
				{
					traceContour(pad_img_ptr, w1, x, y, fg_pixel, back_pixel, kSearchModeFore, &contour);
					contour.m_is_closed = false;
				}

				if (contour.getContourPixelNum() >= min_contour_pixel)
				{
					found_pixel += contour.getContourPixelNum();
					Contouru* contour_ptr = CPOBase::pushBackNew(contour_vec);
					contour_ptr->setValue(contour);
				}
			}
			tmp_pad_ptr++;
		}
	}

	POSAFE_DELETE_ARRAY(tmp_pixel_ptr);
	POSAFE_DELETE_ARRAY(pad_img_ptr);
	return found_pixel;
}

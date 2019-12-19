#include <math.h>
#include "run_table.h"
#include "base.h"

#if defined(POR_WITH_LOG)
#include "performance/openvx_pool/ovx_graph_pool.h"
#endif

CImgRunTable::CImgRunTable()
{
	memset(this, 0, sizeof(CImgRunTable));
}

CImgRunTable::~CImgRunTable()
{
	freeBuffer();
}

i32 CImgRunTable::memSize()
{
	i32 len = 0;
	len += sizeof(m_pixels);
	len += sizeof(m_width);
	len += sizeof(m_height);
	len += sizeof(m_run_count);

	if (m_width > 0 && m_height > 0 && m_run_count > 0)
	{
		len += sizeof(i32)*(m_height + 1);
		len += sizeof(u16)*m_run_count;
	}
	return len;
}

i32 CImgRunTable::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_pixels, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_width, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_height, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_run_count, buffer_ptr, buffer_size);

	if (m_width > 0 && m_height > 0 && m_run_count > 0)
	{
		CPOBase::memWrite(m_pxy_ptr, m_height + 1, buffer_ptr, buffer_size);
		CPOBase::memWrite(m_run2_ptr, m_run_count, buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

i32 CImgRunTable::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	i32 nw = 0, nh = 0;
	i32 count = 0, npixels = 0;
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(npixels, buffer_ptr, buffer_size);
	CPOBase::memRead(nw, buffer_ptr, buffer_size);
	CPOBase::memRead(nh, buffer_ptr, buffer_size);
	CPOBase::memRead(count, buffer_ptr, buffer_size);
		
	if (CPOBase::isPositive(nw) && CPOBase::isPositive(nh) && CPOBase::isPositive(count))
	{
		checkBuffer(nw, nh);
		m_run_count = count; m_pixels = npixels;

		CPOBase::memRead(m_pxy_ptr, m_height + 1, buffer_ptr, buffer_size);
		CPOBase::memRead(m_run2_ptr, m_run_count, buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

bool CImgRunTable::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	i32 nw, nh;
	i32 count, npixels;
	CPOBase::fileRead(npixels, fp);
	CPOBase::fileRead(nw, fp);
	CPOBase::fileRead(nh, fp);
	CPOBase::fileRead(count, fp);

	if (CPOBase::isPositive(nw) && CPOBase::isPositive(nh) && CPOBase::isPositive(count))
	{
		checkBuffer(nw, nh);
		m_run_count = count;
		m_pixels = npixels;

		CPOBase::fileRead(m_pxy_ptr, m_height + 1, fp);
		CPOBase::fileRead(m_run2_ptr, m_run_count, fp);
	}
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CImgRunTable::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_pixels, fp);
	CPOBase::fileWrite(m_width, fp);
	CPOBase::fileWrite(m_height, fp);
	CPOBase::fileWrite(m_run_count, fp);

	if (CPOBase::isPositive(m_width) && CPOBase::isPositive(m_height) && CPOBase::isPositive(m_run_count))
	{
		CPOBase::fileWrite(m_pxy_ptr, m_height+1, fp);
		CPOBase::fileWrite(m_run2_ptr, m_run_count, fp);
	}
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

void CImgRunTable::setValue(CImgRunTable* run_table_ptr)
{
	freeBuffer();
	CPOBase::memCopy(this, run_table_ptr);

	m_pxy_ptr = po_new i32[m_height + 1];
	m_run2_ptr = po_new u16[m_run_count];
	CPOBase::memCopy(m_pxy_ptr, run_table_ptr->m_pxy_ptr, m_height + 1);
	CPOBase::memCopy(m_run2_ptr, run_table_ptr->m_run2_ptr, m_run_count);
}

void CImgRunTable::initBuffer(i32 nw, i32 nh)
{
	m_width = nw;
	m_height = nh;
	m_pixels = 0;
	m_pxy_ptr = po_new i32[nh + 1];
	m_run2_ptr = po_new u16[(nw + 2)*nh];
	memset(m_pxy_ptr, 0, sizeof(i32)*(nh + 1));
}

void CImgRunTable::freeBuffer()
{
	m_width = 0;
	m_height = 0;
	m_pixels = 0;
	m_run_count = 0;
	POSAFE_DELETE_ARRAY(m_pxy_ptr);
	POSAFE_DELETE_ARRAY(m_run2_ptr);
}

void CImgRunTable::checkBuffer(i32 nw, i32 nh)
{
	if (nh <= m_height && nw*nh <= m_width*m_height)
	{
		m_width = nw; m_height = nh; m_pixels = 0;
		return;
	}
	freeBuffer();
	initBuffer(nw, nh);
}

i32* CImgRunTable::getPxyTable()
{
	return m_pxy_ptr;
}

u16* CImgRunTable::getRunTable()
{
	return m_run2_ptr;
}

i32 CImgRunTable::getArraySize()
{
	i32 len = 0;
	len += sizeof(m_pixels);
	len += sizeof(m_run_count);
	len += sizeof(i32) * (m_height + 1);
	len += sizeof(u16) * m_run_count;
	return len;
}

i32 CImgRunTable::getMaxArraySize()
{
	i32 len = 0;
	len += sizeof(m_pixels);
	len += sizeof(m_run_count);
	len += sizeof(i32)*(m_height + 1);
	len += sizeof(u16)*(m_width + 2)*m_height;
	return len;
}

bool CImgRunTable::makeRunTable(u8* img_ptr, i32 nw, i32 nh, bool vx_method)
{
	if (img_ptr == NULL || nw*nh <= 0)
	{
		return false;
	}
	
	buildStart(nw, nh);
	bool is_processed = false;
	
#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr && vx_method)
	{
		CGImgProcCvtImg2RunTable* graph_ptr = (CGImgProcCvtImg2RunTable*)g_vx_gpool_ptr->fetchGraph(
				kGImgProcCvtImg2RunTable, img_ptr, nw, nh, this, kRunTableFlagNone, -1, NULL);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#endif
	if (!is_processed)
	{
		i32 x, y;
		u8* tmp_img_ptr = img_ptr;
		for (y = 0; y < nh; y++)
		{
			for (x = 0; x < nw; x++, tmp_img_ptr++)
			{
				if (*tmp_img_ptr > 0)
				{
					updateNewPixel(x);
				}
				else
				{
					updateFreePixel();
				}
			}
			updateNewLine(y);
		}
	}
	buildStop();
	return true;
}

bool CImgRunTable::makeInvRunTable(u8* img_ptr, i32 nw, i32 nh, bool vx_method)
{
	if (img_ptr == NULL || nw*nh <= 0)
	{
		return false;
	}

	buildStart(nw, nh);
	bool is_processed = false;

#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr && vx_method)
	{
		CGImgProcCvtImg2RunTable* graph_ptr = (CGImgProcCvtImg2RunTable*)g_vx_gpool_ptr->fetchGraph(
				kGImgProcCvtImg2RunTable, img_ptr, nw, nh, this, kRunTableFlagInvert, -1, NULL);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#endif
	if (!is_processed)
	{
		i32 x, y;
		u8* tmp_img_ptr = img_ptr;

		for (y = 0; y < nh; y++)
		{
			for (x = 0; x < nw; x++, tmp_img_ptr++)
			{
				if (*tmp_img_ptr == 0)
				{
					updateNewPixel(x);
				}
				else
				{
					updateFreePixel();
				}
			}
			updateNewLine(y);
		}
	}
	buildStop();
	return true;
}

bool CImgRunTable::makeRunTable(u8* img_ptr, u8* mask_img_ptr, i32 nw, i32 nh, bool vx_method)
{
	if (img_ptr == NULL || mask_img_ptr == NULL || nw*nh <= 0)
	{
		return false;
	}

	buildStart(nw, nh);
	bool is_processed = false;

#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr && vx_method)
	{
		CGImgProcCvtImg2RunTable* graph_ptr = (CGImgProcCvtImg2RunTable*)g_vx_gpool_ptr->fetchGraph(
				kGImgProcCvtImg2RunTable, img_ptr, nw, nh, this, kRunTableFlagMaskImg, -1, mask_img_ptr);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#endif
	if (!is_processed)
	{
		i32 x, y;
		u8* tmp_img_ptr = img_ptr;
		u8* tmp_mask_img_ptr = mask_img_ptr;

		for (y = 0; y < nh; y++)
		{
			for (x = 0; x < nw; x++, tmp_img_ptr++, tmp_mask_img_ptr++)
			{
				if (*tmp_img_ptr > 0 && *tmp_mask_img_ptr > 0)
				{
					updateNewPixel(x);
				}
				else
				{
					updateFreePixel();
				}
			}
			updateNewLine(y);
		}
	}
	buildStop();
	return true;
}

bool CImgRunTable::makeRunTable(u8* img_ptr, i32 nw, i32 nh, i32 val, bool vx_method)
{
	if (img_ptr == NULL || nw*nh <= 0)
	{
		return false;
	}

	buildStart(nw, nh);
	bool is_processed = false;

#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr && vx_method)
	{
		CGImgProcCvtImg2RunTable* graph_ptr = (CGImgProcCvtImg2RunTable*)g_vx_gpool_ptr->fetchGraph(
				kGImgProcCvtImg2RunTable, img_ptr, nw, nh, this, kRunTableFlagMaskVal, val, NULL);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#endif
	if (!is_processed)
	{
		i32 x, y;
		u8* tmp_img_ptr = img_ptr;

		for (y = 0; y < nh; y++)
		{
			for (x = 0; x < nw; x++, tmp_img_ptr++)
			{
				if ((*tmp_img_ptr & val) > 0)
				{
					updateNewPixel(x);
				}
				else
				{
					updateFreePixel();
				}
			}
			updateNewLine(y);
		}
	}
	buildStop();
	return true;
}

bool CImgRunTable::makeImageFromRun(u8* img_ptr, i32 nw, i32 nh, i32 val, bool vx_method)
{
	if (nw != m_width && nh != m_height)
	{
		return false;
	}

	bool is_processed = false;
#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr && vx_method)
	{
		CGImgProcCvtRunTable2Img* graph_ptr = (CGImgProcCvtRunTable2Img*)
				g_vx_gpool_ptr->fetchGraph(kGImgProcCvtRunTable2Img, this, img_ptr, val);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#endif
	if (!is_processed)
	{
		i32 k, y, st_pos, ed_pos;
		u8* scan_imgy_ptr;
		for (y = 0; y < nh; y++)
		{
			scan_imgy_ptr = img_ptr + y*nw;
			st_pos = m_pxy_ptr[y];
			ed_pos = m_pxy_ptr[y + 1];
			for (k = st_pos; k < ed_pos; k += 2)
			{
				memset(scan_imgy_ptr + m_run2_ptr[k], val, m_run2_ptr[k + 1]);
			}
		}
	}
	return true;
}

u8* CImgRunTable::makeImageFromRun(i32& nw, i32& nh, i32 val, bool vx_method)
{
	nw = m_width;
	nh = m_height;
	if (nw*nh <= 0)
	{
		return NULL;
	}

	u8* img_ptr = po_new u8[nw*nh];
	memset(img_ptr, 0, nw*nh);
	if (!makeImageFromRun(img_ptr, nw, nh, val, vx_method))
	{
		POSAFE_DELETE_ARRAY(img_ptr);
		return NULL;
	}
	return img_ptr;
}

CImgRunTable& CImgRunTable::operator=(const CImgRunTable& robj)
{
	checkBuffer(robj.m_width, robj.m_height);
	m_width = robj.m_width;
	m_height = robj.m_height;
	m_run_count = robj.m_run_count;
	m_pixels = robj.m_pixels;
	
	if (robj.m_pxy_ptr != NULL && robj.m_run2_ptr != NULL)
	{
		CPOBase::memCopy(m_pxy_ptr, robj.m_pxy_ptr, m_height + 1);
		CPOBase::memCopy(m_run2_ptr, robj.m_run2_ptr, m_run_count);
	}
	return *this;
}

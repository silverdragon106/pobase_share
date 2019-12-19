#include "emu_samples.h"
#include "base.h"
#include "os/qt_base.h"
#include "proc/image_proc.h"
#include "network/packet.h"
#include "struct/camera_setting.h"

const QString kEmuSamplePattern = QString("sample");

#define EMU_BASE_WIDTH		1280
#define EMU_BASE_HEIGHT		1024

//////////////////////////////////////////////////////////////////////////
CEmuSamples::CEmuSamples()
{

}

CEmuSamples::~CEmuSamples()
{

}

i32 CEmuSamples::memSizeThumb()
{
	return 0;
}

i32 CEmuSamples::memWriteThumb(u8*& buffer_ptr, i32& buffer_size)
{
	return 0;
}

i32 CEmuSamples::getEmuSampleCams()
{
	return 0;
}

i32 CEmuSamples::getEmuTriggerMode()
{
	return kCamTriggerContinuous;
}

i32 CEmuSamples::getEmuTriggerInterval()
{
	return 0;
}

void CEmuSamples::getEmuMaxSampleSize(i32& max_width, i32& max_height)
{
}

bool CEmuSamples::getEmuSmapleGrayImage(i32 cam_id, i32& index, bool force_snap,
							u8* gray_buffer_ptr, i32& capture_width, i32& capture_height)
{
	return false;
}

bool CEmuSamples::getEmuSmapleImage(i32 cam_id, i32& index, bool force_snap,
							u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel)
{
	return false;
}

//////////////////////////////////////////////////////////////////////////
COneEmuSamples::COneEmuSamples(i32 max_width, i32 max_height)
{
	m_max_width = max_width;
	m_max_height = max_height;
	m_base_width = EMU_BASE_WIDTH;
	m_base_height = EMU_BASE_HEIGHT;

	m_emu_changed = false;
	m_emu_image.freeBuffer();
}

COneEmuSamples::~COneEmuSamples()
{
	freeEmuSample();
}

i32 COneEmuSamples::getEmuSampleCams()
{
	return 1;
}

i32 COneEmuSamples::getEmuTriggerMode()
{
	return kCamTriggerCamera;
}

i32 COneEmuSamples::getEmuTriggerInterval()
{
	return 1;
}

void COneEmuSamples::getEmuMaxSampleSize(i32& max_width, i32& max_height)
{
	max_width = m_max_width; 
	max_height = m_max_height;
}

bool COneEmuSamples::getEmuSmapleGrayImage(i32 cam_id, i32& index, bool force_snap,
							u8* gray_buffer_ptr, i32& capture_width, i32& capture_height)
{
	if (!gray_buffer_ptr)
	{
		return false;
	}

	lock_guard();
	if ((!m_emu_changed && !force_snap) || !m_emu_image.isValid())
	{
		return false;
	}

	i32 capture_channel;
	m_emu_image.copyToImage(gray_buffer_ptr, capture_width, capture_height, capture_channel, 1);
	m_emu_changed = false;
	return true;
}

bool COneEmuSamples::getEmuSmapleImage(i32 cam_id, i32& index, bool force_snap,
							u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel)
{
	if (!raw_buffer_ptr)
	{
		return false;
	}

	lock_guard();
	if ((!m_emu_changed && !force_snap) || !m_emu_image.isValid())
	{
		return false;
	}

	m_emu_image.copyToImage(raw_buffer_ptr, capture_width, capture_height, capture_channel);
	m_emu_changed = false;
	return true;
}

bool COneEmuSamples::setEmuSample(ImageData& img_data)
{
	if (!img_data.isValid() || m_max_width * m_max_height <= 0)
	{
		return false;
	}

	freeEmuSample();

	lock_guard();
	m_emu_changed = true;
	m_emu_image.copyImage(img_data, m_base_width, m_base_height);
	m_emu_image.cropImage(m_max_width, m_max_height);
	return true;
}

void COneEmuSamples::freeEmuSample()
{
	lock_guard();
	m_emu_changed = false;
	m_emu_image.freeBuffer();
}

//////////////////////////////////////////////////////////////////////////
CLocalEmuSamples::CLocalEmuSamples()
{
	m_is_inited = false;
	m_is_played = false;
	m_is_thumb_generated = false;

	m_sample_max_width = 0;
	m_sample_max_height = 0;
	m_sample_interval = 1000;

	m_sample_cam_count = 0;
	memset(m_sample_count, 0, sizeof(m_sample_count));
	memset(m_sample_index, 0, sizeof(m_sample_index));
	memset(m_sample_ptr, 0, sizeof(m_sample_count));
	memset(m_sample_thumb_ptr, 0, sizeof(m_sample_index));
}

CLocalEmuSamples::~CLocalEmuSamples()
{
	freeSamples();
}

bool CLocalEmuSamples::setEmuSelected(i32 cam_index, i32 selected, bool need_stop)
{
	if (need_stop)
	{
		stop();
	}

	if (!CPOBase::checkIndex(cam_index, m_sample_cam_count))
	{
		return false;
	}
	if (!CPOBase::checkIndex(selected, m_sample_count[cam_index]))
	{
		return false;
	}

	m_sample_index[cam_index] = selected;
	return true;
}

bool CLocalEmuSamples::setEmuTriggerInterval(i32 interval)
{
	m_sample_interval = interval;
	return true;
}

i32 CLocalEmuSamples::getEmuTriggerMode()
{
	return kCamTriggerContinuous;
}

i32 CLocalEmuSamples::getEmuTriggerInterval()
{
	return m_sample_interval;
}

i32 CLocalEmuSamples::getEmuSampleCams()
{
	return m_sample_cam_count;
}

i32 CLocalEmuSamples::getEmuSampleCount(i32 cam_index)
{
	if (!CPOBase::checkIndex(cam_index, PO_CAM_COUNT))
	{
		return 0;
	}
	return m_sample_count[cam_index];
}

i32 CLocalEmuSamples::getEmuSelected(i32 cam_index)
{
	if (!CPOBase::checkIndex(cam_index, m_sample_cam_count))
	{
		return -1;
	}
	return m_sample_index[cam_index];
}

bool CLocalEmuSamples::loadSamples(const QString& source_path, bool is_thumb_generated)
{
	if (!m_is_inited)
	{
		freeSamples();

		if (!loadSamples(source_path, kEmuSamplePattern, m_sample_ptr, m_sample_count, m_sample_index, m_sample_cam_count))
		{
			printlog_lvs2("There is nothing sample image for emulator.", LOG_SCOPE_CAM);
		}
		if (is_thumb_generated)
		{
			buildThumbSamples(m_sample_ptr, m_sample_cam_count, m_sample_count, m_sample_thumb_ptr);
		}
		m_is_inited = true;
	}
	return true;
}

bool CLocalEmuSamples::setSamples(Packet* packet_ptr, bool is_thumb_generated)
{
	freeSamples();
	if (!packet_ptr)
	{
		return false;
	}

	u8* buffer_ptr = packet_ptr->getData();
	i32 buffer_size = packet_ptr->getDataLen();
	if (!buffer_ptr || buffer_size <= 0)
	{
		return false;
	}

	CPOBase::memRead(m_sample_cam_count, buffer_ptr, buffer_size);
	if (!CPOBase::isCount(m_sample_cam_count))
	{
		return false;
	}

	i32 i, j, w, h, count = -1;
	u8vector data_vec;

	for (i = 0; i < m_sample_cam_count; i++)
	{
		CPOBase::memRead(count, buffer_ptr, buffer_size);
		if (!CPOBase::isCount(count))
		{
			freeSamples();
			return false;
		}

		if (count <= 0)
		{
			continue;
		}

		m_sample_count[i] = count;
		m_sample_ptr[i] = po_new ImageData[count];

		for (j = 0; j < count;j++)
		{
			//load image
			CPOBase::memReadVector(data_vec, buffer_ptr, buffer_size);
#if defined(POR_SUPPORT_COLOR)
			cv::Mat img = CImageProc::decodeImgOpenCV(data_vec, cv::IMREAD_ANYCOLOR);
#else
			cv::Mat img = CImageProc::decodeImgOpenCV(data_vec, cv::IMREAD_GRAYSCALE);
#endif

			//check sample width and height
			if (img.cols <= 0 || img.rows <= 0)
			{
				freeSamples();
				return false;
			}

			w = img.cols;
			h = img.rows;
			m_sample_max_width = po::_max(m_sample_max_width, w);
			m_sample_max_height = po::_max(m_sample_max_height, h);
			m_sample_ptr[i][j].copyImage(img.data, w, h, img.channels());
		}
	}

	if (is_thumb_generated)
	{
		buildThumbSamples(m_sample_ptr, m_sample_cam_count, m_sample_count, m_sample_thumb_ptr);
	}
	return true;
}

bool CLocalEmuSamples::writeSamples(const QString& data_path)
{
	if (m_sample_cam_count <= 0)
	{
		return false;
	}

	QTBase::clearContents(data_path);

	i32 i, j;
	QString filename;
	ImageData* img_data_ptr;
	
	for (i = 0; i < m_sample_cam_count; i++)
	{
		img_data_ptr = m_sample_ptr[i];
		if (!CPOBase::isCount(m_sample_count[i]) || img_data_ptr == NULL)
		{
			continue;
		}
		
		for (j = 0; j < m_sample_count[i]; j++)
		{
			filename = data_path + QString("/%1%2_%3.bmp").arg(kEmuSamplePattern).arg(i).arg(j);
			CImageProc::saveImgOpenCV(filename.toStdTString(), img_data_ptr[j]);
		}
	}
	return true;
}

bool CLocalEmuSamples::loadSamples(const QString& source_path, const QString& sample_pattern,
							ImageData** sample_ptr, i32* sample_count_ptr, i32* sample_index_ptr, i32& sample_cam)
{
	//get sample count per each camera
	i32 i, j;
	sample_cam = po::_min(QTBase::getFileCount(source_path, QString("%1*_0.bmp").arg(sample_pattern)), PO_CAM_COUNT);
	if (sample_cam <= 0)
	{
		return false;
	}

	for (i = 0; i < sample_cam; i++)
	{
		sample_count_ptr[i] = QTBase::getFileCount(source_path, QString("%1%2_*.bmp").arg(sample_pattern).arg(i));
	}

	//load all samples
	i32 w, h;
	QString filename;

	for (i = 0; i < sample_cam; i++)
	{
		w = 0;
		h = 0;
		sample_ptr[i] = po_new ImageData[sample_count_ptr[i]];

		for (j = 0; j < sample_count_ptr[i]; j++)
		{
			//load image
			filename = source_path + QString("/%1%2_%3.bmp").arg(sample_pattern).arg(i).arg(j);
#if defined(POR_SUPPORT_COLOR)
			cv::Mat img = CImageProc::loadImgOpenCV(filename.toStdTString(), cv::IMREAD_ANYCOLOR);
#else
			cv::Mat img = CImageProc::loadImgOpenCV(filename.toStdTString(), cv::IMREAD_GRAYSCALE);
#endif

			//check sample width and height
			if (img.cols <= 0 || img.rows <= 0)
			{
				freeSamples();
				return false;
			}

			w = img.cols;
			h = img.rows;
			
			m_sample_max_width = po::_max(m_sample_max_width, w);
			m_sample_max_height = po::_max(m_sample_max_height, h);
			m_sample_ptr[i][j].copyImage(img.data, w, h, img.channels());
		}
	}

	return true;
}

void CLocalEmuSamples::freeSamples()
{
	m_is_inited = false;
	m_sample_max_width = 0;
	m_sample_max_height = 0;

	freeSamples(m_sample_ptr, m_sample_thumb_ptr, m_sample_count, m_sample_index, m_sample_cam_count);
}

void CLocalEmuSamples::freeSampleThumbs()
{
	freeSampleThumbs(m_sample_thumb_ptr, m_sample_count, m_sample_cam_count);
}

void CLocalEmuSamples::freeSamples(ImageData** sample_ptr, ImageData** sample_thumb_ptr, i32* sample_num_ptr, i32* sample_index_ptr, i32& sample_cam)
{
	if (!sample_ptr && !sample_thumb_ptr)
	{
		return;
	}

	//free all samples
	i32 i, j;
	if (sample_ptr)
	{
		for (i = 0; i < sample_cam; i++)
		{
			if (!sample_ptr[i])
			{
				continue;
			}

			for (j = 0; j < sample_num_ptr[i]; j++)
			{
				sample_ptr[i][j].freeBuffer();
			}
			POSAFE_DELETE_ARRAY(sample_ptr[i]);
		}
	}

	//free sample thumb
	if (sample_thumb_ptr)
	{
		for (i = 0; i < sample_cam; i++)
		{
			if (!sample_thumb_ptr[i])
			{
				continue;
			}

			for (j = 0; j < sample_num_ptr[i]; j++)
			{
				sample_thumb_ptr[i][j].freeBuffer();
			}
			POSAFE_DELETE_ARRAY(sample_thumb_ptr[i]);
		}
	}

	//free sample info
	sample_cam = 0;
	memset(sample_num_ptr, 0, sizeof(i32)*PO_CAM_COUNT);
	memset(sample_index_ptr, 0, sizeof(i32)*PO_CAM_COUNT);
	memset(sample_ptr, 0, sizeof(void*)*PO_CAM_COUNT);
	memset(sample_thumb_ptr, 0, sizeof(void*)*PO_CAM_COUNT);
}

void CLocalEmuSamples::freeSampleThumbs(ImageData** sample_thumb_ptr, i32* sample_num_ptr, i32 sample_cam)
{
	if (!sample_thumb_ptr)
	{
		return;
	}

	//free all samples
	i32 i, j;
	for (i = 0; i < sample_cam; i++)
	{
		for (j = 0; j < sample_num_ptr[i]; j++)
		{
			sample_thumb_ptr[i][j].freeBuffer();
		}
		POSAFE_DELETE_ARRAY(sample_thumb_ptr[i]);
	}

	//free sample info
	memset(sample_thumb_ptr, 0, sizeof(void*)*PO_CAM_COUNT);
}

bool CLocalEmuSamples::getEmuSmapleGrayImage(i32 cam_index, i32& index, bool force_snap,
								u8* gray_buffer_ptr, i32& capture_width, i32& capture_height)
{
	if (!m_is_inited || !gray_buffer_ptr ||
		!CPOBase::checkIndex(cam_index, m_sample_cam_count))
	{
		return false;
	}
	if (m_sample_count[cam_index] <= 0)
	{
		return false;
	}

	if (m_is_played || force_snap)
	{
		m_sample_index[cam_index] = (m_sample_index[cam_index] + 1) % m_sample_count[cam_index];
	}

	i32 capture_channel;
	index = m_sample_index[cam_index];
	m_sample_ptr[cam_index]->copyToImage(gray_buffer_ptr, capture_width, capture_height, capture_channel, 1);
	return true;
}

bool CLocalEmuSamples::getEmuSmapleImage(i32 cam_index, i32& index, bool force_snap,
								u8* raw_buffer_ptr, i32& capture_width, i32& capture_height, i32& capture_channel)
{
	if (!m_is_inited || !raw_buffer_ptr ||
		!CPOBase::checkIndex(cam_index, m_sample_cam_count))
	{
		return false;
	}
	if (m_sample_count[cam_index] <= 0)
	{
		return false;
	}
	
	if (m_is_played || force_snap)
	{
		m_sample_index[cam_index] = (m_sample_index[cam_index] + 1) % m_sample_count[cam_index];
	}

	index = m_sample_index[cam_index];
	m_sample_ptr[cam_index]->copyToImage(raw_buffer_ptr, capture_width, capture_height, capture_channel);
	return true;
}

void CLocalEmuSamples::getEmuSampleInfo(i32 cam, i32& w, i32& h)
{
	w = 0;
	h = 0;
	if (!m_is_inited || !CPOBase::checkIndex(cam, m_sample_cam_count))
	{
		return;
	}
	w = m_sample_ptr[cam]->w;
	h = m_sample_ptr[cam]->h;
}

void CLocalEmuSamples::getEmuMaxSampleSize(i32& max_width, i32& max_heght)
{
	max_width = m_sample_max_width;
	max_heght = m_sample_max_height;
}

void CLocalEmuSamples::play()
{
	m_is_played = true;
}

void CLocalEmuSamples::stop()
{
	m_is_played = false;
}

void CLocalEmuSamples::nextFrame(i32 cam_index)
{
	if (!m_is_inited || !CPOBase::checkIndex(cam_index, m_sample_cam_count))
	{
		return;
	}
	m_sample_index[cam_index] = (m_sample_index[cam_index] + 1) % m_sample_count[cam_index];
}

i32 CLocalEmuSamples::memSizeThumb()
{
	i32 i, j, count, len = 0;
	ImgExpr* img_expr_ptr = NULL;

	len += sizeof(m_sample_cam_count);
	for (i = 0; i < m_sample_cam_count; i++)
	{
		count = m_sample_count[i];
		len += sizeof(count);
		if (!CPOBase::isCount(count) || m_sample_thumb_ptr[i] == NULL)
		{
			continue;
		}

		for (j = 0; j < count; j++)
		{
			len += m_sample_ptr[i][j].memImgSize();
		}
	}
	return len;
}

i32 CLocalEmuSamples::memWriteThumb(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memWrite(m_sample_cam_count, buffer_ptr, buffer_size);
	
	u8* img_ptr = NULL;
	ImageData* img_data_ptr = NULL;
	i32 i, j, count;

	for (i = 0; i < m_sample_cam_count; i++)
	{
		count = m_sample_count[i];
		if (!CPOBase::isCount(count) || m_sample_thumb_ptr[i] == NULL)
		{
			count = 0;
			CPOBase::memWrite(count, buffer_ptr, buffer_size);
			continue;
		}

		CPOBase::memWrite(count, buffer_ptr, buffer_size);
		for (j = 0; j < count; j++)
		{
			m_sample_ptr[i][j].memImgWrite(buffer_ptr, buffer_size);
		}
	}
	return buffer_ptr - buffer_pos;
}

void CLocalEmuSamples::buildThumbSamples(ImageData** sample_ptr, i32 sample_cam_count, i32* sample_count_ptr, ImageData** sample_thumb_ptr)
{
	if (!sample_ptr || !sample_thumb_ptr || !sample_count_ptr || !CPOBase::isCount(sample_cam_count))
	{
		return;
	}

	freeSampleThumbs();

	i32 i, j;
	for (i = 0; i < m_sample_cam_count; i++)
	{
		if (!sample_ptr[i])
		{
			continue;
		}

		m_sample_thumb_ptr[i] = po_new ImageData[m_sample_count[i]];
		for (j = 0; j < m_sample_count[j]; j++)
		{
			CImageProc::makeThumbImage(m_sample_ptr[i][j], m_sample_thumb_ptr[i][j]);
		}
	}
}

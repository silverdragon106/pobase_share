#include "plain_serial.h"
#include "struct.h"
#include "base.h"
#include "log_config.h"

using namespace serial;

CPlainTextSerial::CPlainTextSerial()
{
	m_dev_param_ptr = NULL;

	m_is_inited = false;
	m_is_thread_cancel = false;
	m_is_reopen_device = false;
	
	m_write_buffer_ptr = po_new u8[POIO_READSIZE];
	m_write_buffer_size = 0;
	connect(this, SIGNAL(reopenedDevice()), this, SLOT(onReOpenedDevice()), Qt::QueuedConnection);
}

CPlainTextSerial::~CPlainTextSerial()
{
	POSAFE_DELETE_ARRAY(m_write_buffer_ptr);
}

bool CPlainTextSerial::initInstance(CPlainDevParam* dev_param_ptr)
{
	ERR_PREPARE(0);
	ERR_PREPARE(1);
	if (!dev_param_ptr || m_is_inited)
	{
		return false;
	}

	m_dev_param_ptr = dev_param_ptr;
	m_is_reopen_device = false;
	{
		exlock_guard(m_write_mutex);
		m_write_buffer_size = 0;
	}

#if defined(POR_WITH_IOMODULE)
	m_is_thread_cancel = false;
	QThreadStart();
#endif

	m_is_inited = true;
	return true;
}

void CPlainTextSerial::exitInstance()
{
	if (m_is_inited)
	{
		m_is_inited = false;
		
#if defined(POR_WITH_IOMODULE)
		m_is_thread_cancel = true;
		QThreadStop();
#endif
		{
			exlock_guard(m_write_mutex);
			m_write_buffer_size = 0;
		}
	}
}

bool CPlainTextSerial::openDevice(serial::Serial* port, const CPlainDevParam& dev_param)
{
	if (!port)
	{
		printlog_lvs3("Can't open comport(invalid param)", LOG_SCOPE_COMM);
		return false;
	}

	closeDevice(port);
	singlelog_lvs4(QString("Open comport[%1]").arg(dev_param.m_port_name.c_str()), LOG_SCOPE_COMM);

	//dev_param: thread_safe
	port->setPort(dev_param.m_port_name.c_str());
	port->setRSMode((rsmode_t)dev_param.m_rs_mode);
	port->setBaudrate(dev_param.m_baud_rate);
    port->setBytesize((bytesize_t)dev_param.m_data_bits);
	port->setParity((parity_t)dev_param.m_parity);
	port->setStopbits((stopbits_t)dev_param.m_stop_bits);
	port->setFlowcontrol((flowcontrol_t)dev_param.m_flow_control);

	port->open();
	if (!port->isOpen())
	{
        printlog_lvs3("Can't open comport(internal error).", LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tPort:%1").arg(dev_param.m_port_name.c_str()), LOG_SCOPE_COMM);
		printlog_lvs3(QString("\tRSMode:%1").arg(dev_param.m_rs_mode), LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tBaudrate:%1").arg(dev_param.m_baud_rate), LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tBytesize:%1").arg(dev_param.m_data_bits), LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tParity:%1").arg(dev_param.m_parity), LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tStopbits:%1").arg(dev_param.m_stop_bits), LOG_SCOPE_COMM);
        printlog_lvs3(QString("\tFlowcontrol:%1").arg(dev_param.m_flow_control), LOG_SCOPE_COMM);
		return false;
	}

	port->setTimeout(serial::Timeout::simpleTimeout((uint32_t)1, (uint32_t)2000));
	return true;
}

void CPlainTextSerial::closeDevice(Serial* port)
{
	if (port->isOpen())
	{
		port->close();
	}
	port->clearError();
}

void CPlainTextSerial::run()
{
	singlelog_lv0("The PlainSerial thread is");
	CPlainDevParam plain_dev = m_dev_param_ptr->getValue();

	serial::Serial serial_port;
	openDevice(&serial_port, plain_dev);

	i32 last_len, write_buffer_size = 0;
	i32 read_bytes, max_read_bytes;
	u8* read_buffer_ptr = po_new u8[POIO_READSIZE];
	u8* write_buffer_ptr = po_new u8[POIO_READSIZE];
	u8* buffer_ptr = read_buffer_ptr;
	u8* last_buffer_ptr = read_buffer_ptr;
	bool is_retry = false;

	while (!m_is_thread_cancel)
	{
		if (m_is_reopen_device)
		{
			plain_dev = m_dev_param_ptr->getValue();
			if (openDevice(&serial_port, plain_dev))
			{
				m_is_reopen_device = false;
				emit reopenedDevice();
			}
			else
			{
				QThread::msleep(2000);
				continue;
			}
		}

		if (!serial_port.isOpen() || serial_port.errorNumber() == serial::serialerror::resource_error)
		{
			ERR_OCCUR2(1, printlog_lvs3(QString("Com port can't opened or resource error(%1) [%2]")
											.arg(serial_port.errorNumber()).arg(_err_rep1), LOG_SCOPE_COMM),
			{
				postring serial_port;
				serial_port = plain_dev.getSerialPortName();
				alarmlog2(kPOErrIOModuleOpen, QString("%1,%2").arg(serial_port.c_str()).arg(_err_rep1));

				emit deviceError(kPODescIOManager, kPODevErrConnect, _err_rep1);
			});

			m_is_reopen_device = true;
			serial_port.clearError();
			QThread::msleep(2000);
			continue;
		}
		ERR_UNOCCUR(1);

		//read command
		max_read_bytes = POIO_READSIZE - (buffer_ptr - read_buffer_ptr);
		read_bytes = (i32)serial_port.read((u8*)buffer_ptr, max_read_bytes);
		if (read_bytes > 0)
		{
			buffer_ptr += read_bytes;
			onReadData(last_buffer_ptr, (i32)(buffer_ptr - last_buffer_ptr));

			//[start] pReadBuffer--[readed packet_ptr]--pLastBuffer--[reading packet_ptr]--pBuffer--[end]
			last_len = buffer_ptr - last_buffer_ptr;
			if (last_len >= POIO_HALFREADSIZE) //max packet size
			{
				last_buffer_ptr = buffer_ptr - POIO_SMALLREADSIZE; //reserved readed size
				last_len = POIO_SMALLREADSIZE;
			}
			if (last_buffer_ptr - read_buffer_ptr >= POIO_HALFREADSIZE) //check start packe position
			{
				CPOBase::memCopy(read_buffer_ptr, last_buffer_ptr, last_len);
				last_buffer_ptr = read_buffer_ptr;
				buffer_ptr = read_buffer_ptr + last_len;
				printlog_lv1("PlainSerial ReadBuffer was swaped.");
			}
		}

		//write output
		{
			exlock_guard(m_write_mutex);
			if (m_write_buffer_size > 0)
			{
				CPOBase::memCopy(write_buffer_ptr, m_write_buffer_ptr, m_write_buffer_size);
				write_buffer_size = m_write_buffer_size;
				m_write_buffer_size = 0;
			}
		}
		if (write_buffer_size > 0)
		{
			if (serial_port.write((const uint8_t*)write_buffer_ptr, write_buffer_size) == 0)
			{
				printlog_lvs2(QString("PlainSerial port writing failed, errcode: %1")
								.arg(serial_port.errorNumber()), LOG_SCOPE_COMM);
			}
			write_buffer_size = 0;
		}
		QThread::msleep(1);
	}

	//free buffer & release
	POSAFE_DELETE_ARRAY(read_buffer_ptr);
	closeDevice(&serial_port);
}

bool CPlainTextSerial::restartDevice()
{
	if (m_is_inited)
	{
		printlog_lvs2("PlainSerial rebooting now.", LOG_SCOPE_COMM);
		m_is_reopen_device = true;
	}
	return true;
}

void CPlainTextSerial::onReOpenedDevice()
{
	printlog_lvs2("PlainSerial rebooting is successed.", LOG_SCOPE_COMM);
	onResetDevice();
}

void CPlainTextSerial::onResetDevice()
{
}

bool CPlainTextSerial::onReadData(u8*& buffer_ptr, i32 buffer_size)
{
	return false;
}

bool CPlainTextSerial::writeData(u8* buffer_ptr, i32 buffer_size)
{
	if (!m_is_inited || !buffer_ptr || buffer_size <= 0)
	{
		return false;
	}

	bool is_overlapped = false;
	{
		exlock_guard(m_write_mutex);
		if (m_write_buffer_size + buffer_size < POIO_READSIZE)
		{
			CPOBase::memCopy(m_write_buffer_ptr + m_write_buffer_size, buffer_ptr, buffer_size);
			m_write_buffer_size += buffer_size;
		}
		else
		{
			CPOBase::memCopy(m_write_buffer_ptr, buffer_ptr, buffer_size);
			m_write_buffer_size = buffer_size;
			is_overlapped = true;
		}
	}
	if (is_overlapped)
	{
		printlog_lvs2("PlainSerial write buffer is overlapped...", LOG_SCOPE_COMM);
	}
	return true;
}

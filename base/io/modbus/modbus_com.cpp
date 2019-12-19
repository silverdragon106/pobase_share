#include "modbus_com.h"
#include "struct.h"
#include "base.h"
#include "log_config.h"

using namespace serial;

CModBusCom::CModBusCom()
{
	m_is_master = true;
	m_modbus_param_ptr = NULL;

	m_is_inited = false;
	m_is_thread_cancel = false;
	m_is_reopen_device = false;
	
	m_buffer_pos1 = 0;
	m_buffer_pos2 = 0;
	m_cmd_buffer_ptr = po_new u8[POIO_CMDQUEUE*POIO_CMDMAXLEN];
	memset(m_cmd_buffer_ptr, 0, POIO_CMDQUEUE*POIO_CMDMAXLEN);
	memset(m_cmd_len, 0, POIO_CMDQUEUE);

	m_packet_ptr = po_new CModBusPacket();

	connect(this, SIGNAL(reopenedDevice()), this, SLOT(onReopenedDevice()), Qt::QueuedConnection);
}

CModBusCom::~CModBusCom()
{
	POSAFE_DELETE(m_packet_ptr);
	POSAFE_DELETE_ARRAY(m_cmd_buffer_ptr);
}

bool CModBusCom::initInstance(CModbusDevParam* modbus_param_ptr, bool is_master, i32 scan_interval)
{
	ERR_PREPARE(0);
	ERR_PREPARE(1);
	if (!modbus_param_ptr || m_is_inited)
	{
		return false;
	}

	m_is_master = is_master;
	m_scan_interval = scan_interval;
	m_modbus_param_ptr = modbus_param_ptr;

	m_buffer_pos1 = 0;
	m_buffer_pos2 = 0;
	m_is_reopen_device = false;

#if defined(POR_WITH_IOMODULE)
	m_is_thread_cancel = false;
	QThreadStart();
#endif

	m_is_inited = true;
	return true;
}

void CModBusCom::exitInstance()
{
	if (m_is_inited)
	{
		m_is_inited = false;

#if defined(POR_WITH_IOMODULE)
		m_is_thread_cancel = true;
		QThreadStop();
#endif
	}
}

bool CModBusCom::openDevice(Serial* port, const CModbusDevParam& dev_param)
{
	if (!port)
	{
		printlog_lvs3("Can't open modbus comport(invalid param)", LOG_SCOPE_IO);
		return false;
	}

	closeDevice(port);
	singlelog_lvs4(QString("Open modbus comport[%1]").arg(dev_param.m_port_name.c_str()), LOG_SCOPE_IO);

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
        printlog_lvs3("Can't open comport(internal error).", LOG_SCOPE_IO);
        printlog_lvs3(QString("\tPort:%1").arg(dev_param.m_port_name.c_str()), LOG_SCOPE_IO);
		printlog_lvs3(QString("\tRSMode:%1").arg(dev_param.m_rs_mode), LOG_SCOPE_IO);
        printlog_lvs3(QString("\tBaudrate:%1").arg(dev_param.m_baud_rate), LOG_SCOPE_IO);
        printlog_lvs3(QString("\tBytesize:%1").arg(dev_param.m_data_bits), LOG_SCOPE_IO);
        printlog_lvs3(QString("\tParity:%1").arg(dev_param.m_parity), LOG_SCOPE_IO);
        printlog_lvs3(QString("\tStopbits:%1").arg(dev_param.m_stop_bits), LOG_SCOPE_IO);
        printlog_lvs3(QString("\tFlowcontrol:%1").arg(dev_param.m_flow_control), LOG_SCOPE_IO);
		return false;
	}

	port->setTimeout(serial::Timeout::simpleTimeout((uint32_t)1, (uint32_t)2000));
	return true;
}

void CModBusCom::closeDevice(Serial* port)
{
	if (port->isOpen())
	{
		port->close();
	}
	port->clearError();
}

bool CModBusCom::onAcceptData(u8*& buffer_ptr, i32 len, i32 dev, i32 cmd, bool& is_completed)
{
	//if readed buffer is empty, wait to read next data
	if (len == 0) 
	{
		return false;
	}

	//check current buffer
	if (!m_packet_ptr->isModBusPacket(m_is_master, dev, buffer_ptr, len))
	{
		return false;
	}

	//if current packet_ptr is invalid, increase buffer point when buffer has some error bytes...
	if (!m_packet_ptr->isValid())
	{
		buffer_ptr++;
		return true; 
	}
	if (m_packet_ptr->m_cmd == cmd || m_packet_ptr->m_cmd == (cmd + MODBUS_ERROR))
	{
		onReadPacket(m_packet_ptr);
		is_completed = true;
	}
	buffer_ptr += m_packet_ptr->m_len;
	return true;
}

void CModBusCom::onScanDeviceTimer()
{
}

void CModBusCom::onResetDevice()
{
}

void CModBusCom::writeCmdToDevice(CModBusPacket* pak)
{
	if (!pak->isValid())
	{
		return;
	}

	m_queue_mutex.lock();
	i32 pos = (m_buffer_pos2 + 1) % POIO_CMDQUEUE;
	if (pos != m_buffer_pos1)
	{
		m_buffer_pos2 = pos;
		m_cmd_len[m_buffer_pos2] = pak->m_len;
		CPOBase::memCopy(m_cmd_buffer_ptr + pos*POIO_CMDMAXLEN, pak->m_buffer_ptr, pak->m_len);
		m_queue_mutex.unlock();
		ERR_UNOCCUR(0);
	}
	else
	{
		m_queue_mutex.unlock();
		ERR_OCCUR2(0, printlog_lvs2(QString("IOCom command queue is full. [%1]").arg(_err_rep0), LOG_SCOPE_IO),
				alarmlog2(kPOErrMBComThrottle, QString::number(_err_rep0)));
	}
}

void CModBusCom::writeErrResponse(CModBusPacket* packet_ptr, i32 errtype)
{
	if (m_is_master || !packet_ptr)
	{
		return;
	}

	CModBusPacket pak(false, packet_ptr->m_dev_addr, packet_ptr->m_cmd + MODBUS_ERROR);
	pak.m_data_u->err.code = errtype;
	writeCmdToDevice(&pak);
}

bool CModBusCom::hasQueue()
{
	QMutexLocker l(&m_queue_mutex);
	return (m_buffer_pos1 != m_buffer_pos2);
}

void CModBusCom::run()
{
	singlelog_lv0("The ModbusCom thread is");

	CModbusDevParam modbus_dev = m_modbus_param_ptr->getValue();
	Serial serial_port;
	openDevice(&serial_port, modbus_dev);

	i64 cur_time, last_time = 0;
	QElapsedTimer elasped_timer;
	elasped_timer.start();

	u16 crccode;
	i32 len, lastlen;
	i32 cmd_index, retry_count, last_send_time, send_cmd;
	i32 read_bytes, max_read_bytes;
	u8* cmd_buffer_ptr = po_new u8[POIO_CMDMAXLEN];
	u8* read_buffer_ptr = po_new u8[POIO_READSIZE];
	u8* buffer_ptr = read_buffer_ptr;
	u8* last_buffer_ptr = read_buffer_ptr;
	bool is_retry = false;
	bool is_completed = true;

	while (!m_is_thread_cancel)
	{
		if (m_is_reopen_device)
		{
			modbus_dev = m_modbus_param_ptr->getValue();
			if (openDevice(&serial_port, modbus_dev))
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
											.arg(serial_port.errorNumber()).arg(_err_rep1), LOG_SCOPE_IO),
			{
				postring dev_name;
				dev_name = modbus_dev.getRS232PortName();
				alarmlog2(kPOErrIOModuleOpen, QString("%1,%2").arg(dev_name.c_str()).arg(_err_rep1));

				emit deviceError(kPODescIOManager, kPODevErrConnect, _err_rep1);
			});

			m_is_reopen_device = true;
			serial_port.clearError();
			QThread::msleep(2000);
			continue;
		}
		ERR_UNOCCUR(1);

		//call parent function to send command for device scan
		cur_time = elasped_timer.elapsed();
		if (cur_time - last_time > m_scan_interval)
		{
			// 명령대기렬이 비였는가를 검사한다.
			bool is_empty = false;
			{
				QMutexLocker l(&m_queue_mutex);
				is_empty = (m_buffer_pos1 == m_buffer_pos2);
			}

			// 명령대기렬이 비여있는 경우에만 건반읽기를 진행한다.
			if (is_empty)
			{
				onScanDeviceTimer();
				last_time = cur_time;
			}
		}
		
		if (is_completed)
		{
			//get modbus packet_ptr from queue
			len = 0;
			{
				QMutexLocker l(&m_queue_mutex);
				if (m_buffer_pos1 != m_buffer_pos2)
				{
					cmd_index = (m_buffer_pos1 + 1) % POIO_CMDQUEUE;
					CPOBase::memCopy(cmd_buffer_ptr, m_cmd_buffer_ptr + cmd_index * POIO_CMDMAXLEN, POIO_CMDMAXLEN);
					len = m_cmd_len[cmd_index];
				}
			}

			if (len > 4) //at least modbus packet_ptr has address, function, crccode(u16)
			{
				is_completed = false;
				last_send_time = cur_time;
				send_cmd = cmd_buffer_ptr[1];

				//make CRCcode for each modbus packet_ptr
				crccode = CModBusPacket::makeCRCCode(cmd_buffer_ptr, len - 2);
				CPOBase::memCopy(cmd_buffer_ptr + len - 2, (u8*)&crccode, 2);

				//write command
				if (serial_port.write((const uint8_t*)cmd_buffer_ptr, len) == 0)
				{
					printlog_lvs2(QString("Serial port writing failed, errcode: %1")
									.arg(serial_port.errorNumber()), LOG_SCOPE_IO);
				}
			}
		}
		else
		{
			//read response for each command
			max_read_bytes = POIO_READSIZE - (buffer_ptr - read_buffer_ptr);
			read_bytes = (i32)serial_port.read((u8*)buffer_ptr, max_read_bytes);
			if (read_bytes > 0)
			{
				buffer_ptr += read_bytes;
				while (onAcceptData(last_buffer_ptr, (i32)(buffer_ptr - last_buffer_ptr), 
								modbus_dev.m_dev_address, send_cmd, is_completed));

				//[start] pReadBuffer--[readed packet_ptr]--pLastBuffer--[reading packet_ptr]--pBuffer--[end]
				lastlen = buffer_ptr - last_buffer_ptr;
				if (lastlen >= POIO_HALFREADSIZE) //max packet size
				{
					last_buffer_ptr = buffer_ptr - POIO_SMALLREADSIZE; //reserved readed size
					lastlen = POIO_SMALLREADSIZE;
				}
				if (last_buffer_ptr - read_buffer_ptr >= POIO_HALFREADSIZE && is_completed) //check start packe position
				{
					CPOBase::memCopy(read_buffer_ptr, last_buffer_ptr, lastlen);
					last_buffer_ptr = read_buffer_ptr;
					buffer_ptr = read_buffer_ptr + lastlen;
					printlog_lv1("IOModule ReadBuffer was swaped.");
				}
			}

			//check packet sending completed
			if (is_completed)
			{
				if (send_cmd == m_packet_ptr->m_cmd)
				{
					retry_count = 0;
					QMutexLocker l(&m_queue_mutex);
					m_buffer_pos1 = cmd_index;
				}
				else
				{
					printlog_lvs2("IOModule Modbus recived fail response.", LOG_SCOPE_IO);
					is_retry = true;
				}
			}
			else
			{
				if (cur_time - last_send_time > modbus_dev.m_time_out)
				{
					is_retry = true;
				}
			}

			//retry when fail response, timeout...
			if (is_retry)
			{
				is_retry = false;
				is_completed = true;
				if (retry_count < modbus_dev.m_retry_count)
				{
					retry_count++;
				}
				else
				{
					retry_count = 0;
					m_is_reopen_device = true;
					closeDevice(&serial_port);
				}
			}
		}

		QThread::msleep(1);
	}

	//send closed packet to modbus...
	CModBusPacket* pak = getClosedPacket();
	if (pak && serial_port.isOpen())
	{
		pak->makeCRCCode();
		serial_port.write((const uint8_t*)pak->m_buffer_ptr, pak->m_len);
		serial_port.flush();
	}

	//free buffer & release
	POSAFE_DELETE(pak);
	POSAFE_DELETE_ARRAY(cmd_buffer_ptr);
	POSAFE_DELETE_ARRAY(read_buffer_ptr);
	closeDevice(&serial_port);
}

CModBusPacket* CModBusCom::getClosedPacket()
{
	return NULL;
}

bool CModBusCom::restartDevice()
{
	if (m_is_inited)
	{
		printlog_lvs2("IOModule rebooting now.", LOG_SCOPE_IO);
		m_is_reopen_device = true;
	}
	return true;
}

void CModBusCom::onReopenedDevice()
{
	printlog_lvs2("IOModule rebooting is successed.", LOG_SCOPE_IO);
	onResetDevice();
}

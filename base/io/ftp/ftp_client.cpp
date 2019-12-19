#include "ftp_client.h"
#include "logger/logger.h"
#include "base.h"

#if defined(POR_WITH_FTP)
#include "qftp.h"
#include "qurlinfo.h"

const i32 kFtpCallBackInterval = 100;	//100ms

CFtpClient::CFtpClient()
{
	m_is_inited = false;
	m_ftp_param_ptr = NULL;

	m_ftp_ptr = NULL;
	m_ftp_status = kFtpStatusNone;
	
	moveToThread(this);
	m_callback_timer.moveToThread(this);
}

CFtpClient::~CFtpClient()
{
	exitInstance();
}

bool CFtpClient::initInstance(CFtpDev* ftp_param_ptr)
{
	if (!ftp_param_ptr)
	{
		return false;
	}

	if (!m_is_inited)
	{
		singlelog_lv1("FtpClient InitInstance is");
		m_is_inited = true;
		m_ftp_param_ptr = ftp_param_ptr;

		QThread::start();
	}
	return true;
}

void CFtpClient::exitInstance()
{
	if (m_is_inited)
	{
		singlelog_lv1("FtpClient ExitInstance is");
		QEventLoopStop();
		m_is_inited = false;
	}
}

void CFtpClient::run()
{
	singlelog_lv0("The FtpClient thread is");
	initEvent();
	connectFtp();

	exec(); //start event loop

	disconnectFtp();
	disconnectEvent();
}

void CFtpClient::initEvent()
{
	qRegisterMetaType<CFtpDev>("FtpSetting");

	disconnect();
	m_callback_timer.disconnect();
	
	connect(this, SIGNAL(ftpConnected()), this, SLOT(onConnected()));
	connect(this, SIGNAL(ftpDisconnect()), this, SLOT(onDisconnection()));
	connect(this, SIGNAL(ftpChangeConnection(CFtpDev)), this, SLOT(onChangedConnection(CFtpDev)));

	m_callback_timer.setInterval(kFtpCallBackInterval);
	m_callback_timer.setSingleShot(false);
	m_callback_timer.start();
	connect(&m_callback_timer, SIGNAL(timeout()), this, SLOT(onCallback()), Qt::AutoConnection);
}

void CFtpClient::ftpCommandFinished(i32 cmd_id, bool is_error)
{
	if (!m_ftp_ptr)
	{
		return;
	}

	i32 cmd = m_ftp_ptr->currentCommand();
	switch (cmd)
	{
		case QFtp::ConnectToHost:
		{
			if (is_error && !onErrorFtpConnectHost())
			{
				printlog_lvs3(QString("Disconnect! %1").arg(m_ftp_ptr->errorString()), LOG_SCOPE_FTP);
				emit ftpDisconnect();
				return;
			}
			else
			{
				onFinishedFtpConnected();
			}
			break;
		}
		case QFtp::Login:
		{
			if (is_error && !onErrorFtpLogin())
			{
				printlog_lvs3(QString("Disconnect! %1").arg(m_ftp_ptr->errorString()), LOG_SCOPE_FTP);
				emit ftpDisconnect();
				return;
			}
			else
			{
				onFinishedFtpLogin();
				emit ftpConnected();
			}
			break;
		}
		case QFtp::Close:
		{
			onFinishedFtpClose();
			emit ftpDisconnect();
			break;
		}
		case QFtp::Put:
		{
			if (is_error && !onErrorFtpPut())
			{
				printlog_lvs3("FtpClient put error", LOG_SCOPE_FTP);
			}
			else
			{
				if (onFinishedFtpPut())
				{
					addFtpStatus(kFtpStatusSent);
					onCallback();
				}
			}
			break;
		}
		case QFtp::Mkdir:
		{
			if (is_error && !onErrorFtpMkdir())
			{
				printlog_lvs3(QString("Ftpclient mkdir error: %1").arg(m_ftp_ptr->errorString()), LOG_SCOPE_FTP);
				emit ftpDisconnect();
			}
			else
			{
				onFinishedFtpMkdir();
			}
			break;
		}
		case QFtp::Cd:
		{
			if (is_error && !onErrorFtpCd())
			{
				printlog_lvs3("Ftpclient cd error", LOG_SCOPE_FTP);
			}
			else
			{
				onFinishedFtpCd();
			}
			break;
		}
		default:
		{
			printlog_lvs3(QString("FtpCommand is finished, cmd:%1, error:%2").arg(cmd).arg((i32)is_error), LOG_SCOPE_FTP);
			break;
		}
	}
}

void CFtpClient::disconnectFtp()
{
	if (!m_ftp_ptr)
	{
		return;
	}

	if (m_ftp_ptr)
	{
		//disconnect event
		m_ftp_ptr->disconnect();

		//destory ftp_module
		m_ftp_ptr->abort();
		m_ftp_ptr->deleteLater();
		m_ftp_ptr = NULL;
	}
	m_ftp_status = kFtpStatusNone;
}

void CFtpClient::disconnectEvent()
{
	disconnect();
	m_callback_timer.disconnect();
	m_callback_timer.stop();
}

void CFtpClient::updateDataTransferProgress(qint64 readBytes, qint64 totalBytes)
{
}

void CFtpClient::onCallback()
{
	if (m_ftp_ptr && m_ftp_ptr->state() == QFtp::Disconnected)
	{
		onDisconnection();
	}
	
	if (isFtpAvailable())
	{
		if (m_ftp_ptr->state() != QFtp::LoggedIn)
		{
			onDisconnection(); //when disconnected after logged
		}
		else if (isFtpSent())
		{
			onFtpCallback();
		}
	}
}

void CFtpClient::onConnected()
{
	m_ftp_status |= kFtpStatusLogin;

	onFtpConnected();
	printlog_lvs2("FtpClient is connected", LOG_SCOPE_FTP);
}

void CFtpClient::onDisconnection()
{
	disconnectFtp();

	onFtpDisconnected();
	printlog_lvs3("FtpClient is disconnected", LOG_SCOPE_FTP);
}

void CFtpClient::onChangedConnection(CFtpDev ftp_setting)
{
	disconnectFtp();

	m_ftp_ptr = po_new QFtp(this);
	m_ftp_ptr->moveToThread(this);
	connect(m_ftp_ptr, SIGNAL(commandFinished(i32, bool)), this, SLOT(ftpCommandFinished(i32, bool)));
 	//connect(m_ftp_ptr, SIGNAL(dataTransferProgress(qint64, qint64)), this, SLOT(updateDataTransferProgress(qint64, qint64)));
	
	//connect and login to FtpServer
	//ftp_setting: thread safe
	m_ftp_ptr->connectToHost(ftp_setting.m_ftp_hostname.c_str(), ftp_setting.m_ftp_port);
	m_ftp_ptr->login(ftp_setting.m_ftp_username.c_str(), ftp_setting.m_ftp_password.c_str());
}

void CFtpClient::connectFtp()
{
	if (!m_is_inited || !m_ftp_param_ptr)
	{
		printlog_lv1("FtpClient setting invalid.");
		return;
	}
	emit ftpChangeConnection(*m_ftp_param_ptr);
}

void CFtpClient::addFtpStatus(i32 status)
{
	m_ftp_status |= status;
}

void CFtpClient::removeFtpStatus(i32 status)
{
	m_ftp_status &= ~status;
}

bool CFtpClient::isFtpAvailable()
{
	if (!m_ftp_ptr || !m_ftp_param_ptr || !m_is_inited)
	{
		return false;
	}
	return CPOBase::bitCheck(m_ftp_status, kFtpStatusConnected);
}

bool CFtpClient::isFtpSent()
{
	return CPOBase::bitCheck(m_ftp_status, kFtpStatusSent);
}

bool CFtpClient::onFtpConnected()
{
	return false;
}

bool CFtpClient::onFtpDisconnected()
{
	return false;
}

bool CFtpClient::onFtpCallback()
{
	return false;
}

bool CFtpClient::onFinishedFtpConnected()
{
	return false;
}

bool CFtpClient::onFinishedFtpLogin()
{
	return false;
}

bool CFtpClient::onFinishedFtpClose()
{
	return false;
}

bool CFtpClient::onFinishedFtpPut()
{
	return false;
}

bool CFtpClient::onFinishedFtpMkdir()
{
	return false;
}

bool CFtpClient::onFinishedFtpCd()
{
	return false;
}

bool CFtpClient::onErrorFtpConnectHost()
{
	return false;
}

bool CFtpClient::onErrorFtpLogin()
{
	return false;
}

bool CFtpClient::onErrorFtpPut()
{
	return false;
}

bool CFtpClient::onErrorFtpMkdir()
{
	return false;
}

bool CFtpClient::onErrorFtpCd()
{
	return false;
}
#endif
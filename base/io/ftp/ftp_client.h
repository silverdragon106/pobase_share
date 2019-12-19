#pragma once
#include "struct.h"
#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QTimer>

enum FtpStatusTypes
{
	kFtpStatusNone = 0x00,
	kFtpStatusConnected = 0x01,
	kFtpStatusSent = 0x02,

	kFtpStatusLogin = (kFtpStatusConnected | kFtpStatusSent)
};

class QFtp;
class CFtpClient : public QThread
{
	Q_OBJECT

#if defined(POR_WITH_FTP)
public:
	CFtpClient();
	~CFtpClient();

	virtual bool			initInstance(CFtpDev* ftp_param_ptr);
	virtual void			exitInstance();

	void					connectFtp();
	void					disconnectFtp();

	virtual bool			onFtpConnected();
	virtual bool			onFtpDisconnected();
	virtual	bool			onFtpCallback();

	virtual bool			onFinishedFtpConnected();
	virtual bool			onFinishedFtpLogin();
	virtual bool			onFinishedFtpClose();
	virtual bool			onFinishedFtpPut();
	virtual bool			onFinishedFtpMkdir();
	virtual bool			onFinishedFtpCd();

	virtual bool			onErrorFtpConnectHost();
	virtual bool			onErrorFtpLogin();
	virtual bool			onErrorFtpPut();
	virtual bool			onErrorFtpMkdir();
	virtual bool			onErrorFtpCd();

	void					addFtpStatus(i32 status);
	void					removeFtpStatus(i32 status);

	bool					isFtpAvailable();
	bool					isFtpSent();

	inline QFtp*			getFtp() { return m_ftp_ptr; };

private:
	void					initEvent();
	void					disconnectEvent();

protected:
	void					run() Q_DECL_OVERRIDE;

signals:
	void					ftpConnected();
	void					ftpDisconnect();
	void					ftpChangeConnection(CFtpDev);
	
private slots:
	void					onConnected();
	void					onDisconnection();
	void					onChangedConnection(CFtpDev ftp_setting);
	void					onCallback();
	void					updateDataTransferProgress(qint64 readBytes, qint64 totalBytes);
	void					ftpCommandFinished(i32 cmd_id, bool error);

private:
	std::atomic<bool>		m_is_inited;
	std::atomic<i32>		m_ftp_status;
	CFtpDev*				m_ftp_param_ptr;

	QFtp*					m_ftp_ptr;
	QTimer					m_callback_timer;
#endif
};
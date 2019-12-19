#pragma once

#include "base.h"
namespace test_case2
{
    class Task1 : public QObject
    {
        Q_OBJECT
    public slots:
        void	doWork();
        void	startOnThread(QThread* thread);

    signals:
        void	finished();

    };

    class Task2 : public QObject
    {
        Q_OBJECT
    public slots:
        void	doWork();
        void	startOnThread(QThread* thread);
    signals:
        void	finished();
    };
}

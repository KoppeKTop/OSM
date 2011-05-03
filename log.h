#ifndef MYLOG_H_
#define MYLOG_H_

#include <iostream>
#include <ctime>

using namespace std;

template <typename T>
void log_it(T v)
{
    char buffer[80];
    time_t seconds = time(NULL);
    tm* timeinfo = localtime(&seconds);
    char* format = "%B %d %H:%M:%S ";
    strftime(buffer, 80, format, timeinfo);
    cout << buffer << v << endl;
}

#endif

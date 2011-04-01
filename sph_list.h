#ifndef _SPH_LIST_H_
#define _SPH_LIST_H_

#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <cuda.h>
#include <cutil_inline.h>

class OutOfBoundError: public exception
{
public:
	OutOfBoundError(const char * filename, const int line)
	{
		char msg_str[] = "Coordinate index out of bounds on: %s:%d";
		m_what_str = (char *)malloc(strlen(msg_str) + strlen(filename) + 15);
		sprintf(m_what_str, "Coordinate index out of bounds on: %s:%d\n", filename, line);
	}
	virtual const char* what() const throw()
	{
		return m_what_str;
	}
	
private:
	char * m_what_str;
};

class SizeError: public exception
{
	virtual const char* what() const throw()
	{
		return "Maximum size achieved";
	}
};

class sph_list
{
public: 
    sph_list(int max_len);
    float4 * get_host_ptr()  {   return host;    }
    float4 * get_dev_ptr()
    float4 get(int idx);
    void append(float4 val);
    int size()  {   return sz;  }
    ~sph_list() {
        delete [] host;
        if (dev != NULL)
            cudaFree(dev);
    }
private:
    float4 * dev;
    float4 * host;
    int sz;
}


#endif
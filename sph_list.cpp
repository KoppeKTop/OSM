#include <sph_list.h>


sph_list::sph_list(int max_len)
: sz(0)
{
    host = new float4[max_len];
    dev = NULL;
}

float4 * sph_list::get_dev_ptr()
{
    if (dev == NULL)
    {
        cutilSafeCall( cudaMalloc( (void**) &dev, max_len * sizeof(float4)));
        cutilSafeCall( cudaMemcpy(dev, host, sz*sizeof(float4), cudaMemcpyHostToDevice));
    }
    return dev;
}

float4 sph_list::get(int idx)
{
    if (idx < size)
    {
        return host[idx];
    }
    // raise exception
    throw OutOfBoundError(__FILE__, __LINE__);
}

void sph_list::append(float4 val)
{
    if (size != max_len)
    {
        host[sz] = val;
        if (dev != NULL)
        {
            cutilSafeCall( cudaMemcpy(dev+sz, host+sz, sizeof(float4), cudaMemcpyHostToDevice));
        }
        sz ++;
    }
    else
    {
        // raise exception
        throw SizeError();
    }
}

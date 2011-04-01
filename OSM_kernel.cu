/*
 * Device code.
 */

#ifndef _OSM_KERNEL_H_
#define _OSM_KERNEL_H_

#include <stdio.h>

#define SQR(x)  ((x)*(x))

__device__ __host__ float 
distance(float4 pnt1, float4 pnt2)
{
    return sqrt(SQR(pnt1.x-pnt2.x) + SQR(pnt1.y-pnt2.y) + SQR(pnt1.z-pnt2.z));
}

__device__ __host__ float 
overlapping(float r1, float r2, float dist)
// доля от суммы радиусов
{
    return sqrt((r1+r2+dist)*(r2+dist-r1)*(r1+dist-r2)*(r1+r2-dist))/(dist * (r1+r2));
}    

__device__ __host__ int 
is_overlapped(float4 pnt1, float4 pnt2, float max_overlapping)
{
    float dist = distance(pnt1, pnt2);
    float radius_sum = pnt1.v + pnt2.z;
    return (dist < radius_sum && overlapping(pnt1.v, pnt2.v, dist) > max_overlapping);
}

__global__ void 
overlap_list(float4 * spheres, float4 curr_sph, int * results, float max_overlapping, int curr_cnt)
{
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < curr_cnt)
    {
        float4 cmp_sph = spheres[idx];
        if (is_overlapped(curr_sph, cmp_sph, max_overlapping))
        {
            int old_cnt = atomicAdd(results, 1);
            results[old_cnt+1] = idx;
        }
    }
}

#endif // #ifndef _OSM_KERNEL_H_

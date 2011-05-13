/*
 * Device code.
 */

#ifndef _OSM_KERNEL_H_
#define _OSM_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#define SQR(x)  ((x)*(x))

__device__ __host__ float 
pnt_dist(const float4 pnt1, const float4 pnt2)
{
    return sqrt(SQR(pnt1.x-pnt2.x) + SQR(pnt1.y-pnt2.y) + SQR(pnt1.z-pnt2.z));
}

__device__ __host__ float 
overlapping(const float r1, const float r2, const float dist)
// доля от суммы радиусов
{
    float result = 0;
    if (dist < (r1+r2))    {
        float S = sqrt((r1+r2+dist)*(r2+dist-r1)*(r1+dist-r2)*(r1+r2-dist))/4;
        float h = 2 * S / dist;
        result = 2 * h / (r1 + r2);
    }
    return result;
}    

__device__ __host__ int 
is_overlapped(const float4 pnt1, const float4 pnt2, const float max_overlapping)
{
    float d = pnt_dist(pnt1, pnt2);
    float radius_sum = pnt1.w + pnt2.w;
    return (d < radius_sum && (overlapping(pnt1.w, pnt2.w, d) > max_overlapping));
}

__device__ __host__ int
slightly_overlap(const float4 pnt1, const float4 pnt2, const float max_overlapping)
{   
    float d = pnt_dist(pnt1, pnt2);
    float radius_sum = pnt1.w + pnt2.w;
    return (d < radius_sum && (overlapping(pnt1.w, pnt2.w, d) <= max_overlapping));
}

//__global__ void 
//overlap_list(float4 * spheres, float4 curr_sph, int * results, int * res_cnt, float max_overlapping, int curr_cnt)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < curr_cnt)
//    {
//        float4 cmp_sph = spheres[idx];
//        if (is_overlapped(curr_sph, cmp_sph, max_overlapping))
//        {
//            int old_cnt = atomicAdd(res_cnt, 1);
//            results[old_cnt+1] = idx;
//        }
//    }
//}

__global__ void 
nei_list(float4 * spheres, float4 curr_sph, int * results, int * res_cnt, int curr_cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < curr_cnt)
    {
        float4 cmp_sph = spheres[idx];
        if (pnt_dist(cmp_sph, curr_sph) < 3 * curr_sph.w)
        {
            int old_cnt = atomicAdd(res_cnt, 1);
            results[old_cnt] = idx;
        }
    }
}

#endif // #ifndef _OSM_KERNEL_H_

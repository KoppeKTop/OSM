/*
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <list>
#include <pair>
#include <algorithm>

// includes, project
#include <cutil_inline.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/functional.h>
//#include <thrust/reduce.h>

// includes, kernels
#include <OSM_kernel.cu>

// includes cpp functions
#include <OSM.h>
#include <sph_list.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    srand ( time(NULL) );
    runTest( argc, argv);

    cutilExit(argc, argv);
}

double randf()
{
    return (double)rand()/RAND_MAX;
}

float GetSphereRadius()
{
    return 4.0;
}

float4 GenRndPoint(float3 dim_len)
{
    float4 result;
    result.x = randf() * dim_len.x;
    result.y = randf() * dim_len.y;
    result.z = randf() * dim_len.z;
    result.v = GetSphereRadius();
    return result;
}

#define BLOCK_DIM 256



int * get_overlap_list(sph_list & spheres, float4 curr_sph, max_overlapping)
{
    int max_overlap = 10;
    int * d_overlaps = NULL;
    size_t overlaps_sz = (max_overlap+1)*sizeof(int);
    int * results = (int *)malloc(overlaps_sz);
    if (spheres.size() < BLOCK_DIM * 10)
    {
        // Use CPU
        memset(results, 0, overlaps_sz);
        overlap_list(spheres.host(), curr_sph, results, max_overlapping, spheres.size());
    }
    else
    {
        // Use GPU
        int * d_results = NULL;
        cutilSafeCall( cudaMalloc( (void**) & d_results, overlaps_sz));
        cutilSafeCall( cudaMemset( d_results, 0, overlaps_sz));
        dim3 grid(spheres.size()/BLOCK_DIM+1,1);
        dim3 block(BLOCK_DIM, 1, 1);
        overlap_list<<<grid, block>>>(spheres.dev(), curr_sph, d_results, max_overlapping, spheres.size());
        cutilSafeCall( cudaThreadSynchronize());
        cutilSafeCall( cudaMemcpy(results, d_results, overlaps_sz, cudaMemcpyDeviceToHost));
        cutilSafeCall( cudaFree(d_results));
    }
    return results;
}

bool dist_cmp(pair <float4, float> a, pair <float4, float> b)
{
    return a.second < b.second;
}

struct overlapper {
    float4 curr_sph;
    overlapper(float4 sph) { curr_sph = sph; }
    pair <float4, float> operator() (pair <float4, float> a) 
    {
        a.second = overlapping(a.v, curr_sph.v, distance(a, curr_sph));
        return a;
    }
} overlapper;

float min_dist(float r1, float r2)
{
    return sqrt((r1+r2+dist)*(r2+dist-r1)*(r1+dist-r2)*(r1+r2-dist))

void 

float4 * GenMaxPacked(const int max_cnt, float3 dim_len)
{
    int curr_cnt = 0;
    const int max_holost = 100;
    int holost = 0;
    sph_list spheres(max_cnt);
    
    const int max_moves = 100;
    while (curr_cnt < max_cnt && holost++ < max_holost)
    {
        float4 new_pnt = GenRndPoint(dim_len);
        bool add = false;
        int moves = 0;
        while (moves++ < max_moves && !add)
        {
            int * overlaps = get_overlap_list(spheres, new_pnt, max_overlapping)
            int over_cnt = overlaps[0];
            if (over_cnt != 0)
            {
                vector< pair <float4, float> > overlap_arr(over_cnt);
                for (int i = 0; i < over_cnt; ++i)
                {
                    float4 curr_sph = spheres.get(overlaps[i+1]);
                    overlap_arr[i] = pair <float4, float>(curr_sph, 0);
                }
                transform(overlap_arr.begin(), overlap_arr.end(), overlap_arr.begin(), overlapper(new_pnt))
                sort(overlap_arr.begin(), overlap_arr.end(), dist_cmp);
                if (overlap_arr[0].second > max_ovelapping)
                {
                    move_local(overlap_arr, new_pnt);
                }
            }
            else
            {
                add = true;
            }
            moves = 0;
            free(overlaps);
        }
        if (add)
        {
            spheres.append(new_pnt);
            holost = 0;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    float* h_idata = (float*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
    float* reference = (float*) malloc( mem_size);
    computeGold( reference, h_idata, num_threads);

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat",
                                      h_odata, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res = cutComparef( reference, h_odata, num_threads);
        printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}

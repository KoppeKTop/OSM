/*
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fstream>

#include <vector>
#include <list>
//#include <pair>
#include <algorithm>

// includes, project
//#include <cutil_inline.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

// includes, kernels
#include <OSM_kernel.cu>

// includes cpp functions
#include <OSM.h>

typedef float4 sph;
typedef thrust::device_vector<float4> d_sph_list;
typedef thrust::host_vector<float4>   h_sph_list;

using namespace std;

// For BGL connection algorithm
#include <boost/config.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>

// For logging
#include <glog/logging.h>

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

//    cutilExit(argc, argv);
}

double randf()
{
    return (double)rand()/RAND_MAX;
}

float GetSphereRadius()
{
    return 2.0;
}

sph GenRndPoint(float3 dim_len)
{
    float4 result;
    result.x = randf() * dim_len.x;
    result.y = randf() * dim_len.y;
    result.z = randf() * dim_len.z;
    result.w = GetSphereRadius();
    return result;
}

const float max_overlapping = 0.4;

#define BLOCK_DIM 256


struct dist_gt
{
    sph curr;
    
    dist_gt(sph c)    {   curr = c;   }
    
    __host__ __device__
    bool operator()(const sph first, const sph second) const
    {
        float l1 = overlapping(first.w, curr.w, dist(first, curr));
        float l2 = overlapping(second.w, curr.w, dist(second, curr));
        
        //printf("L1 = %f, L2 = %f\n", l1, l2);
        return l1 > l2;
    }
};

float min_dist(float r1, float r2)
{
    float c = SQR(max_overlapping * (r1+r2));
    return 0.5 * (sqrt(4*SQR(r1) - c) + sqrt(4*SQR(r2) - c));
}

#define EPS 0.000001

bool in_space(const float3 & dim_len, const sph & pnt)
{
    return (0 <= pnt.x && pnt.x < dim_len.x &&
            0 <= pnt.y && pnt.y < dim_len.y &&
            0 <= pnt.z && pnt.z < dim_len.z);
}

void move_pnt(const float3 & dim_len, const sph & center_sph, sph & moved_sph)
{
    float old_dist = dist(center_sph, moved_sph);
    if (old_dist < EPS)
    {
        moved_sph = GenRndPoint(dim_len);
        return;
    }
    float r = min_dist(moved_sph.w, center_sph.w)/old_dist;
    moved_sph.x = (moved_sph.x - center_sph.x)*r + center_sph.x;
    moved_sph.y = (moved_sph.y - center_sph.y)*r + center_sph.y;
    moved_sph.z = (moved_sph.z - center_sph.z)*r + center_sph.z;
    if (!in_space(dim_len, moved_sph))
    {
        moved_sph = GenRndPoint(dim_len);
    }
}

ostream& operator<< (ostream& out, float4& item )
{
    out << item.x << ", " << item.y << ", " << item.z << ", " << item.w;
    return out;
}

template <typename Iterator, typename BinaryPredicate>
Iterator my_max_element(Iterator begin, Iterator end, BinaryPredicate gt_op)
{
    Iterator result = begin;
    Iterator curr = result+1;
    while (curr != end)
    {
        if (gt_op(*curr, *result))
            result = curr;
        ++curr;
    }
    return result;
}

int GenMaxPacked(const int max_cnt, const float3 dim_len, h_sph_list & spheres)
{
    int curr_cnt = 0;
    int max_holost = dim_len.x * dim_len.y * dim_len.z;
    int holost = 0;
    
    const int max_moves = 100;
    while (curr_cnt < max_cnt && holost++ < max_holost)
    {
        sph new_pnt = GenRndPoint(dim_len);
        //printf("New point (%i of %i): (%f, %f, %f)\n", curr_cnt, max_cnt, new_pnt.x, new_pnt.y, new_pnt.z);
        if (curr_cnt == 0) {
            spheres[curr_cnt++] = new_pnt;
            holost = 0;
            continue;
        }
        bool add = false;
        int moves = 0;
        while (moves++ < max_moves)
        {
            sph over_sph = *(my_max_element(spheres.begin(), spheres.begin() + curr_cnt, dist_gt(new_pnt)) );
            if (is_overlapped(over_sph, new_pnt, max_overlapping)) {
                move_pnt(dim_len, over_sph, new_pnt);
            } else {
                add = true;
                break;
            }
        }
        if (add) {
            spheres[curr_cnt++] = new_pnt;
            holost = 0;
            cout << "Point #" << curr_cnt << " of " << max_cnt << ": " << new_pnt << endl;
        }
    }
    return curr_cnt;
}

template <typename T>
class OutputItem
{
   public:
       explicit OutputItem( std::ofstream & stream )
                : stream_(&stream)
                {
                }

       void operator()( T const & item )
       {
           *stream_ << item.x << item.y << item.z << item.w;
       }

   private:
       std::ofstream * stream_;
};

void SaveToFile(const h_sph_list & spheres, const char * filename)
{
    FILE * outFile = fopen(filename, "wb");
    
    for (int i = 0; i < spheres.size(); ++i)
    {
        fwrite(&(spheres[i]), sizeof(spheres[i].x), 4, outFile);
    }
    
    fclose(outFile);
}

void RemovePoints( const h_sph_list & spheres )
{
    // from BGL book p 201
    using namespace boost;
    typedef adjacency_list< vecS, vecS, undirectedS > Graph;
    typedef graph_traits< Graph >::vertex_descriptor Vertex;
    
    Graph vg(spheres.size()); 
    printf("Convert points to graph... ");
    for (int curr_vertex = 0; curr_vertex < spheres.size(); ++curr_vertex)
        for (int adj_vertex = curr_vertex+1; adj_vertex < spheres.size(); ++adj_vertex)
            if (is_overlapped(spheres[curr_vertex], spheres[adj_vertex], max_overlapping))
            {
                add_edge(curr_vertex, adj_vertex, vg);
            }
    printf("Done.\n");
    
    std::vector<int> c(num_vertices(vg));
    int num = 
    connected_components(vg, make_iterator_property_map(c.begin(), get(vertex_index, vg), c[0]));
    
    printf("%d clusters in structure\n", num);
}

void
runTest( int argc, char** argv) 
{
    const float dim_sz = 50.0f;
    const double e_max = 0.5f;
    const double r = 2.0f;
    
    const float3 sz = make_float3(dim_sz,dim_sz,dim_sz);
    const double vol = sz.x * sz.y * sz.z;
    const double vol_sph = (4.0/3.0) * 3.14159 * (r*r*r);
    const int max_cnt = vol / vol_sph * (1.0-e_max);
    
    h_sph_list spheres(max_cnt);
    cout << "Start\n";
    int cnt = GenMaxPacked(max_cnt, sz, spheres);
    h_sph_list h_spheres(spheres.begin(), spheres.begin() + cnt);
    
    SaveToFile( h_spheres, "res.dat");
    
    cout << "Done. Points: " << cnt << " of " << max_cnt
    << ". E = " << (1 - vol_sph * cnt / vol) << endl;
}

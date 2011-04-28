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
#include <set>
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

typedef float4 sph;
typedef thrust::device_vector<float4> d_sph_list;
typedef thrust::host_vector<float4>   h_sph_list;

using namespace std;

// For BGL connection algorithm
#include <boost/config.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>

// from BGL book p 201
using namespace boost;
typedef adjacency_list< vecS, vecS, undirectedS > DirGraph;
typedef graph_traits< DirGraph >::vertex_descriptor Vertex;


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
        float l1 = overlapping(first.w, curr.w, pnt_dist(first, curr));
        float l2 = overlapping(second.w, curr.w, pnt_dist(second, curr));
        
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
    float old_dist = pnt_dist(center_sph, moved_sph);
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
    int max_holost = (int)(dim_len.x * dim_len.y * dim_len.z);
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

h_sph_list * LoadFromFile( const char * filename)
{
    FILE * inFile = fopen(filename, "rb");
    sph curr_pnt;
    vector<sph> tmp;
    while(fread(&curr_pnt, sizeof(curr_pnt.x), 4, inFile))
        tmp.push_back( curr_pnt );
    h_sph_list * spheres = new h_sph_list(tmp.begin(), tmp.end());
    return spheres;
}

template <typename OutputType>
void print(OutputType v)
{
    cout << v << " ";
}

template <typename OutputType>
void println(OutputType v)
{
    cout << v << endl;
}

vector<int> * PercolatedClusters( const h_sph_list & spheres, std::vector<int> clusters, const float3 sz )
{
    set<int> * borders = new set<int>[6];
    // find all spheres on the borders
    for (int sph_idx = 0; sph_idx < spheres.size(); ++sph_idx)
    {
        sph curr_sph = spheres[sph_idx];
        if (curr_sph.x-curr_sph.w < 0)
            borders[0].insert(clusters[sph_idx]);
        if (curr_sph.x+curr_sph.w > sz.x)
            borders[1].insert(clusters[sph_idx]);
        if (curr_sph.y-curr_sph.w < 0)
            borders[2].insert(clusters[sph_idx]);
        if (curr_sph.y+curr_sph.w > sz.y)
       	    borders[3].insert(clusters[sph_idx]);
        if (curr_sph.z-curr_sph.w < 0)
            borders[4].insert(clusters[sph_idx]);
        if (curr_sph.z+curr_sph.w > sz.z)
       	    borders[5].insert(clusters[sph_idx]);
    }
    // and save cluster numbers
    // find intersection between borders
    int min_size = borders[0].size();
    for (int dim = 1; dim < 6; ++dim)
    {
        if (borders[dim].size() < min_size)
            min_size = borders[dim].size();
    }
    if (min_size == 0)
    {
        printf("Not percolate\n");
        return NULL;
    }
    vector<int> * res = new vector<int>(borders[0].begin(), borders[0].end());
    vector<int>::iterator last_it = res->end();
    vector<int> tmp(res->size());
    for (int dim = 1; dim < 6; ++dim)
    {
        vector<int>::iterator it = set_intersection(res->begin(), last_it, borders[dim].begin(), borders[dim].end(), tmp.begin());
        if (it - tmp.begin() == 0)
        {
            printf("Non perc [%d]\n", dim);
            return NULL;
        }
        last_it = copy(tmp.begin(), it, res->begin());
    }
    res->resize(last_it-res->begin());
    delete [] borders;
    printf("Percolated clusters: ");
        
    for (vector<int>::iterator it = res->begin(); it != res->end(); ++it)
    {
        int cnt = 0;
        for (vector<int>::iterator cl_it = clusters.begin(); cl_it != clusters.end(); ++cl_it)
             if (*cl_it == *it) cnt++;
        printf("%d (%d) ", *it, cnt);
    }
    printf("\n");
    return res;
}


vector<sph> * RemovePoints( const h_sph_list & spheres, const float3 sz, const min_cnt )
{    
    DirGraph vg(spheres.size()); 
    vector<sph> * tmp_sph = new vector<sph>(spheres.begin(), spheres.end());
    printf("Convert points to graph (max_overlapping = %f)... \n", max_overlapping);
    for (int curr_vertex = 0; curr_vertex < spheres.size(); ++curr_vertex)
        for (int adj_vertex = curr_vertex+1; adj_vertex < spheres.size(); ++adj_vertex)
            if (slightly_overlap(spheres[curr_vertex], spheres[adj_vertex], max_overlapping))
            {
                add_edge(curr_vertex, adj_vertex, vg);
                printf("Edge: %d and %d\n", curr_vertex, adj_vertex);
            }
    printf("Done.\n");
    
    std::vector<int> c(num_vertices(vg));
    int num = 
    connected_components(vg, make_iterator_property_map(c.begin(), get(vertex_index, vg), c[0]));
    
    vector<int> clusters * PercolatedClusters(spheres, c, sz);
    if (!clusters) 
    {
        printf("Can\'t remove points!\n");
        delete [] tmp_sph;
        return NULL;
    }
    // 
}

void
runTest( int argc, char** argv) 
{
    const float dim_sz = 50.0f;
    const double e_max = 0.4f;
    const double r = 2.0f;
    
    const float3 sz = make_float3(dim_sz,dim_sz,dim_sz);
    const double vol = sz.x * sz.y * sz.z;
    const double vol_sph = (4.0/3.0) * 3.14159 * (r*r*r);
    const int max_cnt =(int) (vol / vol_sph * (1.0-e_max));
    
    cout << "Loading\n";
    h_sph_list * spheres = LoadFromFile("res.dat");
    cout << "Start\n";
    //int cnt = GenMaxPacked(max_cnt, sz, spheres);
    
    RemovePoints(*spheres, sz);
    //h_sph_list h_spheres(spheres.begin(), spheres.begin() + cnt);
    //
    //SaveToFile( h_spheres, "res.dat");
    
    //cout << "Done. Points: " << cnt << " of " << max_cnt
    //<< ". E = " << (1 - vol_sph * cnt / vol) << endl;
}

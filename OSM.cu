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
    return 0;

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

bool move_pnt(const float3 & dim_len, const sph & center_sph, sph & moved_sph)
// returns true if point moved
// returns false if created new point
{
    float old_dist = pnt_dist(center_sph, moved_sph);
    if (old_dist < EPS)
    {
        moved_sph = GenRndPoint(dim_len);
        return false;
    }
    float r = min_dist(moved_sph.w, center_sph.w)/old_dist;
    moved_sph.x = (moved_sph.x - center_sph.x)*r + center_sph.x;
    moved_sph.y = (moved_sph.y - center_sph.y)*r + center_sph.y;
    moved_sph.z = (moved_sph.z - center_sph.z)*r + center_sph.z;
    if (!in_space(dim_len, moved_sph))
    {
        moved_sph = GenRndPoint(dim_len);
        return false;
    }
    return true;
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

vector<sph> * CollectNeighbours( h_sph_list::const_iterator start, h_sph_list::const_iterator stop, const sph curr)
{
    vector<sph> * res = new vector<sph>;
    for (; start != stop; ++start)
        if (pnt_dist(*start, curr) < 3 * curr.w)
            res->push_back(*start);
    return res;
}

int GenMaxPacked(const int max_cnt, const float3 dim_len, h_sph_list & spheres)
{
    int curr_cnt = 0;
    int max_holost = (int)(dim_len.x * dim_len.y);
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
        vector<sph> * neigh = CollectNeighbours(spheres.begin(), spheres.begin() + curr_cnt, new_pnt);
        while (moves++ < max_moves)
        {
            sph over_sph = *(my_max_element(neigh.begin(), neigh.end(), dist_gt(new_pnt)) );
            if (is_overlapped(over_sph, new_pnt, max_overlapping)) {
                if (! move_pnt(dim_len, over_sph, new_pnt) )    {
                    neigh = CollectNeighbours(spheres.begin(), spheres.begin() + curr_cnt, new_pnt);
                }
            } else {
                add = true;
                break;
            }
        }
        delete neigh;
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

void SaveToFile(const vector<sph> & spheres, const char * filename)
{
    FILE * outFile = fopen(filename, "wb");
    
    for (int i = 0; i < spheres.size(); ++i)
    {
        fwrite(&(spheres[i]), sizeof(spheres[i].x), 4, outFile);
    }
    
    fclose(outFile);
    printf("%d spheres saved to file %s\n", spheres.size(), filename);
}

vector<sph> * LoadFromFile( const char * filename)
{
    FILE * inFile = fopen(filename, "rb");
    sph curr_pnt;
    vector<sph> * tmp = new vector<sph>();
    while(fread(&curr_pnt, sizeof(curr_pnt.x), 4, inFile))
        tmp->push_back( curr_pnt );
    return tmp;
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

vector<vector<sph> > * PercolatedClusters( const list<sph> & spheres, const float3 sz )
{
    DirGraph vg(spheres.size()); 
    list<sph>::const_iterator it1, it2;
    int curr_vertex, adj_vertex;
    for (curr_vertex = 0, it1 = spheres.begin(); curr_vertex < spheres.size(); ++curr_vertex, ++it1)
        for (adj_vertex = curr_vertex, it2 = it1; adj_vertex < spheres.size(); ++adj_vertex, ++it2)
            if (it2 != it1 && slightly_overlap(*it1, *it2, max_overlapping))
            {
                add_edge(curr_vertex, adj_vertex, vg);
            }
    
    std::vector<int> clusters(num_vertices(vg));
    int num = 
    connected_components(vg, make_iterator_property_map(clusters.begin(), get(vertex_index, vg), clusters[0]));

    set<int> * borders = new set<int>[6];
    // find all spheres on the borders
    int sph_idx = 0;
    for (it1 = spheres.begin(); it1 != spheres.end(); ++it1, ++sph_idx)
    {
        sph curr_sph = *it1;
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
        delete [] borders;
        return NULL;
    }
    vector<int> * perc_clusters = new vector<int>(borders[0].begin(), borders[0].end());
    vector<int>::iterator last_it = perc_clusters->end();
    vector<int> tmp(perc_clusters->size());
    for (int dim = 1; dim < 6; ++dim)
    {
        vector<int>::iterator it = set_intersection(perc_clusters->begin(), last_it, borders[dim].begin(), borders[dim].end(), tmp.begin());
        if (it - tmp.begin() == 0)
        {
            printf("Non perc [%d]\n", dim);
            delete [] borders;
            return NULL;
        }
        last_it = copy(tmp.begin(), it, perc_clusters->begin());
    }
    perc_clusters->resize(last_it-perc_clusters->begin());
    
    vector<vector <sph> > * res = new vector<vector <sph> >(perc_clusters->size());
    
    int clust_idx = 0;
    for (vector<int>::iterator it = perc_clusters->begin(); 
         it != perc_clusters->end(); ++it, ++clust_idx)
    {
        it1 = spheres.begin();
        for (vector<int>::iterator cl_it = clusters.begin(); cl_it != clusters.end(); ++cl_it, ++it1)
            if (*cl_it == *it)
                res->at(clust_idx).push_back(*it1);
    }
    printf("Percolated clusters:\n");
    for (vector<vector<sph> >::iterator it = res->begin(); it != res->end(); ++it)
        println(it->size());
    
    return res;
}

double Volume(double radius)
{
    return (4.0/3.0) * 3.14159 * (radius*radius*radius);
}

//template <class SphSequense>
double CalcVolume(const vector<sph> & spheres)
{
    double res = 0;
    vector<sph>::const_iterator it;
    it = spheres.begin();
    while(it != spheres.end()) 
    {
        res += Volume(it->w);
        ++it;
    }
    return res;
}

vector<sph> * RemovePoints( const vector<sph> & spheres, const float3 sz, const double min_volume )
{
    list<sph> tmp_sph(spheres.begin(), spheres.end());
    vector<vector<sph> > * clusters = PercolatedClusters(tmp_sph, sz);
    if (!clusters)
    {
        printf("Can\'t remove points!\n");
        return NULL;
    }
    // choose biggest cluster:
    double max_cluster_size = CalcVolume(clusters->at(0));
    int max_cluster_idx = 0;
    for (int i = 1; i < clusters->size(); ++i)
    {
        double vol = CalcVolume(clusters->at(i));
        if (vol > max_cluster_size)
        {
            max_cluster_size = vol;
            max_cluster_idx = i;
        }
    }
    if (max_cluster_size < min_volume)
    {
        printf("Percolated cluster too small\n");
        delete clusters;
        return NULL;
    }
    tmp_sph.resize(clusters->at(max_cluster_idx).size());
    copy(clusters->at(max_cluster_idx).begin(), clusters->at(max_cluster_idx).end(), tmp_sph.begin());
    delete clusters;
    printf("Start deleting operations\n");
    
    int holost_iter = 0;
    while(1)
    {
        printf("Go..\n");
        if (holost_iter > 3*tmp_sph.size())
        {
            printf("Cant achieve target volume\n");
            return NULL;
        }
        int del_idx = rand() % tmp_sph.size();
        list<sph>::iterator it = tmp_sph.begin();
        for (int i = 0; i<del_idx; ++i, ++it); // empty
        
        sph save_sph = *(it);
        tmp_sph.erase(it);
        clusters = PercolatedClusters(tmp_sph, sz);
        if (! clusters)
        {
            tmp_sph.push_back(save_sph);
            holost_iter++;
            continue;
        }
        // choose biggest cluster:
        double max_cluster_size = CalcVolume(clusters->at(0));
        int max_cluster_idx = 0;
        for (int i = 1; i < clusters->size(); ++i)
        {
            double vol = CalcVolume(clusters->at(i));
            if (vol > max_cluster_size)
            {
                max_cluster_size = vol;
                max_cluster_idx = i;
            }
        }
        printf("Biggest cluster have volume: %f\n", max_cluster_size);
        if (max_cluster_size < 0.95*min_volume)
        {
            tmp_sph.push_back(save_sph);
            holost_iter++;
            printf("Cluster too small\n");
            delete clusters;
            continue;
        }
        if (max_cluster_size < min_volume)
        {
            vector<sph> *res = new vector<sph>(clusters->at(max_cluster_idx).begin(), clusters->at(max_cluster_idx).end());
            delete clusters;
            return res;
        }
        tmp_sph.resize(clusters->at(max_cluster_idx).size());
        copy(clusters->at(max_cluster_idx).begin(), clusters->at(max_cluster_idx).end(), tmp_sph.begin());
        delete clusters;
        holost_iter = 0;
        printf("Current volume = %f, must be %f\n", max_cluster_size, min_volume);
    }
    // never come here
}

void
runTest( int argc, char** argv) 
{
    const float dim_sz = 100.0f;
    const double e_max = 0.3f;
    const double r = 2.0f;
    
    const float3 sz = make_float3(dim_sz,dim_sz,dim_sz);
    const double vol = sz.x * sz.y * sz.z;
    const double vol_sph = Volume(r);
    const int max_cnt =(int) (vol / vol_sph * (1.0-e_max));
    
    //cout << "Loading\n";
    //h_sph_list * spheres = LoadFromFile("max_40.dat");
    cout << "Start\n";
    h_sph_list spheres(max_cnt);
    int cnt = GenMaxPacked(max_cnt, sz, spheres);
    vector<sph> * v_spheres = new vector<sph>(spheres.begin(), spheres.begin() + cnt);
    SaveToFile(*v_spheres, "max_100_30.dat");
    double need_e = 0.9;
    double need_vol = vol*(1-need_e);
    vector<sph> * res = RemovePoints(*v_spheres, sz, need_vol);
    //h_sph_list h_spheres(spheres.begin(), spheres.begin() + cnt);
    //
    SaveToFile( *res, "res_100_90.dat");
    
    //cout << "Done. Points: " << cnt << " of " << max_cnt
    //<< ". E = " << (1 - vol_sph * cnt / vol) << endl;
}

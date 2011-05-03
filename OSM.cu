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
#include <utility>
#include <algorithm>

#include <percolated.h>

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
typedef adjacency_list< vecS, vecS, undirectedS > UndirGraph;
typedef graph_traits< UndirGraph >::vertex_descriptor Vertex;
typedef graph_traits< UndirGraph >::out_edge_iterator OutEdgeIter;
typedef graph_traits< UndirGraph >::edge_descriptor EdgeDescriptor;


// For logging
#include <log.h>

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

struct dist_less
{   
    sph curr;

    dist_less(sph c)    {   curr = c;   }

    __host__ __device__
    bool operator()(const sph & first, const sph & second) const
    {
        float l1 = overlapping(first.w, curr.w, pnt_dist(first, curr));
        float l2 = overlapping(second.w, curr.w, pnt_dist(second, curr));

        //printf("L1 = %f, L2 = %f\n", l1, l2);
        return l1 < l2;
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

ostream& operator<< (ostream& out, const float4& item )
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

set<sph, dist_gt> * CollectNeighbours( h_sph_list::const_iterator start, h_sph_list::const_iterator stop, const sph curr)
// returns sorted set of neibours
{
    
    set<sph, dist_gt> * res = new set<sph, dist_gt>( dist_gt(curr) );
    for (; start != stop; ++start)
        if (pnt_dist(*start, curr) < 3 * curr.w)
            res->insert(*start);
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
        bool maybe_add = false;
        int moves = 0;
        set<sph, dist_gt> * neigh = CollectNeighbours(spheres.begin(), spheres.begin() + curr_cnt, new_pnt);
        while (moves++ < max_moves)
        {
            if (neigh->empty())	{
                add = true;
                break;
            }
            sph over_sph = *(neigh->begin());
            if (is_overlapped(over_sph, new_pnt, max_overlapping)) {
                maybe_add = false;
                if (! move_pnt(dim_len, over_sph, new_pnt) )    {
                    delete neigh;
                    neigh = CollectNeighbours(spheres.begin(), spheres.begin() + curr_cnt, new_pnt); 
                } else {
                    set<sph, dist_gt> * tmp = new set<sph, dist_gt>(dist_gt(new_pnt));
                    tmp->insert(neigh->begin(), neigh->end());
                    delete neigh;
                    neigh = tmp;
                }
            } else {
                if (!maybe_add) {
                    delete neigh;
                    neigh = CollectNeighbours(spheres.begin(), spheres.begin() + curr_cnt, new_pnt);
                    maybe_add = true;
                    continue;
                }
                add = true;
                break;
            }
        }
        if (add) {

            // test
//            for (int i = 0; i < curr_cnt; ++i)
//                if (is_overlapped(spheres[i], new_pnt, max_overlapping) )
//                {
//                    printf("Error!\n");
//                }

            spheres[curr_cnt++] = new_pnt;
            holost = 0;
            cout << "Point #" << curr_cnt << " of " << max_cnt << ": " << new_pnt << endl;
        }
        delete neigh;
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

struct BorderIndex
{
    BorderIndex(const vector<sph> & spheres, const float3 sz):
    m_spheres(&spheres), m_sz(sz)
    {
    }
    
    vector<int> * operator()(int idx)
    {
        vector<int> * res = new vector<int>;
        sph curr_sph = m_spheres->at(idx);
        if (curr_sph.x-curr_sph.w < 0)
            res->push_back(0);
        if (curr_sph.x+curr_sph.w > m_sz.x)
            res->push_back(1);
        if (curr_sph.y-curr_sph.w < 0)
            res->push_back(2);
        if (curr_sph.y+curr_sph.w > m_sz.y)
       	    res->push_back(3);
        if (curr_sph.z-curr_sph.w < 0)
            res->push_back(4);
        if (curr_sph.z+curr_sph.w > m_sz.z)
       	    res->push_back(5);
       	return res;
    }
private:
    const vector<sph> * m_spheres;
    float3 m_sz;
};

//vector<vector<sph> > * PercolatedClusters( const list<sph> & spheres, const float3 sz )
//{
//    std::vector<int> clusters(num_vertices(vg));
//    int num = 
//    connected_components(vg, make_iterator_property_map(clusters.begin(), get(vertex_index, vg), clusters[0]));
//
//    set<int> * borders = new set<int>[6];
//    // find all spheres on the borders
//    // and save cluster numbers
//    int sph_idx = 0;
//    for (it1 = spheres.begin(); it1 != spheres.end(); ++it1, ++sph_idx)
//    {
//        sph curr_sph = *it1;
//        if (curr_sph.x-curr_sph.w < 0)
//            borders[0].insert(clusters[sph_idx]);
//        if (curr_sph.x+curr_sph.w > sz.x)
//            borders[1].insert(clusters[sph_idx]);
//        if (curr_sph.y-curr_sph.w < 0)
//            borders[2].insert(clusters[sph_idx]);
//        if (curr_sph.y+curr_sph.w > sz.y)
//       	    borders[3].insert(clusters[sph_idx]);
//        if (curr_sph.z-curr_sph.w < 0)
//            borders[4].insert(clusters[sph_idx]);
//        if (curr_sph.z+curr_sph.w > sz.z)
//       	    borders[5].insert(clusters[sph_idx]);
//    }
//    // find intersection between borders
//    int min_size = borders[0].size();
//    for (int dim = 1; dim < 6; ++dim)
//    {
//        if (borders[dim].size() < min_size)
//            min_size = borders[dim].size();
//    }
//    if (min_size == 0)
//    {
//        printf("Not percolate\n");
//        delete [] borders;
//        return NULL;
//    }
//    vector<int> * perc_clusters = new vector<int>(borders[0].begin(), borders[0].end());
//    vector<int>::iterator last_it = perc_clusters->end();
//    vector<int> tmp(perc_clusters->size());
//    for (int dim = 1; dim < 6; ++dim)
//    {
//        vector<int>::iterator it = set_intersection(perc_clusters->begin(), last_it, borders[dim].begin(), borders[dim].end(), tmp.begin());
//        if (it - tmp.begin() == 0)
//        {
//            printf("Non perc [%d]\n", dim);
//            delete [] borders;
//            return NULL;
//        }
//        last_it = copy(tmp.begin(), it, perc_clusters->begin());
//    }
//    perc_clusters->resize(last_it-perc_clusters->begin());
//    
//    vector<vector <sph> > * res = new vector<vector <sph> >(perc_clusters->size());
//    
//    int clust_idx = 0;
//    for (vector<int>::iterator it = perc_clusters->begin(); 
//         it != perc_clusters->end(); ++it, ++clust_idx)
//    {
//        it1 = spheres.begin();
//        for (vector<int>::iterator cl_it = clusters.begin(); cl_it != clusters.end(); ++cl_it, ++it1)
//            if (*cl_it == *it)
//                res->at(clust_idx).push_back(*it1);
//    }
//    printf("Percolated clusters:\n");
//    for (vector<vector<sph> >::iterator it = res->begin(); it != res->end(); ++it)
//        println(it->size());
//    
//    return res;
//}

double Volume(double radius)
{
    return (4.0/3.0) * 3.14159 * (radius*radius*radius);
}

double CalcVolume(const vector<sph> & spheres, const vector<int> & indicies)
{
    double res = 0;
    vector<int>::const_iterator it = indicies.begin();
    while(it != indicies.end()) 
    {
        res += Volume(spheres[*it].w);
        ++it;
    }
    return res;
}

struct Adjust
{
    Adjust(const vector<sph> & spheres, float max_over):
    m_max_overlapping(max_over),
    m_spheres(spheres)
    {
    }
    bool operator()(int idx1, int idx2)
    {
        return slightly_overlap(m_spheres[idx1], m_spheres[idx2], m_max_overlapping);
    }
private:
    vector<sph> m_spheres;
    float m_max_overlapping;
};

vector<sph> * ConvertIndToSph(const vector<sph> & spheres, const vector<int> & indicies)
{
    vector<sph> * res = new vector<sph>(indicies.size());
    for (int idx = 0; idx < indicies.size(); ++idx)
    {
        res->at(idx) = spheres[indicies[idx]];
    }
    return res;
}

vector<sph> * RemovePoints( const vector<sph> & spheres, const float3 sz, const double min_volume )
{
    Percolation<Adjust, BorderIndex > perc(spheres.size(), Adjust(spheres, max_overlapping), BorderIndex(spheres, sz));
    
    if (!perc.IsPercolated())
    {
        printf("Can\'t remove points!\n");
        return NULL;
    }
    
    // choose biggest cluster:
    double max_cluster_size = CalcVolume(spheres, perc.GetPercClusterItems(0));
    int max_cluster_idx = 0;
    for (int i = 1; i < perc.GetPercClustersCnt(); ++i)
    {
        double vol = CalcVolume(spheres, perc.GetPercClusterItems(i));
        if (vol > max_cluster_size)
        {
            max_cluster_size = vol;
            max_cluster_idx = i;
        }
    }
    if (max_cluster_size < min_volume)
    {
        printf("Percolated cluster too small\n");
        return NULL;
    }
    perc.OnlyPerc(max_cluster_idx);
    
    printf("Start deleting operations\n");
    
    while(1)
    {
        int del_idx = perc.TestRandomVertex();
        if (del_idx == -1)
        {
            log_it("Nope..");
            continue;
        } else if (del_idx == -2) {
            log_it("Spheres goes to end.");
            return NULL; // TODO: return final cluster
        }
        
        // choose biggest cluster:
        double max_cluster_size = CalcVolume(spheres, perc.GetPercClusterItems(0));
        int max_cluster_idx = 0;
        for (int i = 1; i < perc.GetPercClustersCnt(); ++i)
        {
            double vol = CalcVolume(spheres, perc.GetPercClusterItems(i));
            if (vol > max_cluster_size)
            {
                max_cluster_size = vol;
                max_cluster_idx = i;
            }
        }
        printf("Biggest cluster have volume: %f\n", max_cluster_size);
        if (max_cluster_size < 0.95*min_volume)
        {
            perc.RestoreState();
            printf("Cluster too small\n");
            continue;
        }
        if (max_cluster_size < min_volume)
        {
            vector<sph> *res = ConvertIndToSph(spheres, perc.GetPercClusterItems(max_cluster_idx));
            return res;
        }
        perc.OnlyPerc(max_cluster_idx);
        printf("Current volume = %f, must be %f\n", max_cluster_size, min_volume);
    }
    // never come here
}

void
runTest( int argc, char** argv) 
{
    const float dim_sz = 100.0f;
//    const double e_max = 0.3f;
    const double r = 2.0f;
    
    const float3 sz = make_float3(dim_sz,dim_sz,dim_sz);
    const double vol = sz.x * sz.y * sz.z;
    const double vol_sph = Volume(r);
//    const int max_cnt =(int) (vol / vol_sph * (1.0-e_max));
    
    cout << "Loading\n";
    vector<sph> * v_spheres = LoadFromFile("max_100_30_fast.dat");
//    cout << "Start\n";
//    h_sph_list spheres(max_cnt);
//    int cnt = GenMaxPacked(max_cnt, sz, spheres);

    // test
//    for (int idx1 = 0; idx1 < spheres.size(); ++idx1)
//    {
//        for (int idx2 = idx1+1; idx2 < spheres.size(); ++idx2)
//        {
//            if (is_overlapped(spheres[idx1], spheres[idx2], max_overlapping))
//            {
//                cout << "Test failed! 1:" << spheres[idx1] << " 2: " << spheres[idx2] << endl;
//               // return;
//            }
//        }
//    }
//    vector<sph> * v_spheres = new vector<sph>(spheres->begin(), spheres->end());
//    SaveToFile(*v_spheres, "max_100_30_fast.dat");
    double need_e = 0.9;
    double need_vol = vol*(1-need_e);
    vector<sph> * res = RemovePoints(*v_spheres, sz, need_vol);

//    for (int idx1 = 0; idx1 < spheres.size(); ++idx1)
//    {
//        for (int idx2 = idx1+1; idx2 < spheres.size(); ++idx2)
//        {
//            if (is_overlapped(res[0][idx1], res[0][idx2], max_overlapping))
//            {
//                cout << "Test failed! 1: " << res[0][idx1] << " 2: " << res[0][idx2] << endl;
//               // return;
//            }
//        }
//    }
    //h_sph_list h_spheres(spheres.begin(), spheres.begin() + cnt);
    //
    SaveToFile( *res, "res_100_90_fast.dat");
    
    //cout << "Done. Points: " << cnt << " of " << max_cnt
    //<< ". E = " << (1 - vol_sph * cnt / vol) << endl;
}

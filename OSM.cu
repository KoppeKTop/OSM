/*
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctime>
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
#include <thrust/device_ptr.h>

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


#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)

inline cudaError cutilDeviceSynchronize()
{
#if CUDART_VERSION >= 4000
	return cudaDeviceSynchronize();
#else
	return cudaThreadSynchronize();
#endif
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) );
        exit(-1);
    }
}

inline void __cudaSafeThreadSync( const char *file, const int line )
{
    cudaError err = cutilDeviceSynchronize();
    if ( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaDeviceSynchronize() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) );
        exit(-1);
    }
}


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

float GetSphereRadius(float ret_r = -1)
{
    static float r = 0;
    if (ret_r > 0)
    {
        r = ret_r;
    }
    return r;
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
    const int cnt = stop - start;
// #pragma omp parallel for shared(cnt, res, curr)
    for (int idx = 0; idx < cnt; idx++)
    {
        sph curr_nei = *(start+idx);
        if (pnt_dist(curr_nei, curr) < 3 * curr.w)
        {
//            #pragma omp single
            res->insert(curr_nei);
        }
    }
    return res;
}

set<sph, dist_gt> * CollectNeighbours( sph * d_sph, h_sph_list & spheres, int cnt, const sph curr_sph)
// GPU version
{
//    printf("Start to CollectNeighbours\n");
    const int THREADS_PER_BLOCK = 256;
    if (cnt < THREADS_PER_BLOCK)
    {
        return CollectNeighbours(spheres.begin(), spheres.begin()+cnt, curr_sph);
    }
    int * d_results_idx = NULL;
    int * d_res_cnt = NULL;
    
    const int max_results = 1000;
    const int res_sz = max_results * sizeof(*d_results_idx);
    cutilSafeCall(cudaMalloc((void **) &d_results_idx, res_sz));
    thrust::device_ptr<int> t_results_idx(d_results_idx);
    cutilSafeCall(cudaMemset(d_results_idx, 0, res_sz));
    cutilSafeCall(cudaMalloc((void **) &d_res_cnt, sizeof(*d_res_cnt)) );
    thrust::device_ptr<int> t_res_cnt(d_res_cnt);
    cutilSafeCall(cudaMemset(d_res_cnt, 0, sizeof(*d_res_cnt)));
    
//    cout << "Curr d_res_cnt: " << t_res_cnt[0] << endl;
//    cout << "Curr d_curr_sph: " << t_curr_sph[0] << endl;
//    cout << "Curr sph: " << curr_sph << endl;
//    cout << "Curr d_cnt: " << t_cnt[0] << endl;
    
    
    int block = THREADS_PER_BLOCK;
    int grid = cnt / THREADS_PER_BLOCK;
    if (cnt % THREADS_PER_BLOCK != 0) grid += 1;
//    cout << grid << " " << cnt << " " << cnt / THREADS_PER_BLOCK << endl;
//    printf("Start GPU\n");
//    sph * d_sph = thrust::raw_pointer_cast(&d_spheres[0]);
//    printf("d_sph = %p\n", d_sph);
    nei_list<<<dim3(grid,1), dim3(block, 1, 1)>>>(d_sph, curr_sph, d_results_idx, d_res_cnt, cnt);
    cutilSafeCall(cudaThreadSynchronize());
    cutilSafeCall(cudaGetLastError());
//    printf("GPU done\n");
    
    int results_cnt = t_res_cnt[0];
//    cout << "Res cnt = " << results_cnt << endl;
//    cutilSafeCall(cudaMemcpy(&results_cnt, d_res_cnt, sizeof(int), cudaMemcpyDeviceToHost));
//    cutilSafeCall(cudaThreadSynchronize());
    
    set<sph, dist_gt> * res = new set<sph, dist_gt>( dist_gt(curr_sph) );
    if (results_cnt != 0)   
    {
        int * results_idx = new int[results_cnt];
        try
        {
            thrust::copy(t_results_idx, t_results_idx+results_cnt, results_idx);
        }
        catch (thrust::system::system_error err)
        {
            cout << "Res_cnt = " << results_cnt << endl;
            cout << results_idx << endl;
            exit(100);
        }
//        cutilSafeCall(cudaMemcpy(results_idx, d_results_idx, results_cnt*sizeof(int), cudaMemcpyDeviceToHost));
//        cutilSafeCall(cudaThreadSynchronize());
        
//        printf("%d results copied\n", results_cnt);
        
        for (int idx = 0; idx < results_cnt; idx++)
        {
//            printf("sphere #%d\n", results_idx[idx]);
            res->insert(spheres[ results_idx[idx] ]);
        }
        
        delete [] results_idx;
    }
    cudaFree(d_results_idx);
    cudaFree(d_res_cnt);
    
    //printf("CollectNeighbours end\n");
    
//    set<sph, dist_gt> * tmp = CollectNeighbours(spheres.begin(), spheres.begin()+cnt, curr_sph);
//    cout << "CollectNeighbours results:\n";
//    set<sph, dist_gt>::const_iterator pnt;
//    cout << "CPU (" << tmp->size() << ")\n";
//    for (pnt = tmp->begin(); pnt != tmp->end(); ++pnt)
//    {
//        cout << *pnt << endl;
//    }
//    cout << "GPU (" << res->size() << ")\n";
//    for (pnt = res->begin(); pnt != res->end(); ++pnt)
//    {
//        cout << *pnt << endl;
//    }
//    delete tmp;
    return res;
}


int GenMaxPacked(const int max_cnt, const float3 dim_len, sph * spheres)
{
    h_sph_list h_spheres(max_cnt);
    int curr_cnt = 0;
    int max_holost = (int)(dim_len.x);
    int holost = 0;
    
    const int max_moves = 20;
    while (curr_cnt < max_cnt && holost++ < max_holost)
    {
        sph new_pnt = GenRndPoint(dim_len);
        //printf("New point (%i of %i): (%f, %f, %f)\n", curr_cnt, max_cnt, new_pnt.x, new_pnt.y, new_pnt.z);
        if (curr_cnt == 0) {
            h_spheres[curr_cnt] = new_pnt;
            cutilSafeCall(cudaMemcpy(spheres+curr_cnt, &new_pnt, sizeof(sph), cudaMemcpyHostToDevice));
            curr_cnt ++;
            holost = 0;
            continue;
        }
        bool add = false;
        bool maybe_add = false;
        int moves = 0;
        set<sph, dist_gt> * neigh = CollectNeighbours(spheres, h_spheres, curr_cnt, new_pnt);
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
                    neigh = CollectNeighbours(spheres, h_spheres, curr_cnt, new_pnt);
                } else {
                    set<sph, dist_gt> * tmp = new set<sph, dist_gt>(dist_gt(new_pnt));
                    tmp->insert(neigh->begin(), neigh->end());
                    delete neigh;
                    neigh = tmp;
                    holost++;
                    moves = 0;
                }
            } else {
                if (!maybe_add) {
                    delete neigh;
                    neigh = CollectNeighbours(spheres, h_spheres, curr_cnt, new_pnt);
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
            h_spheres[curr_cnt] = new_pnt;
            cutilSafeCall(cudaMemcpy(spheres+curr_cnt, &new_pnt, sizeof(sph), cudaMemcpyHostToDevice));
            curr_cnt ++;
            holost = 0;
            if (curr_cnt % (max_cnt / 10) == 0)
            {
                time_t time_since_epoch;
                time( &time_since_epoch );
                tm *current_time = localtime( &time_since_epoch );
                
                cout << "Point #" << curr_cnt << " of " << max_cnt << ": " << asctime( current_time );
            }
        }
        delete neigh;
    }
    printf("Generated %d points\n", curr_cnt);
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

UndirGraph * ConvertSphToGraph(const vector<sph> & spheres)
{
    d_sph_list d_sph(spheres.begin(), spheres.end());
    
    const int THREADS_CNT = 256;
    int grid_dim = spheres.size()/THREADS_CNT;
    if (spheres.size()/THREADS_CNT != 0) grid_dim += 1;
    dim3 grid(grid_dim, 1);
    dim3 block(THREADS_CNT, 1, 1);
    
    const int res_cnt = 100 + 1;
    
    int * d_results = NULL;
    int * h_results = new int[res_cnt];
    sph * d_spheres_ptr = thrust::raw_pointer_cast(&d_sph[0]);
    const size_t res_sz = res_cnt * sizeof(int); // max 100 results + 0th element – res_cnt
    
    cudaMalloc((void **) &d_results, res_sz);
    
    UndirGraph * vg = new UndirGraph(spheres.size());
    int curr_vertex;
    for (curr_vertex = 0; curr_vertex < spheres.size(); ++curr_vertex)
    {
        cudaMemset(d_results, 0, res_sz);
        slight_nei_list<<<grid, block>>>(d_spheres_ptr, curr_vertex, spheres.size(), max_overlapping, d_results);
        cudaThreadSynchronize();
        cudaMemcpy(h_results, d_results, res_sz, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        if (h_results[0] >= res_cnt)
        {
            printf("Too much results!\n");
            exit(199);
        }
        for (int adj_vertex = 0; adj_vertex < h_results[0]; ++adj_vertex)
        {
            add_edge(curr_vertex, h_results[adj_vertex+1], *vg);
        }
    }
    return vg;
}

vector<sph> * RemovePoints( const vector<sph> & spheres, const float3 sz, const double min_volume )
{
    printf("Start to convert points... ");
    UndirGraph * vg = ConvertSphToGraph(spheres);
    printf("Done\n");
    Percolation<Adjust, BorderIndex > perc(*vg, spheres.size(), Adjust(spheres, max_overlapping), BorderIndex(spheres, sz));
    delete vg;
    
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
        // +- 1% of min_volume is acceptable
        if (max_cluster_size < 0.99*min_volume)
        {
            perc.RestoreState();
            printf("Cluster too small\n");
            continue;
        }
        if (max_cluster_size < 1.01*min_volume)
        {
            vector<sph> *res = ConvertIndToSph(spheres, perc.GetPercClusterItems(max_cluster_idx));
            return res;
        }
        perc.StopSaving();
        perc.OnlyPerc(max_cluster_idx);
        printf("Current volume = %f, must be %f\n", max_cluster_size, min_volume);
    }
    // never come here
}

void
runTest( int argc, char** argv) 
{
    const float dim_sz = 500.0f;
    const double e_max = 0.3f;
    const float r = 3.0;
    GetSphereRadius(r);
    
    const float3 sz = make_float3(dim_sz,dim_sz,dim_sz);
    const double vol = sz.x * sz.y * sz.z;
    const double vol_sph = Volume(r);
    const int max_cnt =(int) (vol / vol_sph * (1.0-e_max));
    
    unsigned pref_gpu = 0;
    if (argc > 1)
    {
        int res = sscanf(argv[1], "%u", &pref_gpu);
        if (res == EOF)
        {
            printf("Using: %s [gpu_number]", argv[0]);
            exit(10);
        }
    }
    cutilSafeCall(cudaSetDevice((int)pref_gpu));
    cout << "GPU#" << pref_gpu << endl;

    cout << "Start\n";
    
//    cout << "Loading\n";
//    vector<sph> * v_spheres = LoadFromFile("max_500_r25_gpu.dat");
//    d_sph_list d_spheres(max_cnt);


    sph * d_spheres_raw = NULL;
    cutilSafeCall(cudaMalloc((void **) &d_spheres_raw, max_cnt*sizeof(sph)));
    cutilSafeCall(cudaMemset(d_spheres_raw, 0, max_cnt*sizeof(sph)));
    int cnt = GenMaxPacked(max_cnt, sz, d_spheres_raw);
    thrust::device_ptr<sph> d_spheres(d_spheres_raw);
    h_sph_list spheres(d_spheres, d_spheres + cnt);
    cudaFree(d_spheres_raw);
//    cout << "Test passed. Saving\n";
    vector<sph> * v_spheres = new vector<sph>(spheres.begin(), spheres.end());
    SaveToFile(*v_spheres, "max_500_r30_gpu_2.dat");
    
    // test
//    cout << "Start test\n";
//    for (int idx1 = 0; idx1 < spheres.size(); ++idx1)
//    {
//        for (int idx2 = idx1+1; idx2 < spheres.size(); ++idx2)
//        {
//            if (is_overlapped(spheres[idx1], spheres[idx2], max_overlapping))
//            {
//                cout << "Test failed! 1:" << spheres[idx1] << " 2: " << spheres[idx2] << endl;
//                exit(200);
//            }
//        }
//    }
    double need_e = 1.0-0.1/2.2;
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
    SaveToFile( *res, "res_500_r30_90_gpu_2.dat");
    delete res;
    
    //cout << "Done. Points: " << cnt << " of " << max_cnt
    //<< ". E = " << (1 - vol_sph * cnt / vol) << endl;
}

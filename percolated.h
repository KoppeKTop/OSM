#ifndef PERCOLATION_H_
#define PERCOLATION_H_


#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <utility>
#include <set>
#include <algorithm>

#include <log.h>

// For BGL connection algorithm
#include <boost/config.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>

// from BGL book p 201
using namespace boost;
typedef adjacency_list< vecS, vecS, undirectedS > UndirGraph;
typedef graph_traits< UndirGraph >::vertex_descriptor Vertex;
typedef graph_traits< UndirGraph >::out_edge_iterator OutEdgeIter;
typedef graph_traits< UndirGraph >::adjacency_iterator AdjIter;
typedef graph_traits< UndirGraph >::edge_descriptor EdgeDescriptor;

typedef float4 sph;

typedef map<Vertex, vector<Vertex> > EdgesMap;

template < typename BinaryOperationFunctor, typename Functor>
class Percolation
{
public:
    Percolation( const UndirGraph & gr,
                 int sph_cnt, 
                 BinaryOperationFunctor is_adjust, 
                 Functor get_border_idx):
                 perc_sph(NULL), _is_saving(false)
    {
//        vg = new UndirGraph(sph_cnt);
//        int curr_vertex, adj_vertex;
//        for (curr_vertex = 0; curr_vertex < sph_cnt; ++curr_vertex)
//            for (adj_vertex = curr_vertex+1; adj_vertex < sph_cnt; ++adj_vertex)
//                if (is_adjust(curr_vertex, adj_vertex) )
//                {
//                    add_edge(curr_vertex, adj_vertex, *vg);
//                }
        vg = new UndirGraph(gr); 
        
        vertex_vector.resize(sph_cnt);
        for (int i = 0; i < sph_cnt; ++i)
        {
            vertex_vector[i] = i;
        }
        random_shuffle(vertex_vector.begin(), vertex_vector.end());
        
        border_vertex = new vector<int>[6];
        for (int i = 0; i < sph_cnt; ++i)
        {
            vector<int> * border_idx = get_border_idx(i);
            if (border_idx)
            {
                for (int i = 0; i < border_idx->size(); ++i)
                {
                    border_vertex[border_idx->at(i)].push_back(i);
                }
                delete border_idx;
            }
        }
    }
    
    ~Percolation()
    {
        delete vg;
//        delete old_vg;
        delete [] border_vertex;
        delete perc_sph;
    }
    
    // permutation procedures:
    void SaveState()
    {
        _is_saving = true;
        _saving_edges.clear();
//        log_it("Save state");
//        if (old_vg) delete old_vg;
//        old_vg = new UndirGraph(*vg);
//        log_it("Done");
    }
    
    void StopSaving()
    {
        _is_saving = false;
        _saving_edges.clear();
    }
    
    void RestoreState()
    {
        _is_saving = false;
        log_it("Restore state");
        EdgesMap::iterator in_it;
        vector<Vertex>::iterator out_it;
        for (in_it = _saving_edges.begin(); in_it != _saving_edges.end(); ++in_it)
        {
            for (out_it = in_it->second.begin(); out_it != in_it->second.end(); ++out_it)
            {
                add_edge(in_it->first, *out_it, *vg);
            }
        }
        _saving_edges.clear();
//        *vg = *old_vg;
        log_it("Done");
    }
    
    void DeleteVertex(int vertex_idx)
    {
        if (deleted_vertexes.find(vertex_idx) != deleted_vertexes.end())
        {
            log_it("Delete again!");
            exit(0);
        }
        deleted_vertexes.insert(vertex_idx);
        Vertex v = vertex( vertex_idx, *vg);
        if (_is_saving)
        {
            AdjIter adj_begin, adj_end;
            tie(adj_begin, adj_end) = adjacent_vertices(v, *vg);
            _saving_edges[v] = vector<Vertex>(adj_begin, adj_end);
        }
        clear_vertex(v, *vg);
    }
    
    // main proc
    bool IsPercolated()
    {
        log_it("Calc connectivity…");
        std::vector<int> clusters(num_vertices(*vg));
        int num = 
        connected_components(*vg, make_iterator_property_map(clusters.begin(), get(vertex_index, *vg), clusters[0]));
        log_it("Done");
    
        log_it("Find percolated clusters…");
        set<int> * borders_clusters = new set<int>[6];
        // find all spheres on the borders
        // and save cluster numbers
        for (int border_idx = 0; border_idx < 6; ++border_idx)
        {
            for (vector<int>::const_iterator it = border_vertex[border_idx].begin(); 
                 it != border_vertex[border_idx].end(); ++it)
            {
                borders_clusters[border_idx].insert(clusters[*it]);
            }
        }
        // find intersection between borders
        int min_size = borders_clusters[0].size();
        for (int dim = 1; dim < 6; ++dim)
        {
            if (borders_clusters[dim].size() < min_size)
                min_size = borders_clusters[dim].size();
        }
        if (min_size == 0)
        {
            printf("Not percolate\n");
            delete [] borders_clusters;
            return false;
        }
        vector<int> perc_clusters(borders_clusters[0].begin(), borders_clusters[0].end());
        vector<int>::iterator last_it = perc_clusters.end();
        vector<int> tmp(perc_clusters.size());
        for (int dim = 1; dim < 6; ++dim)
        {
            vector<int>::iterator it = set_intersection(perc_clusters.begin(), last_it, 
                                                        borders_clusters[dim].begin(), borders_clusters[dim].end(), tmp.begin());
            if (it - tmp.begin() == 0)
            {
                printf("Non perc [%d]\n", dim);
                delete [] borders_clusters;
                return false;
            }
            last_it = copy(tmp.begin(), it, perc_clusters.begin());
        }
        perc_clusters.resize(last_it-perc_clusters.begin());
        log_it("Done");
        
        log_it("Collect percolation clusters..");
        //if (perc_sph) delete perc_sph;
        perc_sph = new vector<vector <int> >(perc_clusters.size());
        
        int clust_idx = 0;
        for (vector<int>::iterator it = perc_clusters.begin(); 
             it != perc_clusters.end(); ++it, ++clust_idx)
        {
            int sph_idx = 0;
            for (vector<int>::iterator cl_it = clusters.begin(); cl_it != clusters.end(); ++cl_it, ++sph_idx)
                if (*cl_it == *it)
                    perc_sph->at(clust_idx).push_back(sph_idx);
            printf("Cluster %d: %d\n", clust_idx, perc_sph->at(clust_idx).size());
        }
        return true;
    }
    
    // wrapper to it
    bool IsPercolatedWithout(int vertex_idx, bool restore_if_yes = false)
    {
        SaveState();
        DeleteVertex( vertex_idx);
        bool res = IsPercolated();
        if (res && restore_if_yes || !res)  {
            RestoreState();
        }
        return res;
    }
    
    int GetRandomVertex()
    {
        if (vertex_vector.size() == 0)
        {
            return -1;
        }   else    {
            int old_sz  = vertex_vector.size();
            printf("vertex_vector.size() = %d, now = ", vertex_vector.size());
            int res = vertex_vector[0];
            vertex_vector.erase(vertex_vector.begin() );
            printf("%d \n", vertex_vector.size());
            printf("Popped: %d, last now = %d\n", res, vertex_vector[0]);
            if (old_sz - vertex_vector.size() != 1)
            {
                printf("removed strange\n");
                exit(0);
            }
            return res;
        }
    }
    
    int TestRandomVertex()
    // the most complex function
    // delete next random vertex
    // test percolation
    // if there is no more untested vertexes – returns -2
    // if graph percolates returns it's index of deleted vertex
    // else restores vertex and returns -1
    {
        int vertex_idx = GetRandomVertex();
        if (vertex_idx == -1)
        {
            vertex_idx = -2;
        }   else if (! IsPercolatedWithout(vertex_idx) )   {
            vertex_idx = -1;
        }
        return vertex_idx;
    }
    
    void OnlyPerc(int perc_cluster_idx)
    {
        int vert_cnt = num_vertices(*vg);
        int j = 0;
        for (int i = 0; i < vert_cnt; ++i)
        {
            if (perc_sph->at(perc_cluster_idx)[j] != i)
            {
                if (deleted_vertexes.find(i) != deleted_vertexes.end() )
                // already deleted
                    continue;
                remove(vertex_vector.begin(), vertex_vector.end(), i);
                DeleteVertex(i);
            }
            else 
            {
                j++;
            }
        }
        prev_perc_sph = perc_sph->at(perc_cluster_idx);
        delete perc_sph;
        perc_sph = NULL;
    }
    
    int GetPercClustersCnt()
    {
        return perc_sph->size();
    }
    
    const vector<int> & GetPercClusterItems(int idx)
    {
        return perc_sph->at(idx);
    }
    
private:
    UndirGraph * vg;
//    UndirGraph * old_vg;
    vector<int> vertex_vector;
    
    vector<int> * border_vertex;
    vector<vector <int> > * perc_sph;
    vector <int> prev_perc_sph;
    
    set<int> deleted_vertexes;
    
    bool _is_saving;
    EdgesMap _saving_edges;
};

#endif


////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
int is_overlapped(float4 pnt1, float4 pnt2, float max_overlapping);

void overlap_list(float4 * spheres, float4 curr_sph, int * results, float max_overlapping, int curr_cnt)
{
    for(int idx = 0; idx < curr_cnt; ++idx)
    {
        float4 cmp_sph = spheres[idx];
        if (is_overlapped(curr_sph, cmp_sph, max_overlapping))
        {
            results[0]++;
            results[results[0]] = idx;
        }
    }
}
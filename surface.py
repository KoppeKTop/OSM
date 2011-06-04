#!/usr/bin/env python

import numpy
import random, sys
import math

import pp

def dist(p1, p2):
    return sqrt( sum([(p1-p2)**2 for dim in xrange(len(p1)-1)]) )

EPS = 1e-8

def is_overlapping(pnts, p):
    tmp_p = p.copy()
    tmp_p[-1] *= -1
    deltas_2 = (pnts-tmp_p)**2
    distances = numpy.add.reduce(deltas_2[:,:-1], axis=1)
    differs = distances - deltas_2[:,-1]
#    print min(differs)
    
    return numpy.any(differs < -EPS)

def __has_neigbours(pnts, p):
    job_server = pp.Server(ppservers=())
    parts = 128
    jobs = []
    start= 0
    end = pnts.shape[0]
    step = (end - start) / parts + 1
    for index in xrange(parts):
        starti = start+index*step
        endi = starti + step
        jobs.append(job_server.submit(has_neigbours_pp, (pnts, p, starti, endi), (), ("numpy",)))
    for job in jobs:
        result = job()
        if result:
            return True
    return False
    
#    return [pnt for pnt in points if dist(pnt, rand_pnt) < (pnt[3] + rand_pnt[3])]
#
#def get_sph_s(radius):
#    return 4*pi*radius**2

def get_total_surface(points):
    return 4 * math.pi * numpy.sum(points[:,3]**2)

def get_total_volume(points):
    return (4./3) * math.pi * numpy.sum(points[:,3]**3)

def mesopore_surfase_area(points, r_test):
    random.seed()
    cnt = points.shape[0]
    total_surfase = get_total_surface(points) * 10**(-18) # in m^2
    miss = 0
    cycles = 0.0
    old_miss = 0.0
    while 1:
        old_miss += miss
        miss = 0
        for i in xrange(5000):
            if (i % 100 == 0):
                print "Iter", i, ", Miss: ", miss
            tetta = 2*math.pi*random.random()
            lam = 2*math.pi*random.random()
            # translate to Dekart coodinates
            real_pnt = points[random.randint(0, points.shape[0]-1)]
            r = real_pnt[3]+r_test
            x = r*math.sin(tetta)*math.cos(lam) + real_pnt[0]
            y = r*math.sin(tetta)*math.sin(lam) + real_pnt[1]
            z = r*math.cos(tetta) + real_pnt[2]
            new_pnt = numpy.array([x, y, z, r_test])      
            if is_overlapping(points, new_pnt):
                miss += 1
        print cycles+1, ":" , miss
        if cycles != 0:
            if abs(miss-old_miss/cycles)/float(max(miss, old_miss/cycles)) < 0.01:
                break
            print old_miss/cycles
        cycles += 1
    return (1 - old_miss/cycles/5000) * total_surfase

def load(filename):
    data = numpy.fromfile(filename, dtype=numpy.float32)
    data.shape = (-1, 4)
    return data
    

def main():
    if len(sys.argv) < 2:
        print "Usage: %s points_file" % (sys.argv[0],)
        return
    
    r_test = 0.227
    density = 2.2
    
    print "Load"
    points = load(sys.argv[1])
    
    print "Calc area"
    S = mesopore_surfase_area(points, r_test) # in m^2
    Vol = get_total_volume(points)
    total_volume = Vol * 10**(-21) # in ml
    total_mass = total_volume * density # in grams
    Syd = S/total_mass
    print "Result: ", Syd
    
if __name__ == "__main__":
    main()
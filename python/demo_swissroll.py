#!/usr/bin/python

import Manifold
import ManifoldVTK
import Utils
import operator
import sys
import numpy

coords = Utils.read_point_file("./datasets/swissroll.txt")

embedded_coords, mapping = Manifold.do_embedding(coords,k=10)

coords = coords.take(mapping,axis=0)
nb_points = len(coords)
    
# sorting
sorted_index =  list( index for index, item in sorted( enumerate( embedded_coords[:,0] ),
                                                       key=operator.itemgetter(1)
                                                       ) )

colors = [(255,255,255) for i in range(0,nb_points)]
for i in range(0,nb_points):
    if i < nb_points / 2:
        f = float(i) / float(nb_points / 2)
        color = ( 0, f*255, 255 - f*255 )
        #color = ( 0, 0, 255 )
    else:
        f = float(i - nb_points / 2) / float(nb_points / 2)
        color = ( f*255, 255 - f*255, 0 )
        #color = ( 255, 0, 0 )
    colors[sorted_index[i]] = color
    #colors[i] = color


actor1 = ManifoldVTK.vtk_point_cloud( coords, colors, 4 )
actor2 = ManifoldVTK.vtk_point_cloud( embedded_coords[:,[0,1]], colors, 4 )
actor3 = ManifoldVTK.vtk_point_cloud( embedded_coords, colors, 4 )

ManifoldVTK.vtk_Nviews( [actor1, actor2, actor3] )

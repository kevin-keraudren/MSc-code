# ref http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.119.2686&rep=rep1&type=pdf

import numpy as np

def procrustes_analysis(shapes):

    nb_shapes = len(shapes)

    # center all shapes on zero
    for i in range(nb_shapes):
        # center on zero
        shapes[i] -= shapes[i].mean(axis=0)
        #resize
        norm = np.linalg.norm(shapes[i])
        shapes[i] /= norm

    # rotate to align with the first shape
    corr = np.zeros((2,2))
    for i in range(1,nb_shapes):
        for j in range(len(shapes[0])):
            corr += np.dot(shapes[0][j].T, shapes[i][j])
        (u,s,v) = np.linalg.svd(corr)
        rot = np.dot(v,u.T)
        for j in range(len(shapes[0])):
            shapes[i] = np.dot(shapes[i],rot)       

    return shapes

    





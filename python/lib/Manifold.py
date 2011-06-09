"""
Manifold Learning using Landmark Isomap

This module implements the Landmark Isomap method, also known as Nystrom Isomap.

The steps are as follows:
given a set of points,
 - build a neighbourhood graph (typically 5-neighbourhood)
 - select the largest connected component
 - select landmarks
 - build the distance matrix of all points to the landmarks using the
 neighbourhood graph to compute the geodesic distance
 - run the Nystrom approximation to perform dimensionality reduction.
 
"""

import multiprocessing
from scipy.spatial import KDTree
import numpy
import networkx
from math import sqrt
from scipy import stats
import cv # OpenCV

def _search_kdtree( p ):
    """" Needs to be pickable to be used by a multiprocessing.Pool() """
    global tree
    global _k    
    return  tree.query(p,_k)
    
def kdTreeSearch( coords, k, parallel=True ):
    """
    Build a k-neighbourhood graph using a scipy.spatial.KDTree
    
    Parameters
    ----------
    coords : numpy array, each row is a point
    k : number of neighbours to search
    parallel: boolean, whether to use multiprocessing
    
    Returns
    -------
    edges : numpy array of edges, each row is of the form
            [ point_1, point_2, distance ]    
    """
    global tree
    global _k
    tree = KDTree( coords )
    _k = k                     

    if parallel:
        pool = multiprocessing.Pool()
        results = pool.map( _search_kdtree, coords )
    else:
        results = map( _search_kdtree, coords )
        
    edges = []
    for i in range( len(coords) ):
        (dist,res) = results[i]
        for n in range(0,k):
            edges.append([i,res[n],dist[n]])
      
    return edges

def _search_spilltree(i):
    """" Needs to be pickable to be used by a multiprocessing.Pool() """
    global tree
    global _k
    res = cv.CreateMat(1, _k, cv.CV_32SC1)
    dist = cv.CreateMat(1, _k, cv.CV_64FC1)
    cv.FindFeatures(tree,cv_coords[i,:],res,dist,_k,20)
    edges = []
    for n in range(0,_k):
        edges.append([i,int(res[0,n]),dist[0,n]])
    return edges
    
def spillTreeSearch( coords, k, parallel=True ):
    """
    Build a k-neighbourhood graph using an OpenCV spilltree
    Patch required:
    https://code.ros.org/trac/opencv/ticket/1082
    
    Parameters
    ----------
    coords : numpy array, each row is a point
    k : number of neighbours to search
    parallel: boolean, whether to use multiprocessing
    
    Returns
    -------
    edges : numpy array of edges, each row is of the form
            [ point_1, point_2, distance ]    
    """    
    global tree
    global _k
    global cv_coords
    nb_points = len(coords)
    cv_coords = cv.fromarray(coords)
    tree = cv.CreateSpillTree(cv_coords,100,0.7,0.1)
    _k = k

    if parallel:
        pool = multiprocessing.Pool()
        results = pool.map(_search_spilltree, range(cv_coords.rows))
        edges = [e for es in results for e in es]
        
    else:
        res = cv.CreateMat(nb_points, k, cv.CV_32SC1)
        dist = cv.CreateMat(nb_points, k, cv.CV_64FC1)
        cv.FindFeatures(tree,cv_coords,res,dist,k)
        edges = []
        for i in range(0,nb_points):
            for n in range(0,k):
                edges.append([i,int(res[i,n]),dist[i,n]])
            
    return edges
        
def build_neighbourhood_graph(edges):
    """
    Given edges and weights, returns a networkx.Graph()
    
    Parameters
    ----------
    edges : numpy array of edges, each row is of the form
            [ point_1, point_2, distance ]        
    
    Returns
    -------
    graph : a weighted undirected networkx.Graph()
    """
    graph = networkx.Graph()
    edges = map( lambda e: [e[0],e[1],{'weight':e[2]}], edges)
    graph.add_edges_from(edges)
    return graph

def select_largest_connected_component(graph):
    """
    Given a networkx.Graph(), returns the largest connected component
    
    Parameters
    ----------
    graph : a weighted undirected networkx.Graph()
    
    Returns
    -------
    new_graph : a weighted undirected networkx.Graph()
    """    
    cc_list = networkx.connected_components(graph)
    return graph.subgraph(cc_list[0]) # they are correctly sorted

def clean_graph(graph):
    """
    Given a networkx.Graph(), returns a new graph with the longest edges
    removed (cutting at the 95th percentile)
    
    Parameters
    ----------
    graph : a weighted undirected networkx.Graph()
    
    Returns
    -------
    new_graph : a weighted undirected networkx.Graph()
    """        
    weights = [ graph[e[0]][e[1]]['weight'] for e in graph.edges()]
    treshold = stats.scoreatpercentile(weights, 95)
    new_edges = []
    for e in graph.edges():
        if graph[e[0]][e[1]]['weight'] < treshold:
            new_edges.append([e[0],e[1],{'weight':graph[e[0]][e[1]]['weight']}])
    new_graph = networkx.Graph()
    new_graph.add_edges_from(new_edges)
    return new_graph

def _do_one_landmark(n):
    """" Needs to be pickable to be used by a multiprocessing.Pool() """
    global _graph
    d = networkx.shortest_path_length(_graph,n,target=None, weighted=True)
    d = d.values()
    return map(lambda x: x*x, d)
    
def distance_matrix(graph,nb_landmarks):
    """
    Compute the matrix of geodesic distances to the landmark points
    
    Parameters
    ----------
    graph : a weighted undirected networkx.Graph()
    nb_landmarks : number of landmarks, the first points in the graph will be
                   used
    
    Returns
    -------
    distances : a numpy array corresponding to the distance matrix,
                with as many columns as there are landmarks
    """   
    global _graph
    _graph = graph
    nb_nodes = len(graph)
    distances = []
    if nb_landmarks > nb_nodes:
        print "**ERROR** nb_landmarks > nb_nodes :", nb_landmarks, nb_nodes
    nodes = graph.nodes()

    pool = multiprocessing.Pool()
    distances = pool.map(_do_one_landmark, nodes[0:nb_landmarks])
    
    distances = numpy.array(distances)
    return distances.T

def landmark_isomap(C):
    """
    Landmark isomap dimensionality reduction method
    This is the version we implemented in
    http://www.doc.ic.ac.uk/~kpk09/facecrumbs.html
    
    Parameters
    ----------
    C : matrix of geodesic distances to the landmarks
    
    Returns
    -------
    embedded_coords : embedded coordinates of the original points
    """      
    (nb_points, nb_landmarks) = map(float, C.shape)
    D = C[0:nb_landmarks,0:nb_landmarks]
    H= numpy.eye(nb_landmarks) - 1/nb_landmarks*numpy.ones((nb_landmarks,nb_landmarks))
    W = -0.5*numpy.dot(numpy.dot(H,D),H) # those are numpy arrays
    Ew,Uw = numpy.linalg.eig(W)
    Uw = Uw.T
    ind = Ew.argsort()
    ind = ind[::-1] # we want decreasing order
    Ew = Ew.take(ind)
    Uw = Uw.take(ind, axis=0)
    # make sure all the eigenvalues used are positives
    count = 0
    for e in Ew:
        if e > 0:
            count = count + 1
        else:
            break
    Ew = Ew[0:count]
    Uw = Uw[0:count]
    E = nb_points/ nb_landmarks * Ew
    E = numpy.diag(E)
    EwpI = numpy.linalg.inv(E)
    D_col_mean = numpy.mean(D,axis=0)
    C = 0.5 * numpy.array([D_col_mean]*int(nb_points)) - C
    U = sqrt( nb_landmarks / nb_points ) * numpy.dot(numpy.dot(C, Uw.T) , EwpI)
    return U*[map(sqrt, Ew)]


def dr_toolbox(D):
    """
    Landmark isomap dimensionality reduction method
    This is the version implemented in
    http://homepage.tudelft.nl/19j49/Matlab_Toolbox_for_Dimensionality_Reduction.html
    
    Parameters
    ----------
    C : matrix of geodesic distances to the landmarks
    
    Returns
    -------
    embedded_coords : embedded coordinates of the original points
    """    
    (n, nl) = map(float, D.shape)
    subB = -0.5 * (
        D - numpy.array(
            [D.mean(axis=1)]*int(nl)
            ).transpose() - numpy.array(
            [D.mean(axis=0)]*int(n)
            ) + numpy.ones(D.shape)*D.sum() / (n * nl))
    subB2 = numpy.dot(subB.T, subB)
    beta,alpha = numpy.linalg.eig(subB2)
    val = map(numpy.sqrt, numpy.asarray(beta,dtype=complex))
    invVal = numpy.linalg.inv(numpy.diag(val))
    vec = numpy.dot( numpy.dot(subB,  alpha), invVal)

    # Computing final embedding
    val = numpy.array(map(numpy.real,val))
    ind = val.argsort()
    ind = ind[::-1] # we want decreasing order
    val = val.take(ind)
    vec = vec.take(ind, axis=1)
    return vec*[map(sqrt, val)]


def do_embedding(coords, tree='kdtree', landmarks=100, verbose=True, clean=True,
                 nystrom='facecrumbs',k=6):
    """
    Perform the whole dimensionality reduction process.

    Parameters
    ----------
    coords : numpy array, each row is a point
    tree : 'kdtree' or 'spilltree'
    landmarks : number of landmarks
    verbose : True or False
    clean : True or False, remove the long edges of the neighbourhood graph
    nystrom : 'facecrumbs' or 'dr_toolbox'
    k : number of neighbours for the neighbourhood graph
    
    Returns
    -------
    embedded_coords : embedded coordinates of the original points
    mapping : initial indices of the points actually embedded
    """

    if tree == 'kdtree':
        edges = kdTreeSearch(coords,k)
    elif tree == 'spilltree':
        edges = spillTreeSearch(coords,k)

    if verbose: print "search done"

    graph = build_neighbourhood_graph(edges)
    graph = select_largest_connected_component(graph)

    if clean:
        graph = clean_graph(graph)
        graph = select_largest_connected_component(graph)

    distances = distance_matrix(graph,landmarks)

    if nystrom == 'facecrumbs':
        embedded_coords = landmark_isomap(distances)
    else:
        embedded_coords = dr_toolbox(distances)
        
    return numpy.asarray(embedded_coords, dtype=float),graph.nodes()

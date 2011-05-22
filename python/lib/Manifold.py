import multiprocessing
from scipy.spatial import KDTree
import numpy
import networkx
from math import sqrt
from scipy import stats
import cv # OpenCV

def search_kdtree( p ):
    global tree
    global k    
    return  tree.query(p,k)
    
def kdTreeSearch( coords, _k, parallel=True ):
    global tree
    global k
    tree = KDTree( coords )
    k = _k                     

    if parallel:
        pool = multiprocessing.Pool()
        results = pool.map( search_kdtree, coords )
    else:
        results = map( search_kdtree, coords )
        
    edges = []
    for i in range( len(coords) ):
        (dist,res) = results[i]
        for n in range(0,k):
            edges.append([i,res[n],dist[n]])
      
    return edges

def search_spilltree(i):
    global tree
    global k
    res = cv.CreateMat(1, k, cv.CV_32SC1)
    dist = cv.CreateMat(1, k, cv.CV_64FC1)
    cv.FindFeatures(tree,cv_coords[i,:],res,dist,k,20)
    edges = []
    for n in range(0,k):
        edges.append([i,int(res[0,n]),dist[0,n]])
    return edges
    
def spillTreeSearch( coords, _k, parallel=True ):
    global tree
    global k
    global cv_coords
    nb_points = len(coords)
    cv_coords = cv.fromarray(coords)
    tree = cv.CreateSpillTree(cv_coords,100,0.7,0.1)
    k = _k

    if parallel:
        pool = multiprocessing.Pool()
        results = pool.map(search_spilltree, range(cv_coords.rows))
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
    graph = networkx.Graph()
    edges = map( lambda e: [e[0],e[1],{'weight':e[2]}], edges)
    graph.add_edges_from(edges)
    return graph

def select_largest_connected_component(graph):
    cc_list = networkx.connected_components(graph)
    return graph.subgraph(cc_list[0]) # they are correctly sorted

def clean_graph(graph):
    weights = [ graph[e[0]][e[1]]['weight'] for e in graph.edges()]
    treshold = stats.scoreatpercentile(weights, 95)
    new_edges = []
    for e in graph.edges():
        if graph[e[0]][e[1]]['weight'] < treshold:
            new_edges.append([e[0],e[1],{'weight':graph[e[0]][e[1]]['weight']}])
    new_graph = networkx.Graph()
    new_graph.add_edges_from(new_edges)
    return new_graph

def do_one_landmark(n):
    global graph
    d = networkx.shortest_path_length(graph,n,target=None, weighted=True)
    d = d.values()
    return map(lambda x: x*x, d)
    
def distance_matrix(_graph,nb_landmarks):
    global graph
    graph = _graph
    nb_nodes = len(graph)
    distances = []
    if nb_landmarks > nb_nodes:
        print "**ERROR** nb_landmarks > nb_nodes :", nb_landmarks, nb_nodes
    nodes = graph.nodes()

    pool = multiprocessing.Pool()
    distances = pool.map(do_one_landmark, nodes[0:nb_landmarks])
    
    distances = numpy.array(distances)
    return distances.T

def landmark_isomap(C):
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
    #return U


def dr_toolbox(D):
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
    # if size(vec, 2) < no_dims
    #   no_dims = size(vec, 2);
    #   warning(['Target dimensionality reduced to ', num2str(no_dims), '...']);
    # end

    # disp('Computing final embedding');
    val = numpy.array(map(numpy.real,val))
    ind = val.argsort()
    ind = ind[::-1] # we want decreasing order
    val = val.take(ind)
    vec = vec.take(ind, axis=1)
    #return vec
    return vec*[map(sqrt, val)]
    #return numpy.dot(vec, numpy.array([map(sqrt, val)]*int(n)).transpose() )
    #return numpy.dot(vec,numpy.diag( val) )
    # [val, ind] = sort(real(diag(val)), 'descend'); 
    # vec = vec(:,ind(1:no_dims));
    # val = val(1:no_dims);
    # mappedX = real(bsxfun(@times, vec, sqrt(val)'));

def do_embedding(coords, tree='kdtree', landmarks=100, verbose=True, clean=True,
                 nystrom='facecrumbs',k=6):

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

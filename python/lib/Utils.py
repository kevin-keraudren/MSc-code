import numpy
import multiprocessing
#import matplotlib.image as mpimg
import csv
import Image
import cv

def read_point_file( filename ):
    reader = csv.reader( open( filename, 'rb'), delimiter=' ' );
    points = []
    for row in reader:
        row = filter( lambda x: x != '', row ) # remove empty strings
        row = map( float, row )                # convert strings to float
        points.append( row )
    return numpy.array(points)

def read_int_file( filename ):
    points = read_point_file( filename )
    return points.astype(int)

def save_points(filename,coords):
    f = open( filename, 'wb' )
    i_max = len(coords)
    j_max = len(coords[0])
    for i in range(i_max):
        for j in range(j_max):
            f.write( str(coords[i][j]) )
            if j != j_max - 1:
                f.write( str(coords[i][j]) )
        if i != i_max - 1:
            f.write("\n")
    f.close()

# def save_mapping(filename,mapping):
#     f = open( filename, 'wb' )
#     for  m in mapping:
#         f.write( str(m))
#         if m != mapping[-1]:
#             f.write("\n")
#     f.close()

def slurp( filename ):
    lines = []
    f = open( filename, 'rb' )
    for l in f:
        l = l.strip()
        lines.append(l)
        # if len(l) > 0:
        #     lines.append(l)
    f.close()
    return lines

# from http://opencv.willowgarage.com/wiki/PythonInterface
def cv2array(im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

  arrdtype=im.depth
  a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im

def read_image_gray( filename ):
    img = Image.open(filename)
    img = img.convert("L")
    a_img = numpy.asarray(img)
    return a_img.flatten(1)

def read_image_hue( filename ):
    img = cv.LoadImage( filename, 1)
    
    #Compute HSV image and separate into colors
    hsv = cv.CreateImage( cv.GetSize(img), cv.IPL_DEPTH_8U, 3 )
    cv.CvtColor( img, hsv, cv.CV_BGR2HSV )

    h_plane = cv.CreateImage( cv.GetSize( img ), 8, 1 )
    s_plane = cv.CreateImage( cv.GetSize( img ), 8, 1 )
    v_plane = cv.CreateImage( cv.GetSize( img ), 8, 1 )
    #_plane = cv.CreateImage( cv.GetSize( img ), 8, 1 )
    cv.Split( hsv, h_plane, s_plane, v_plane,None )

    a_img = cv2array(h_plane)
    return a_img.flatten(1)

def read_image_canny( filename ):
    img = cv.LoadImage( filename, 1)
    gray = cv.CreateImage((img.width, img.height), 8, 1)
    edge = cv.CreateImage((img.width, img.height), 8, 1)
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    cv.Smooth(gray, edge, cv.CV_BLUR, 3, 3, 0)
    #cv.Not(gray, edge)

    # run the edge dector on gray scale
    cv.Canny(gray, edge, 80, 160, 3)
    cv.Not(edge, edge)
    cv.Smooth(edge, edge, cv.CV_GAUSSIAN, 3, 3, 0)
    
    a_img = cv2array(edge)
    return a_img.flatten(1)

def read_images(filenames, colorspace='gray'):
    pool = multiprocessing.Pool()
    if colorspace == 'gray':
        results = pool.map(read_image_gray, filenames)
    elif colorspace == 'hue':
        results = pool.map(read_image_hue, filenames)
    elif colorspace == 'canny':
        results = pool.map(read_image_canny, filenames)
    elif colorspace == 'points':
        results = pool.map(read_point_file, filenames)
    else:
        print "**ERROR**"
        sys.exit(1)
    return numpy.asarray(results,dtype=float)

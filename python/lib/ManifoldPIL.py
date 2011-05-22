import Image, Utils,numpy, ImageDraw

def define_scale(points,w,h):
    x_min = points[:,0].min()
    y_min = points[:,1].min()        
    x_max = points[:,0].max()
    y_max = points[:,1].max()
    return float(min(w,h)) / max(x_max-x_min, y_max-y_min)

def points2image(points,w,h, scale=None, center=None, point_size=2):
    print scale
    if scale == None:
        scale = define_scale(points,w,h)
              
    if center == None:
        center = points.mean(axis=0)
        
    img = Image.new('L', (w,h), 255)
    draw = ImageDraw.Draw(img)
    for p in points:
        p = (p - center) * scale + [w/2,h/2]
        pbox = [(p[0]-point_size,p[1]-point_size),(p[0]+point_size,p[1]+point_size)]
        draw.ellipse(pbox, fill=0 )

    return img
        
    

def render2D( images, coords, output_filename="output.jpg",
              width=20, height=20, image_w=64, image_h=64, style="image" ):
    coords = coords[:,0:2]
    image_grid = []

    min_x = min( coords[:,0])
    min_y = min( coords[:,1])
    max_x = max( coords[:,0])
    max_y = max( coords[:,1])

    grid_w = (max_x - min_x) / (width -1)
    grid_h = (max_y - min_y) / (height-1)

    for i in range(0,width):
        image_grid.append([])
        for j in range(0,height):
            image_grid[i].append([])

    for c in range(0,len(coords)):
        i = int( (coords[c,0] - min_x) / grid_w )
        j = int( (coords[c,1] - min_y) / grid_h )
        image_grid[i][j].append(images[c])

    mode =  "RGB"
    color = (255, 255, 255)
    output = Image.new(mode, (width*image_w,height*image_h), color)
    for i in range(0,width):
        for j in range(0,height):
            if len(image_grid[i][j]) > 0:
                if style == 'images':
                    img = Image.open(image_grid[i][j][0])
                    img = img.resize( (image_w,image_h) )
                elif style == 'edges':
                    img = Utils.read_image_canny(image_grid[i][j][0])
                    img = Image.fromarray(numpy.reshape(img,(image_w,image_h)))
                elif style == 'points':
                    img = points2image( image_grid[i][j][0],image_w,image_h,
                                        scale=min(image_w,image_h),center=[0,0])
                output.paste(img,(i*image_w,j*image_h))

    output.save(output_filename)

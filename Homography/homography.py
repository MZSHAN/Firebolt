"""
Further work:
    1. This homography calculation will fail for matrix with large condition number
        Rectify this

    2. Implement algorithm to find the convex hull. Currently using matplotlib
    3. Implement the SVD solver - currently using at the one from numpy for homograhy
"""

"""
Caveats: 
    Python Image Library uses the inverted coordinate frame to represent pixels - so (width, height) ~ (x, y)
    Numpy uses the matrix representation (x, y) ~ (height, width) 
    Hence be careful when using Image.fromarray - 
"""




from matplotlib.path import Path
from PIL import Image
import csv
import numpy as np
from os import listdir
from os.path import isfile, join



def read_vertices2(vertices_csv):
    """
    Reads the vertices of the a polynomial from a csv file
    :param vertices_csv: path of csv file
    :return: np array with dims (#images, #vertices per image, 2)
    """
    points = []
    with open(vertices_csv) as vertices_csv:
        csv_reader = csv.reader(vertices_csv, delimiter=',')
        for i, row in enumerate(csv_reader):
            for j in range(int(len(row)/2)):
                if i == 0:
                    points.append([])
                points[j].append([float(row[2*j]), float(row[2*j+1])])
    return np.array(points)


def points_in_convex_hull(vertices, width, height):
    """
    Function takes a set of vertices and width and height of the the image
    Returns all the points contained within the vertices and a image mask for
    the contained points
    The order of the vertices must be either clockwise or anti-clockwise

    Function expects
    :param vertices: 2D vertices of the polygon - list of lists
    :param width: width of the image - int
    :param height: height of the image - int
    :return:
        mask: mask with points inside polygon made by vertices
        in_points: tuple of arrays with co-ordinates of points in convex hull
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    convex_hull = Path(vertices)
    mask = convex_hull.contains_points(points)
    mask = mask.reshape(height, width)
    in_points = mask.T.nonzero()
    return mask, in_points




def calculate_homography(source_pnts, target_pnts):
    """
    Given a set of matching source and target points, calculate the homography matrix
    :param source_pnts: list of 2d source points
    :param target_pnts: list of 2d target points
    :return: 3 X 3 numpy Homography matrix between the source and target points
    """
    assert(len(source_pnts) >= 4)
    assert(len(source_pnts) == len(target_pnts))

    A = np.zeros((2 * len(source_pnts), 9))

    i = 0
    for s, t in zip(source_pnts, target_pnts):
        A[i] = np.array([-s[0], -s[1], -1, 0 , 0, 0, s[0]*t[0], s[1]*t[0] , t[0]])
        A[i+1] = np.array([0, 0 , 0, -s[0], -s[1], -1, s[0]*t[1], s[1]*t[1], t[1]])
        i += 2
    _, _, vh = np.linalg.svd(A)

    homography =  np.reshape(vh[-1], (3, 3))
    homography = homography / homography[2, 2]
    return homography


def apply_homography(source_pnts, H):
    """
    Given a set of source points and a Homography transformation, applies to transformation to get
    target points
    :param source_pnts: list of 2D points
    :param H:
    :return:
    """
    source_homog = np.vstack((np.array(source_pnts), np.ones(len(source_pnts[0]))))
    target_homog = np.dot(H, source_homog)
    target_homog = target_homog / target_homog[2]
    return np.ceil(target_homog[0:2]).astype(int)


def embed_source_in_target(target_pnts, source_pnts, target, source):
    """
    Embed average of target and source at target_pnts and source_pnts respectively
    in the target image
    :param target_pnts: list of 2d source location points
    :param source_pnts: list of 2d target location points
    :param target: PIL target image
    :param source: PIL source image
    :return: target PIL image with embedded source image
    """
    target = np.array(target)
    source = np.array(source)

    target_pnts = (target_pnts[1], target_pnts[0])
    source_pnts = (source_pnts[1], source_pnts[0])

    target[target_pnts] = target[target_pnts]/2 + source[source_pnts]/2
    return Image.fromarray(target)


if __name__ == "__main__":
    filepath = "images/barcaReal/"
    targetfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    targetfiles = sorted(targetfiles)

    target_vertices = read_vertices2("video_pts.csv")
    assert(len(targetfiles) == target_vertices.shape[1], "Labels don't exist for all images")

    source = Image.open("images/logos/cmu_logo.jpg")
    width, height = source.size
    source_vert = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

    result_images = []
    for i, filename in enumerate(targetfiles):
        target_vert = target_vertices[i]
        target = Image.open("images/barcaReal/" + targetfiles[i])

        H = calculate_homography(target_vert, source_vert)
        _, all_target_pnts = points_in_convex_hull(target_vert, *target.size)
        source_pnts = apply_homography(all_target_pnts, H)
        result = embed_source_in_target(all_target_pnts, source_pnts, target, source)
        result_images.append(result)

    result_images[0].save('output.gif', save_all=True,
                          append_images=result_images[1:], optimize=False, duration=1.5)






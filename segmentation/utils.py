import json
import os
import cv2
import shutil
from shapely import affinity
from shapely import geometry
import numpy as np

def load_json(path):
    with open(path) as f:
        metadata = json.load(f)
    return metadata

def write_dir(dir, img, fname):
    write_path = os.path.join(dir, fname)
    cv2.imwrite(write_path, img)

def create_folders(paths, delete_existing=True):
    for p in paths:
        if delete_existing and os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

def shrink_polygon(poly_points, xfact, yfact):
    polygon = geometry.Polygon(poly_points)
    scaled_polygon = affinity.scale(polygon, xfact, yfact)
    xs, ys = scaled_polygon.exterior.coords.xy
    xs = xs.tolist()
    ys = ys.tolist()

    assert len(xs) == len(ys)
    shrinked_points = np.array([[int(xs[i]), int(ys[i])] for i in range(len(xs))])
    return shrinked_points

def get_poly_points(seg):
    pts = []
    for i in range(0, len(seg), 2):
        x = seg[i]
        y = seg[i+1]
        pts.append([x, y])
    pts = np.array(pts)
    return pts


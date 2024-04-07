from utils import (
    load_json,
    get_poly_points, 
    shrink_polygon,
    write_dir,
    create_folders,
)
from collections import defaultdict
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import (
    METADATA, 
    TRAIN_XRAY_PATH,
    TRAIN_MASKS_PATH,
)

def get_id_segment_map(annotations):
    id_segment_map = defaultdict(list)

    for annotation in annotations:
        img_id = annotation['image_id']
        seg = annotation['segmentation']
        assert len(seg) == 1
        id_segment_map[img_id].append(seg[0])
    
    return id_segment_map

def make_mask(im, segments, scale_fact):
    mask = np.zeros(im.shape[:2], np.uint8)
    for i, seg in enumerate(segments):
        poly_points = get_poly_points(seg)
        poly_points = shrink_polygon(poly_points, xfact=scale_fact, yfact=scale_fact)
        cv2.drawContours(mask, [poly_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask

def create_img_masks(image_metadata, id_segment_map):
    for img_meta in image_metadata:
        img_id = img_meta['id']
        fname = img_meta['file_name']
        segments = id_segment_map[img_id]
        img_path = os.path.join(TRAIN_XRAY_PATH, fname)
        im = cv2.imread(img_path)
        mask = make_mask(im, segments, 0.9)
        write_dir(TRAIN_MASKS_PATH, mask, fname)
        

metadata = load_json(METADATA)
image_metadata = metadata['images']
annotations = metadata['annotations']
id_segment_map = get_id_segment_map(annotations)
create_folders([TRAIN_MASKS_PATH])
create_img_masks(image_metadata, id_segment_map)



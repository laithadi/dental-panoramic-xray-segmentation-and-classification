import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import shutil
import pandas as pd

TRAIN_PATH = "train/xrays"
METADATA = "train/metadata.json"

CROPPED = "cropped_dataset/train/cropped"
MASK = "cropped_dataset/train/mask"
BLACK_BG = "cropped_dataset/train/black_bg"
WHTIE_BG = "cropped_dataset/train/white_bg"
DATASET = "cropped_dataset"
LABELS = "cropped_dataset/labels.csv"

def get_id_img_map(image_metadata):
    id_img_map = {}

    for img_meta in image_metadata:
        id = img_meta['id']
        fname = img_meta['file_name']
        id_img_map[id] = fname
    
    return id_img_map

def load_json(path):
    with open(path) as f:
        metadata = json.load(f)
    
    return metadata

def get_poly_points(seg):
    pts = []
    for i in range(0, len(seg), 2):
        x = seg[i]
        y = seg[i+1]
        pts.append([x, y])
    pts = np.array(pts)
    return pts

def crop_img(im, poly_points):
    rect = cv2.boundingRect(poly_points)
    x,y,w,h = rect
    cropped = im[y:y+h, x:x+w].copy()

    poly_points = poly_points - poly_points.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [poly_points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    croped_black_bg = cv2.bitwise_and(cropped, cropped, mask=mask)

    white_bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(white_bg, white_bg, mask=mask)
    croped_white_bg = white_bg + croped_black_bg

    return [cropped, mask, croped_black_bg, croped_white_bg]

def create_folder_structure(delete_existing=True):
    if delete_existing and os.path.exists(DATASET):
        shutil.rmtree(DATASET)

    paths = [CROPPED, MASK, BLACK_BG, WHTIE_BG]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

def write_dir(dir, img, fname):
    write_path = os.path.join(dir, fname)
    cv2.imwrite(write_path, img)

def create_segmented_imgs_and_labels(id_img_map, annotations):
    labels = {}
    for i, annotation in enumerate(annotations):
        img_id = annotation['image_id']
        seg = annotation['segmentation']
        assert len(seg) == 1
        img_path = os.path.join(TRAIN_PATH, id_img_map[img_id])
        im = cv2.imread(img_path)
        poly_points = get_poly_points(seg[0])
        croped_imgs = crop_img(im, poly_points)

        cropped = croped_imgs[0]
        mask = croped_imgs[1]
        black_bg = croped_imgs[2]
        white_bg = croped_imgs[3]
        fname = "{}.png".format(i)

        write_dir(CROPPED, cropped, fname)
        write_dir(MASK, mask, fname)
        write_dir(BLACK_BG, black_bg, fname)
        write_dir(WHTIE_BG, white_bg, fname)

        label = annotation['category_id_3']
        src = id_img_map[img_id]
        labels[i] = [label, src]

    df = pd.DataFrame.from_dict(data = labels, orient='index') 
    df.columns = ['label', 'source_img']
    df.index.name = 'id'
    df.to_csv(LABELS) 

metadata = load_json(METADATA)
image_metadata = metadata['images']
annotations = metadata['annotations']
id_img_map = get_id_img_map(image_metadata)
create_folder_structure()
create_segmented_imgs_and_labels(id_img_map, annotations)

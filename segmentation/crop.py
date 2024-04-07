import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils import (
    load_json, 
    write_dir, 
    create_folders, 
    get_poly_points, 
    shrink_polygon)
from constants import (
    TRAIN_XRAY_PATH,
    METADATA,

    CROPPED,
    MASK,
    BLACK_BG,
    WHTIE_BG,
    LABELS_PATH,
    EXPERIMENT_PATH
)

def get_id_img_map(image_metadata):
    id_img_map = {}

    for img_meta in image_metadata:
        id = img_meta['id']
        fname = img_meta['file_name']
        id_img_map[id] = fname
    
    return id_img_map

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

def create_segmented_imgs_and_labels(id_img_map, annotations):
    labels = {}
    for i, annotation in enumerate(annotations):
        img_id = annotation['image_id']
        seg = annotation['segmentation']
        assert len(seg) == 1
        img_path = os.path.join(TRAIN_XRAY_PATH, id_img_map[img_id])
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
    df.to_csv(LABELS_PATH)

def create_crop_img_with_mask_overlay(id_img_map, annotations, scale_fact):
    dir = os.path.join(EXPERIMENT_PATH, str(scale_fact))
    create_folders([dir])
    for i, annotation in enumerate(annotations):
        img_id = annotation['image_id']
        seg = annotation['segmentation']
        assert len(seg) == 1
        img_path = os.path.join(TRAIN_XRAY_PATH, id_img_map[img_id])
        im = cv2.imread(img_path)
        poly_points = get_poly_points(seg[0])
        poly_points = shrink_polygon(poly_points, xfact=scale_fact, yfact=scale_fact)
        croped_imgs = crop_img(im, poly_points)

        cropped = croped_imgs[0]
        mask = croped_imgs[1]
        fname = "{}.png".format(i+1)
        file_path = os.path.join(dir, fname)

        plt.figure()
        plt.imshow(cropped, 'gray', interpolation='none')
        plt.imshow(mask, 'jet', interpolation='none', alpha=0.5)
        plt.savefig(file_path)
        plt.close()

        if (i+1) == 100:
            break

def main():
    EXPERIMENT_MODE = False
    metadata = load_json(METADATA)
    image_metadata = metadata['images']
    annotations = metadata['annotations']
    id_img_map = get_id_img_map(image_metadata)
    if EXPERIMENT_MODE:
        create_crop_img_with_mask_overlay(id_img_map, annotations, 0.95)
    else:
        create_folders([CROPPED, MASK, BLACK_BG, WHTIE_BG])
        create_segmented_imgs_and_labels(id_img_map, annotations)

main()

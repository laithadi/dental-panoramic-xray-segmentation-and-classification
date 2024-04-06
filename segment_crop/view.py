import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json

TRAIN_PATH = "train/xrays"
METADATA = "train/metadata.json"

def load_json(path):
    with open(path) as f:
        metadata = json.load(f)
    
    return metadata

def get_img_id(image_metadata, fname):
    for img_meta in image_metadata:
        if fname == img_meta['file_name']:
            return img_meta['id']
    return None

def show_with_bbox(id, im, annotations, box_type="polygon"):
    for annotation in annotations:
        img_id = annotation['image_id']
        seg = annotation['segmentation']
        assert len(seg) == 1
        if img_id == id:
            if box_type == "rectangle":
                bbox = annotation['bbox']
                label = annotation['category_id_3']
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                cv2.rectangle(im, (x, y), (x+w, y+h), color=(36,255,12), thickness=10)
                cv2.putText(im, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 4)
            else:
                poly_points = seg[0] 
                i = 0
                pts = []
                for i in range(0, len(poly_points), 2):
                    x = poly_points[i]
                    y = poly_points[i+1]
                    pts.append([x, y])
                pts = np.array(pts)
                cv2.polylines(im, [pts], True, color=(255,0,0), thickness=10)

    plt.imshow(im)
    plt.show()

metadata = load_json(METADATA)
image_metadata = metadata['images']
annotations = metadata['annotations']

img_fname = "train_251.png"
img_path = os.path.join(TRAIN_PATH, img_fname)
im = cv2.imread(img_path)
id = get_img_id(image_metadata, img_fname)
show_with_bbox(id, im, annotations, "rectangle")
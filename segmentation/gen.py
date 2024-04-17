from dataset import TeethSegmentationData
import os
from constants import (
    TRAIN_XRAY_PATH, 
    GEN_POLYGON_PATH, 
    GEN_OVERLAY_PATH, 
    GEN_CROPPED,
    GEN_BLACK_BG,
    GEN_MASK, 
    GEN_WHTIE_BG,
    GEN_SOURCES_PATH)
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import random 
from net import ResUnet
import cv2
from utils import create_folders, write_dir, get_poly_points
from shapely import geometry
from crop import crop_img
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

def get_dataloader():
    fnames = os.listdir(TRAIN_XRAY_PATH)
    transforms = Transforms.Compose([
        Transforms.Resize((512, 512)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.5,), (0.5,))

    ])
    ds = TeethSegmentationData(fnames, transforms)
    data_loader = DataLoader(dataset=ds, batch_size=1, shuffle=True, num_workers=1)

    return data_loader

def convert_mask(tensor):
    image = tensor.cpu().clone().detach().numpy() 
    image = image.transpose(1,2,0)
    image = image * np.array((1, 1, 1)) 
    image = image.clip(0, 1) 
    return image

def create_masks(model, data_loader, overlay=True):
    if overlay:
        create_folders([GEN_OVERLAY_PATH])
    else:
        create_folders([GEN_POLYGON_PATH, GEN_CROPPED, GEN_MASK, GEN_BLACK_BG, GEN_WHTIE_BG])

    sources = {}
    k = 0
    for i, sample in enumerate(data_loader):
        inp = sample['img']
        fname = sample['fname']

        img_path = os.path.join(TRAIN_XRAY_PATH, fname[0])
        img = cv2.imread(img_path)
        w = img.shape[1]
        h = img.shape[0]
        mask = convert_mask(model(inp.to(device))[0])
        resized_mask = cv2.resize(mask.astype(np.float32), (w, h))

        if overlay:
            save_path = os.path.join(GEN_OVERLAY_PATH, fname[0])
            plt.figure()
            plt.imshow(img, 'gray', interpolation='none')
            plt.imshow(resized_mask, 'jet', interpolation='none', alpha=0.5)
            plt.savefig(save_path)
            plt.close()
        else:
            save_path = os.path.join(GEN_POLYGON_PATH, fname[0])
            kernel =(np.ones((3,3), dtype=np.float32))

            # remove noise
            resized_mask=cv2.morphologyEx(resized_mask, cv2.MORPH_OPEN, kernel)

            # to gray scale
            gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

            grayy = (gray*255*10).astype(np.uint8)
            thresh = cv2.threshold(grayy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            erosion = cv2.erode(thresh, kernel, iterations=2) #,iterations=2
            
            contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            polygons = []

            for obj in contours:
                coords = []
                    
                for point in obj:
                    x = int(point[0][0])
                    y = int(point[0][1])

                    coords.append(x)
                    coords.append(y)

                polygons.append(coords)

            for polygon in polygons:
                pts = get_poly_points(polygon)

                try: 
                    poly = geometry.Polygon(pts)
                    if poly.area > 5000.0:
                        croped_imgs = crop_img(img, pts)
                        cropped = croped_imgs[0]
                        mask = croped_imgs[1]
                        black_bg = croped_imgs[2]
                        white_bg = croped_imgs[3]
                        f = "{}.png".format(k)

                        write_dir(GEN_CROPPED, cropped, f)
                        write_dir(GEN_MASK, mask, f)
                        write_dir(GEN_BLACK_BG, black_bg, f)
                        write_dir(GEN_WHTIE_BG, white_bg, f)

                        sources[k] = fname
                        k += 1
                        cv2.polylines(img, [pts], True, color=(255,0,0), thickness=10)
                except:
                    pass

            plt.figure()
            plt.imshow(img)
            plt.savefig(save_path)
            plt.close()

            df = pd.DataFrame.from_dict(data = sources, orient='index') 
            df.columns = ['source_img']
            df.index.name = 'id'
            df.to_csv(GEN_SOURCES_PATH)

def main():
    seed_everything(20790117)

    data_loader = get_dataloader()
    model = ResUnet()
    model.load_state_dict(torch.load('models/best_unet_051722_v1.pth'))
    model.to(device)
    create_masks(model, data_loader, False)

main()

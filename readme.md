# Aim of this project 
The project aims to leverage a comprehensive dataset of dental panoramic X-rays, collected in 2023 and fully labeled with diagnoses among other attributes. The primary objective is to develop a segmentation model capable of accurately depicting each tooth within the panoramic X-ray images. These segmented tooth images will then be extracted and utilized as inputs for a Convolutional Neural Network (CNN). The CNN's role is to predict the diagnostic class for each tooth, thereby facilitating efficient and precise dental diagnosis.

# Introduction 

## Link to dataset 
https://zenodo.org/records/7812323#.ZDQE1uxBwUG

## Summary 
- **DENTEX Challenge Overview:**
  - The DENTEX challenge was held in 2023 during MICCAI.
  - Its main goal was to develop algorithms for accurately detecting abnormal teeth and providing associated diagnoses on panoramic X-rays, aiding in precise treatment planning and reducing errors.

- **Hierarchically Annotated Data:**
  - The challenge offers three types of annotated data structured using the FDI system.
    1. Partially labeled data with quadrant information.
    2. Partially labeled data with quadrant and enumeration information.
    3. Fully labeled data containing quadrant-enumeration-diagnosis information for each abnormal tooth. All participant algorithms will be evaluated using this data.

- **Purpose of DENTEX:**
  - DENTEX aims to evaluate AI effectiveness in dental radiology analysis and its potential to enhance dental practice by comparing algorithms that identify abnormal teeth with enumeration and diagnosis on panoramic X-rays.

## About dataset 
As mentioned in the previous section, the challenge offers three types of annotated data structured using the FDI system.
1. Partially labeled data with quadrant information.
2. Partially labeled data with quadrant and enumeration information.
3. Fully labeled data containing quadrant-enumeration-diagnosis information for each abnormal tooth. All participant algorithms will be evaluated using this data.

### 1. Partially labeled data with quadrant information (693 X-rays)
- Labels: Quadrant information only.
- Explanation: Each quadrant is numbered from 1 through 4, with the top right quadrant being 1, top left quadrant being 2, bottom left quadrant being 3, and bottom right quadrant being 4.

### 2. Partially labeled data with quadrant and enumeratin information (634 X-rays)
- Labels: Quadrant and tooth enumeration information.
- Explanation: Each tooth within a quadrant is numbered from 1 through 8 according to the FDI numbering system. For instance, the numbering starts from the front middle tooth, and the numbers increase towards the back. Therefore, the back tooth on the lower left side would be labeled as 48 according to the FDI notation, indicating quadrant 4, tooth number 8.

### 3. Fully labeled data for abnormal tooth detection with quadrant, tooth, enumeration, and diagnosis (1005 X-rays)
- Labels: Quadrant, tooth enumeration, and diagnosis information.
- Explanation: In addition to quadrant and tooth enumeration information, this dataset includes diagnosis classes such as Caries, Deep caries, Periapical lesions, and Impacted teeth.

## How we will use the different datasets 
We will be using the fully labeled data for abnormal tooth detection with quadrant, tooth enumeration, and diagnosis classes. 

# Data 

## X-rays
We will display a couple X-ray images later in the EDA (Exploratory Data Analysis) section. 

## Annotations 
The annotation files (jsons) consists of the same root fields or top-level fields.<br>
```json
{
  "images": [...],
  "annotations": [...], 
  "categories_1": [...], -> maps encoded values to quadrant labels 1, 2, 3, 4
  "categories_2": [...], -> maps encoded values to enumeration labels 1, 2, 3, 4, 5, 6, 7, 8
  "categories_3": [...] -> maps encoded values to diagnosis labels Impacted, Caries, Periapical Lesion, Deep Caries 
}
```
Not every annotation files consists of all the root fields above. Depending on the type of data provided in the dataset discussed earlier, some of the root fields might be excluded. 

### Root field: images 
The images field consists information about the panoramix X-rays. It is a list of dictionaries, each dictionary representing an X-ray. <br> 
```json
{
  "images": [
    {
        "height": 1316,
        "width": 2744,
        "id": 1,
        "file_name": "train_673.png"
    },
    {
        "height": 1504,
        "width": 2872,
        "id": 2,
        "file_name": "train_283.png"
    },
    ...
  ],
  "annotations": [...], 
  "categories_1": [...], 
  "categories_2": [...], 
  "categories_3": [...] 
}
```

### Root field: annotations
The annotations field consists of the annotations for each image. It is a list of dictionaries, each dictionary represents the annotations an image. <br>
Note: It is important to know that there could be more than one annotations dictionary/object for an image. Since a single image can consist of multiple labels. 
```json
{
  "images": [...],
  "annotations": [
    {
        "iscrowd": 0,
        "image_id": 1,
        "bbox": [
            542.0,
            698.0,
            220.0,
            271.0
        ],
        "segmentation": [
            [
                621,
                703,
                573,
                744,
                542,
                885,
                580,
                945,
                650,
                969,
                711,
                883,
                762,
                807,
                748,
                741,
                649,
                698
            ]
        ],
        "id": 1,
        "area": 39683,
        "category_id_1": 3,
        "category_id_2": 7,
        "category_id_3": 0
    },
    ...
  ], 
  "categories_1": [...],
  "categories_2": [...], 
  "categories_3": [...] 
}
```

### Root field: categories_1
Maps encoded values to class labels. 
```json
{
  "images": [...],
  "annotations": [...], 
  "categories_1": [
    {
        "id": 0,
        "name": "1",
        "supercategory": "1"
    },
    {
        "id": 1,
        "name": "2",
        "supercategory": "2"
    },
    {
        "id": 2,
        "name": "3",
        "supercategory": "3"
    },
    {
        "id": 3,
        "name": "4",
        "supercategory": "4"
    }
  ], 
  "categories_2": [...], 
  "categories_3": [...] 
}
```

### Root field: categories_2
Maps encoded values to class labels. 
```json
{
  "images": [...],
  "annotations": [...], 
  "categories_1": [...], 
  "categories_2": [
    {
        "id": 0,
        "name": "1",
        "supercategory": "1"
    },
    {
        "id": 1,
        "name": "2",
        "supercategory": "2"
    },
    {
        "id": 2,
        "name": "3",
        "supercategory": "3"
    },
    {
        "id": 3,
        "name": "4",
        "supercategory": "4"
    },
    {
        "id": 4,
        "name": "5",
        "supercategory": "5"
    },
    {
        "id": 5,
        "name": "6",
        "supercategory": "6"
    },
    {
        "id": 6,
        "name": "7",
        "supercategory": "7"
    },
    {
        "id": 7,
        "name": "8",
        "supercategory": "8"
    }
  ], 
  "categories_3": [...] 
}
```

### Root field: categories_3
Maps encoded values to class labels. 
```json
{
  "images": [...],
  "annotations": [...], 
  "categories_1": [...], 
  "categories_2": [...], 
  "categories_3": [
    {
        "id": 0,
        "name": "Impacted",
        "supercategory": "Impacted"
    },
    {
        "id": 1,
        "name": "Caries",
        "supercategory": "Caries"
    },
    {
        "id": 2,
        "name": "Periapical Lesion",
        "supercategory": "Periapical Lesion"
    },
    {
        "id": 3,
        "name": "Deep Caries",
        "supercategory": "Deep Caries"
    }
  ] 
}
```

# Method
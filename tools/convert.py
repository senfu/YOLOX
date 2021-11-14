import os
import os.path as osp
import numpy as np
from PIL import Image
import json
import shutil
from tqdm import tqdm
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, default="/data/junyanli/widerface/zip")
parser.add_argument("-o", type=str, default="/data/junyanli/YOLOX/datasets/widerface_coco")
args = parser.parse_args()

dataset_root = args.i
destination_root = args.o


def make_dataset(labels_file, images_dir, output_json_file, output_images_dir):
    input = open(labels_file, "r")
    output = open(output_json_file, "w")
    data = {}
    for line in input.readlines():
        line = line.rstrip().split()
        if line[0] == "#":
            image_name = line[-1]
            data[image_name] = []
        else:
            label = list(map(float, line))
            bbox = np.array(label[:4]) # xywh
            landmark = np.zeros(10)
            if len(label) != 4:
                landmark[0] = label[4]    # l0_x
                landmark[1] = label[5]    # l0_y
                landmark[2] = label[7]    # l1_x
                landmark[3] = label[8]    # l1_y
                landmark[4] = label[10]   # l2_x
                landmark[5] = label[11]   # l2_y
                landmark[6] = label[13]   # l3_x
                landmark[7] = label[14]   # l3_y
                landmark[8] = label[16]   # l4_x
                landmark[9] = label[17]   # l4_y
            bbox = bbox.tolist()
            landmark = landmark.tolist()
            data[image_name].append({"bbox": bbox,
                                     "landmark": landmark})
    instances_json = dict(
        info=dict(
            year=2021,
            version=1.0,
            description="wider face in coco format",
            date_created=2021
        ),
        licenses=None,
        categories=[{"id": 1, "name": "face", "supercategory": "face"}],
        images=[],
        annotations=[]
    )
    
    image_id_cnt = 0
    annotation_id_cnt = 0
    for image_name in tqdm(data):
        annotations = data[image_name]
        image_folder, image_name = image_name.split("/")
        image_full_path = osp.join(images_dir, 
                                   image_folder, 
                                   image_name)
        image = Image.open(image_full_path)
        height = image.height
        width = image.width
        image_id_cnt += 1
        file_name = image_folder+"=="+image_name
        instances_json["images"].append(dict(
            date_captured="2021",
            file_name=file_name,
            id=image_id_cnt,
            height=height,
            width=width
        ))
        shutil.copy(image_full_path, osp.join(output_images_dir, file_name))
        for annotation in annotations:
            annotation_id_cnt += 1
            instances_json["annotations"].append(dict(
                id=annotation_id_cnt,
                image_id=image_id_cnt,
                category_id=1,
                bbox=annotation['bbox'],
                area=annotation['bbox'][-1]*annotation['bbox'][-2],
                segmentation=annotation['landmark'],
                iscrowd=0
            ))
    json.dump(instances_json, output, indent=2)
    input.close()
    output.close()



train_images_dir = osp.join(dataset_root, 'WIDER_train', 'images')
val_images_dir = osp.join(dataset_root, 'WIDER_val', 'images')
train_labels_file = osp.join(dataset_root, "train/label.txt")
val_labels_file = osp.join(dataset_root, "val/label.txt")

os.makedirs(destination_root, exist_ok=True)
os.makedirs(osp.join(destination_root, "annotations"), exist_ok=True)
os.makedirs(osp.join(destination_root, "train2017"), exist_ok=True)
os.makedirs(osp.join(destination_root, "val2017"), exist_ok=True)
make_dataset(train_labels_file,
             train_images_dir,
             osp.join(destination_root, "annotations", "instances_train2017.json"),
             osp.join(destination_root, "train2017"))
make_dataset(val_labels_file,
             val_images_dir,
             osp.join(destination_root, "annotations", "instances_val2017.json"),
             osp.join(destination_root, "val2017"))

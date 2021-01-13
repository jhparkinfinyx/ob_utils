# VOC dataset annotation info
# Format: xml
# Labels(in String): 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'

import json
import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

input_base_dir = os.path.expanduser('~/data/cars')

imgs_dir = os.path.join(input_base_dir, 'VOC/VOC2012/JPEGImages')
anno_dir = os.path.join(input_base_dir, 'VOC/VOC2012/Annotations')

output_json_path = os.path.join(anno_dir, 'car_classify.json')


def voc2json():
    output_annos = []
    imgs_dir = os.path.join(input_base_dir, 'VOC/VOC2012/JPEGImages')
    anno_dir = os.path.join(input_base_dir, 'VOC/VOC2012/Annotations')

    for img_filename in tqdm(sorted(os.listdir(imgs_dir)), desc='VOC'):
        img_path = os.path.join(imgs_dir, img_filename)
        anno_path = os.path.join(anno_dir, img_filename.replace('.jpg', '.xml'))
        assert os.path.exists(anno_path)

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[0], img.shape[1]
        img_anno = {"filename": img_path.replace(input_base_dir, ''), "width": img_width, "height": img_height, "ann": {"bboxes": [], "labels": [], "bboxes_ignore": [], "labels_ignore": []}}
        for object in ET.parse(anno_path).getroot():
            if object.tag != 'object':
                continue

            name = object.find('name')
            if name.text == 'bus' or name.text == 'car':
                bbox = object.find('bndbox')
                x1, y1, x2, y2 = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
                img_anno['ann']['bboxes'].append([x1, y1, x2, y2])
                img_anno['ann']['labels'].append(1)

        if len(img_anno['ann']['bboxes']) > 0:
            output_annos.append(img_anno)

    return output_annos


if __name__ == '__main__':
    output_annos = voc2json()
    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

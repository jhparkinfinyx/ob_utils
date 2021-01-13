import json
import os
import random

import cv2
from tqdm import tqdm

from open_data_convert.anno_converter import *

input_base_dir = os.path.expanduser('~/data/cars')
output_json_path = os.path.expanduser('~/data/cars/car_classify.json')
debug_dir = os.path.expanduser('~/data/cars/debug_convert')

DEBUG = False
DEBUG_RANDOM_SHUFFLE = True
DEBUG_MAX_NUM = 1000  # -1 to all


def debug(max_num):
    colors = [(127, 127, 127), (0, 255, 0)]
    json_data = open(output_json_path, 'r')
    data_list = json.load(json_data)

    if DEBUG_RANDOM_SHUFFLE:
        random.shuffle(data_list)

    count = 0
    for data in tqdm(data_list):
        if count > max_num != -1:
            break
        count += 1

        bboxes_ignore = data['ann']['bboxes_ignore']
        filename, bboxes, labels = data['filename'], data['ann']['bboxes'], data['ann']['labels']
        img_path = os.path.join(input_base_dir + filename)  # + Due to relative path in data['filename']

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for ig_bbox in bboxes_ignore:
            x1, y1, x2, y2 = int(ig_bbox[0]), int(ig_bbox[1]), int(ig_bbox[2]), int(ig_bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[0], thickness=-1)
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(label)])

        debug_path = os.path.join(debug_dir, filename.replace('/', '_'))
        cv2.imwrite(debug_path, img)

    json_data.close()


if __name__ == '__main__':
    output_annos = []

    print('Converting dataset annotations to json')
    output_annos.extend(iipt_train2json())
    output_annos.extend(kitti2json())
    output_annos.extend(mot2json())
    output_annos.extend(ua_detrac2json())
    output_annos.extend(voc2json())

    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

    if DEBUG:
        print('\nDebugging (MAX_NUM={}, RANDOM_SHUFFLE={})'.format(DEBUG_MAX_NUM, DEBUG_RANDOM_SHUFFLE))
        os.makedirs(debug_dir, exist_ok=True)
        debug(DEBUG_MAX_NUM)

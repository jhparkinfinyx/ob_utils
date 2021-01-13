# UA_DETRAC dataset annotation info
# Format: xml
# Labels(in String): 'car', 'bus', 'van', 'others'

import json
import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

input_base_dir = os.path.expanduser('~/data/cars')

imgs_dir = os.path.join(input_base_dir, 'UA_DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train/')
anno_dir = os.path.join(input_base_dir, 'UA_DETRAC/DETRAC-Train-Annotations-XML')

output_json_path = os.path.join(anno_dir, 'car_classify.json')


def ua_detrac2json():
    output_annos = []
    imgs_dir = os.path.join(input_base_dir, 'UA_DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train/')
    anno_dir = os.path.join(input_base_dir, 'UA_DETRAC/DETRAC-Train-Annotations-XML')

    for video_name in tqdm(sorted(os.listdir(imgs_dir), key=lambda index: int(index.split('_')[-1])), desc='UA_DETRAC'):
        video_annos = {}
        video_dir = os.path.join(imgs_dir, video_name)
        video_width, video_height = 0, 0

        ignore_bboxes = []
        anno_path = os.path.join(anno_dir, '{}.xml'.format(video_name))
        assert os.path.exists(anno_path)

        tree = ET.parse(anno_path)
        root = tree.getroot()
        for frame in root:
            if frame.tag == 'ignored_region':
                for ignore_bbox in frame:
                    x1, y1, w, h = float(ignore_bbox.get('left')), float(ignore_bbox.get('top')), float(ignore_bbox.get('width')), float(ignore_bbox.get('height'))
                    ignore_bboxes.append([int(x1), int(y1), int(x1 + w), int(y1 + h)])
            if frame.tag != 'frame':
                continue
            frame_index = frame.get('num')
            img_path = os.path.join(video_dir, 'img{}.jpg'.format(str(frame_index).zfill(5)))
            assert os.path.exists(img_path)
            if video_width == 0 and video_height == 0:
                img = cv2.imread(img_path)
                video_height, video_width = img.shape[0], img.shape[1]
            video_annos[frame_index] = {"filename": img_path.replace(input_base_dir, ''), "width": video_width, "height": video_height, "ann": {"bboxes": [], "labels": [], "bboxes_ignore": ignore_bboxes, "labels_ignore": [1 for _ in ignore_bboxes]}}

            targets = frame.find('target_list')
            for target in targets:
                bbox = target.find('box')
                x1, y1, w, h = float(bbox.get('left')), float(bbox.get('top')), float(bbox.get('width')), float(bbox.get('height'))
                video_annos[frame_index]['ann']['bboxes'].append([int(x1), int(y1), int(x1 + w), int(y1 + h)])
                video_annos[frame_index]['ann']['labels'].append(1)
        output_annos.extend(video_annos.values())

    return output_annos


if __name__ == '__main__':
    output_annos = ua_detrac2json()
    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

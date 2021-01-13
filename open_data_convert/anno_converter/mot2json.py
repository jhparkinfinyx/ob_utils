# MOT dataset annotation info
# Format: 0 Frame number 1 Identity number 2 Bounding Box left 3 Bounding Box top 4 Bounding Box width 5 Bounding Box height 6 Confidence score 7 Class 8 Visibility
# Labels: 0 background, 1 Pedestrian, 2 Person on vehicle, 3 Car, 4 Bicycle, 5 Motorbike, 6 Non motorized vehicle, 7 Static person, 8 Distractor, 9 Occluder, 10 Occluder on the ground, 11 Occluder full, 12 Reflection

import json
import os

import cv2
from tqdm import tqdm

input_base_dir = os.path.expanduser('~/data/cars')

mot_dir = os.path.join(input_base_dir, 'MOT')
output_json_path = os.path.join(mot_dir, 'car_classify.json')


def mot2json():
    output_annos = []
    mot_dir = os.path.join(input_base_dir, 'MOT')

    for version in tqdm(sorted(os.listdir(mot_dir)), desc='MOT'):
        version_dir = os.path.join(mot_dir, version)
        if version == '2DMOT2015' or not os.path.isdir(version_dir):  # It has a different annotation format with others
            continue

        train_dir = os.path.join(version_dir, 'train')
        for video_name in sorted(os.listdir(train_dir)):
            video_annos = {}
            video_dir = os.path.join(train_dir, video_name, 'img1')
            video_width, video_height = 0, 0

            anno_path = os.path.join(train_dir, video_name, 'gt', 'gt.txt')
            anno_data = open(anno_path, 'r')
            for line in anno_data.readlines():
                line_split = line.replace('\n', '').split(',')
                frame_index = int(line_split[0])
                label = int(line_split[7])
                if label != 3:
                    continue

                if frame_index not in video_annos.keys():
                    img_path = os.path.join(video_dir, '{}.jpg'.format(str(frame_index).zfill(6)))
                    assert os.path.exists(img_path)
                    if video_width == 0 and video_height == 0:
                        img = cv2.imread(img_path)
                        video_height, video_width = img.shape[0], img.shape[1]
                    video_annos[frame_index] = {"filename": img_path.replace(input_base_dir, ''), "width": video_width, "height": video_height, "ann": {"bboxes": [], "labels": [], "bboxes_ignore": [], "labels_ignore": []}}

                x1, y1, w, h = float(line_split[2]), float(line_split[3]), float(line_split[4]), float(line_split[5])
                video_annos[frame_index]['ann']['bboxes'].append([int(x1), int(y1), int(x1 + w), int(y1 + h)])
                video_annos[frame_index]['ann']['labels'].append(1)

            anno_data.close()
            output_annos.extend(video_annos.values())

    return output_annos


if __name__ == '__main__':
    output_annos = mot2json()
    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

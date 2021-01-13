# KITTI dataset annotation info
# Format: 0 Frame number 2 Label 6 Bounding box x1 7 Bounding box y1 8 Bounding box x2 9 Bounding box y2
# Labels(in String): 'DontCare', 'Car', 'Cyclist', 'Van', 'Truck', 'Pedestrian', 'Misc', 'Person', 'Tram'

import json
import os

import cv2
from tqdm import tqdm

input_base_dir = os.path.expanduser('~/data/cars')

imgs_dir = os.path.join(input_base_dir, 'KITTI/data_tracking_image_2/training/image_02')
anno_dir = os.path.join(input_base_dir, 'KITTI/data_tracking_label_2/training/label_02')

output_json_path = os.path.join(anno_dir, 'car_classify.json')


def kitti2json():
    output_annos = []
    imgs_dir = os.path.join(input_base_dir, 'KITTI/data_tracking_image_2/training/image_02')
    anno_dir = os.path.join(input_base_dir, 'KITTI/data_tracking_label_2/training/label_02')

    for video_name in tqdm(sorted(os.listdir(imgs_dir), key=lambda index: int(index)), desc='KITTI'):
        video_annos = {}
        video_dir = os.path.join(imgs_dir, video_name)
        video_width, video_height = 0, 0

        anno_path = os.path.join(anno_dir, '{}.txt'.format(video_name))
        anno_data = open(anno_path, 'r')
        for line in anno_data.readlines():
            line_split = line.replace('\n', '').split(' ')
            frame_index = int(line_split[0])
            label = line_split[2]
            if label not in ('Car', 'Van', 'Truck', 'DontCare'):
                continue

            if frame_index not in video_annos.keys():
                img_path = os.path.join(video_dir, '{}.png'.format(str(frame_index).zfill(6)))
                assert os.path.exists(img_path)
                if video_width == 0 and video_height == 0:
                    img = cv2.imread(img_path)
                    video_height, video_width = img.shape[0], img.shape[1]
                video_annos[frame_index] = {"filename": img_path.replace(input_base_dir, ''), "width": video_width, "height": video_height, "ann": {"bboxes": [], "labels": [], "bboxes_ignore": [], "labels_ignore": []}}

            x1, y1, x2, y2 = int(float(line_split[6])), int(float(line_split[7])), int(float(line_split[8])), int(float(line_split[9]))
            if label == 'DontCare':
                video_annos[frame_index]['ann']['bboxes_ignore'].append([x1, y1, x2, y2])
                video_annos[frame_index]['ann']['labels_ignore'].append(1)
            else:
                video_annos[frame_index]['ann']['bboxes'].append([x1, y1, x2, y2])
                video_annos[frame_index]['ann']['labels'].append(1)

        anno_data.close()
        output_annos.extend(video_annos.values())

    return output_annos


if __name__ == '__main__':
    output_annos = kitti2json()
    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

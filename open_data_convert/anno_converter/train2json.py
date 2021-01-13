# iitp_train dataset annotation info
# Format: Same with output json file but has a different labels

import json
import os

from tqdm import tqdm

input_base_dir = os.path.expanduser('~/data/cars')

imgs_dir = os.path.join(input_base_dir, 'train/train')
anno_path = os.path.join(input_base_dir, 'annotation_merge_vis_coco_filtered.json')

output_json_path = os.path.join(input_base_dir, 'train/car_classify.json')


def iipt_train2json():
    output_annos = []
    anno_path = os.path.join(input_base_dir, 'annotation_merge_vis_coco_filtered.json')

    json_data = open(anno_path, 'r')
    data_list = json.load(json_data)
    for data in tqdm(data_list, desc='train'):
        new_anno = {"bboxes": [], "labels": [], "bboxes_ignore": [], "labels_ignore": []}
        bboxes, labels = data['ann']['bboxes'], data['ann']['labels']
        bboxes_ignore, labels_ignore = data['ann']['bboxes_ignore'], data['ann']['labels_ignore']

        for bbox, label in zip(bboxes, labels):
            if int(label) != 4:
                continue
            new_anno['bboxes'].append(bbox)
            new_anno['labels'].append(1)
        for bbox_ignore, label_ignore in zip(bboxes_ignore, labels_ignore):
            if int(label) != 4:
                continue
            new_anno['bboxes_ignore'].append(bbox_ignore)
            new_anno['labels_ignore'].append(1)

        if len(new_anno['bboxes']) > 0 or len(new_anno['bboxes_ignore']) > 0:
            data['filename'] = os.path.join('/train', data['filename'])
            data['ann'] = new_anno
            output_annos.append(data)
    json_data.close()

    return output_annos


if __name__ == '__main__':
    output_annos = iipt_train2json()
    with open(output_json_path, 'w') as fp:
        json.dump(output_annos, fp)

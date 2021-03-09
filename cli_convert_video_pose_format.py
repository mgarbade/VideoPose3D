"""
Converts the aisc pose annotation format (annotation.json)
to the format required by VideoPose3D

TODO-MG: interpolates all missing keypoints -> use placeholder for missing
    keypoints instead. (eg "-100")
"""

from collections import OrderedDict

import cv2
import numpy as np
import json


def decode_annotation_json(npz_filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print('Processing {}'.format(npz_filename))
    data = np.load(npz_filename, encoding='latin1', allow_pickle=True)
    metadata = data['metadata'].item()

    # bb = data['boxes']
    # kp = data['keypoints']


    src_file = '/mnt/datasets/gelenke/gelenke_snap2__39_02_itr1540k.json'
    template_file = '/mnt/datasets/gelenke/2021-02-26_Video3D/no_interp/results3D/gelenke_snap2.mp4.npz'
    width = 1920
    height = 1080

    with open(src_file) as fp:
        person_annotations = json.load(fp, object_hook=OrderedDict)

    print("Start converting poses")
    trgt_kp_cfg = ['Nose',
                   'LeftEye',
                   'RightEye',
                   'LeftEar',
                   'RightEar',
                   'LeftShoulder',
                   'RightShoulder',
                   'LeftElbow',
                   'RightElbow',
                   'LeftWrist',
                   'RightWrist',
                   'LeftHip',
                   'RightHip',
                   'LeftKnee',
                   'RightKnee',
                   'LeftAnkle',
                   'RightAnkle']
    results_bb = []
    results_kp = []
    counter_left_eye = 0
    for frame in person_annotations:
        humans = frame['humans']

        new_pose = np.full((17, 2), np.nan, dtype=np.float32)
        new_pose[:] = np.nan

        # Has at least one human
        if len(humans) == 0:
            new_bbox = np.full(4, np.nan, dtype=np.float32)
            results_bb.append(new_bbox)
            results_kp.append(new_pose)
            continue

        # Select a single human
        human = humans[0]  # TODO-MG: Take the most centered one / tracked person

        # Get x,y coordinates of selected human
        human_dict = {}
        for bp in human['body_parts']:
            part_name = bp['part_name']
            human_dict[part_name] = [bp['x'] * width, bp['y'] * height]

        for idx, name in enumerate(trgt_kp_cfg):
            if name in human_dict.keys():
                new_pose[idx, :] = human_dict[name]
                if name == 'LeftShoulder':
                    # print("found a LeftEye")
                    counter_left_eye = counter_left_eye + 1

        # Get bbox of new pose
        new_bbox = get_bbox(human_dict)

        # Append to results
        results_bb.append(new_bbox)
        results_kp.append(new_pose)

    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        mask = ~np.isnan(kp[:, i, 0])
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

    print("Finished interpolating")
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')

    return [{
        'start_frame': 0,  # Inclusive
        'end_frame': len(kp),  # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata


def get_bbox(human_dict):
    x_locations = []
    y_locations = []
    for k, v in human_dict.items():
        x_locations.append(v[0])
        y_locations.append(v[1])
    found_keypoints = np.concatenate((np.expand_dims(np.array(x_locations), -1),
                                      np.expand_dims(np.array(y_locations), -1)), axis=1)
    bbox = cv2.boundingRect(found_keypoints.astype(np.int))
    return bbox

if __name__ == '__main__':
    decode_annotation_json()

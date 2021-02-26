#!/bin/bash

#input_dir=/mnt/datasets/fitness/2021-02-25_mg_exercising_medium/
#output_dir=/mnt/datasets/fitness/2021-02-25_mg_exercising_medium/results3D
#input_video=vid.mp4

#input_video=FitnessSnap2_complete_small_na_interp.mp4
#input_dir=/mnt/datasets/fitness/2021-02-26-Youtube-Videos-Snap2/
#output_dir=${input_dir}results3D/

input_video=gelenke_snap2.mp4
input_dir=/mnt/datasets/gelenke/2021-02-26_Video3D/no_interp/
output_dir=${input_dir}results3D/

mkdir $output_dir

#python inference/infer_video_d2.py     \
#                        --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml  \
#                        --output-dir $output_dir \
#                        --image-ext mp4   \
#                          $input_dir

cd data
python prepare_data_2d_custom.py -i $output_dir -o detectron2_pt_coco
cd ..

python run.py -d custom \
                -k detectron2_pt_coco \
                -arc 3,3,3,3,3 \
                -c checkpoint \
                --evaluate pretrained_h36m_detectron_coco.bin \
                --render --viz-subject $input_video \
                --viz-action custom \
                --viz-camera 0 \
                --viz-video $input_dir$input_video \
                --viz-output $input_dir${input_video::-4}_results3D.mp4 \
                --viz-size 6

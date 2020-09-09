#!/bin/bash
git pull
cd /home/huiyu/omz_demos_build/human_pose_estimation_demo/
make
cd /home/huiyu//omz_demos_build/intel64/Release/
source /opt/intel/openvino/bin/setupvars.sh
./human_pose_estimation_demo -m /opt/intel/openvino_2020.4.287/deployment_tools/tools/model_downloader/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml -i cam -d HDDL

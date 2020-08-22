// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include<cmath>
#include<string>
#include <utility>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"
//全局变量
double temppointarr[18][2];
double curpointarr[18][2];
//函数声明
double distance(double x1,double y1,double x2,double y2);//计算两点间的距离
void zerotemparr();//将临时数组重置
void zerocurarr();//将当前数组重置
double calheight();//计算人的大致高度
bool exist(double k);//判断某个关键点是否存在
//将数组重置
//计算两点间的距离
double distance(double x1,double y1,double x2,double y2){
    return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
}
void zerotemparr(){
    for(int k =0;k<18;k++){
        temppointarr[k][0]=-1.0;
        temppointarr[k][1]=-1.0;
    }
}
//将当前数组重置
void zerocurarr(){
    for(int k =0;k<18;k++){
        curpointarr[k][0]=-1.0;
        curpointarr[k][1]=-1.0;
    }
}
//计算人的大致高度
double calheight(){
    //如果脖子和臀部在的话，直接计算
    if(exist(temppointarr[1][1])&&exist(temppointarr[11][1])){
        return distance(temppointarr[1][0],temppointarr[1][1],temppointarr[11][0],temppointarr[11][1])*3;
    }else if(exist(temppointarr[1][1])&&exist(temppointarr[8][1])){
        return distance(temppointarr[1][0],temppointarr[1][1],temppointarr[8][0],temppointarr[8][1])*3;
    }
    //如果脖子和膝盖都在
    if(exist(temppointarr[1][1])&&exist(temppointarr[12][1])){
        return distance(temppointarr[1][0],temppointarr[1][1],temppointarr[12][0],temppointarr[12][1])*2;
    }else if(exist(temppointarr[1][1])&&exist(temppointarr[9][1])){
        return distance(temppointarr[1][0],temppointarr[1][1],temppointarr[9][0],temppointarr[9][1])*2;
    }
    return 0.0;
}
//判断某个点是否存在
bool exist(double k){
    return k!=-1.0;
}
namespace human_pose_estimation {
void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image,double curtime,double* pointarr) {
    CV_Assert(image.type() == CV_8UC3);
    std::cout <<"当前时间是"<< curtime/1000<<"s"<<std::endl;
    static const cv::Scalar colors[HumanPoseEstimator::keypointsNumber] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
    };
    static const std::pair<int, int> limbKeypointsIds[] = {
        {1, 2},  {1, 5},   {2, 3},
        {3, 4},  {5, 6},   {6, 7},
        {1, 8},  {8, 9},   {9, 10},
        {1, 11}, {11, 12}, {12, 13},
        {1, 0},  {0, 14},  {14, 16},
        {0, 15}, {15, 17}
    };

    const int stickWidth = 4;
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);
    //写入文件版
    // std::ofstream OpenFile("out.txt",std::ios::out|std::ios::app);
    // for (const auto& pose : poses) {
    //     CV_Assert(pose.keypoints.size() == HumanPoseEstimator::keypointsNumber);
    //     OpenFile<<"第"<<i+1<<"个人"<<'\n';
    //     for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
    //         if (pose.keypoints[keypointIdx] != absentKeypoint) {
    //             cv::circle(image, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);
    //             OpenFile<<"typeid="<<pose.keypoints[keypointIdx].x<<","<<pose.keypoints[keypointIdx].y<<'\n';
    //             pointlist[keypointIdx][0] = pose.keypoints[keypointIdx].x;
    //             pointlist[keypointIdx][1] = pose.keypoints[keypointIdx].y;
    //         }
    //     }
    //     i++;
    // }
    // OpenFile<<"-----opline-----"<<'\n';	
    // OpenFile.close();
    //直接打印版
    //将两个数组初始化
    zerotemparr();
    double maxheight=0;//当前检测到的最大身高,如果不超出太多的话则会记入
    for (const auto& pose : poses) {
        CV_Assert(pose.keypoints.size() == HumanPoseEstimator::keypointsNumber);
        for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
            if (pose.keypoints[keypointIdx] != absentKeypoint) {
                cv::circle(image, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);
                //将当前读取的点的数据放入临时数组
                temppointarr[keypointIdx][0] = pose.keypoints[keypointIdx].x;
                temppointarr[keypointIdx][1] = pose.keypoints[keypointIdx].y;
                // pointarr[keypointIdx*2] = pose.keypoints[keypointIdx].x;
                // pointarr[keypointIdx*2+1] = pose.keypoints[keypointIdx].y;
            }
        }
        //现在只看一个人的状态,需要过滤掉其他人的干扰
        //估算人的大致身高
        double tempheight = calheight();
        if(maxheight!=0.0){
            //因为人在移动后身高会随着离摄像头的距离改变而改变
            if(tempheight>0.9*maxheight&&tempheight<1.1*maxheight){
                maxheight = tempheight;
            }
        }else{
            maxheight = tempheight;
            for(int k=0;k<18;k++){
                pointarr[k*2] = temppointarr[k][0];
                pointarr[k*2+1] = temppointarr[k][1];
            }
        }
    }
    //std::cout<<"估计身高为:"<<maxheight<<std::endl;
    cv::Mat pane = image.clone();
    for (const auto& pose : poses) {
        for (const auto& limbKeypointsId : limbKeypointsIds) {
            std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                    pose.keypoints[limbKeypointsId.second]);
            if (limbKeypoints.first == absentKeypoint
                    || limbKeypoints.second == absentKeypoint) {
                continue;
            }

            float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
            float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
            cv::Point difference = limbKeypoints.first - limbKeypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                             angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
        }
    }
    cv::addWeighted(image, 0.4, pane, 0.6, 0, image);
}
}  // namespace human_pose_estimation

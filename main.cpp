// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/
#include<cmath>
#include <string>
#include <vector>
#include <chrono>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;
//全局变量
int timelength = 20;
bool falldownflag=false;//判断是否摔倒了
double* pointarr = new double[36];
double** timescale = new double*[20];//时空循环指针数组
char globaltemp[10];//用于返回关节编号对应的数据
int arrlength=0;//记录读取的数组数
//函数声明
char* translate(int k);//将关节编号翻译成数据
double distances(double x1,double y1,double x2,double y2);//计算两点间的距离
double getdeg(double x1,double y1,double x2,double y2);//获取与地面之间的夹角
void takemeddetect();//服药检测
void weightmovetest();//重心移动检测
//获取与地面之间的夹角
double getdeg(double x1,double y1,double x2,double y2){
    if((x1-x2)!=0){
        return atan((y2-y1)/abs(x1-x2))*180/3.1415;
    }else{
        //直角
        return 90;
    }
}

double distances(double x1,double y1,double x2,double y2){
    return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
}
//静态姿势跌倒判断
void falltest(){
    //获取关键点
    double top_x;
    double top_y;
    double weight_x;
    double weight_y;
    double foot_x;
    double foot_y;
    bool useknee=false;
    //头部代表
    if(pointarr[0]>0){//优先用鼻子
        top_x = pointarr[0];
        top_y = pointarr[1];
    }else if(pointarr[28]>0&&pointarr[30]>0){//鼻子没有就用眼睛的对称点
        top_x = (pointarr[28]+pointarr[30])/2;
        top_y = (pointarr[29]+pointarr[31])/2;
    }else if(pointarr[32]>0&&pointarr[34]>0){//鼻子眼睛都没有就用耳朵
        top_x = (pointarr[32]+pointarr[34])/2;
        top_y = (pointarr[33]+pointarr[35])/2;
    }else if(pointarr[10]>0&&pointarr[4]>0){//在没有就用肩膀
        top_x = (pointarr[10]+pointarr[4])/2;
        top_y = (pointarr[11]+pointarr[5])/2;
    }else{//都没有就没得判断
        //std::cout<<"头部数据不足，静态无法判断"<<std::endl;
    std::cout<<"检测中......."<<std::endl;
        return;
    }
    //重心代表
    if(pointarr[11*2]>0&&pointarr[8*2]>0){//如果左右臀都在同时使用
        weight_x = (pointarr[11*2]+pointarr[8*2])/2;
        weight_y = (pointarr[11*2+1]+pointarr[8*2+1])/2;
    }else if(pointarr[11*2]>0){//只有左臀
        weight_x = pointarr[11*2];
        weight_y = pointarr[11*2+1];
    }else if(pointarr[8*2]>0){//只有右臀
        weight_x = pointarr[8*2];
        weight_y = pointarr[8*2+1];
    }else{
        //std::cout<<"重心数据不足，静态无法判断"<<std::endl;
    std::cout<<"检测中......."<<std::endl;
        return;
    }
    //脚部代表
    if(pointarr[13*2]>0&&pointarr[10*2]>0){
        foot_x = (pointarr[13*2]+pointarr[10*2])/2;
        foot_y = (pointarr[13*2+1]+pointarr[10*2]+1)/2;
    }else if(pointarr[13*2]>0){
        foot_x = pointarr[13*2];
        foot_y = pointarr[13*2+1];
    }else if(pointarr[10*2]>0){
        foot_x = pointarr[10*2];
        foot_y = pointarr[10*2+1];
    }else if(pointarr[24]>0&&pointarr[18]>0){//没有脚，用膝盖
        foot_x = (pointarr[24]+pointarr[18])/2;
        foot_y = (pointarr[25]+pointarr[19])/2;
        useknee = true;
    }else if(pointarr[24]>0){
        foot_x = pointarr[24];
        foot_y = pointarr[25];
        useknee = true;
    }else if(pointarr[18]>0){
        foot_x = pointarr[18];
        foot_y = pointarr[19];
        useknee = true;
    }else{
        //std::cout<<"脚部数据不足，静态无法判断"<<std::endl;
    std::cout<<"检测中......."<<std::endl;
        return;
    }
    double d1;
    double d2;
    if(useknee){
        d1 = distances(top_x,top_y,weight_x,weight_y);
        d2 = 2*distances(weight_x,weight_y,foot_x,foot_y);
    }else{
        d1 = distances(top_x,top_y,weight_x,weight_y);
        d2 = distances(weight_x,weight_y,foot_x,foot_y);
    }
    double p = d1/d2;
    double deg1 = getdeg(top_x,top_y,weight_x,weight_y);
    double deg2 = getdeg(weight_x,weight_y,foot_x,foot_y);
    if(p<1.35&&p>0.9){
        //std::cout<<"处于站立状态"<<std::endl;
    }else if(p<2.35&&p>=1.35){
        //std::cout<<"处于坐下状态"<<std::endl;
    }else if(p<3.5&&p>=2.35){
        //std::cout<<"处于蹲下状态"<<std::endl;
    }
    if(deg1<25||deg2<25){
        std::cout<<"跌倒了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        falldownflag = true;
    }else{
    std::cout<<"检测中......."<<std::endl;
        falldownflag = false;
    }
}
//重心移动检测
void weightmovetest(){
}
//服药检测
void takemeddetect(){
    if(pointarr[6]>0&&pointarr[8]>0&&pointarr[28]>0){
        //右肘到右腕的向量与地面的夹角
        double deg1 = getdeg(pointarr[6],pointarr[7],pointarr[8],pointarr[9]);
        //从左眼到右肘的向量与地面的夹角
        double deg2 = getdeg(pointarr[6],pointarr[7],pointarr[28],pointarr[29]);
        if(abs(deg1-deg2)<15){
            std::cout<<"正在服药"<<std::endl;
        }else{
            std::cout<<"检测中......"<<std::endl;
        }
    }else{
        //std::cout<<"数据不足"<<std::endl;
    }
}
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    for(int k=0;k<20;k++){
        double* temppoint = new double[37];//一个包含37个浮点数的数组，记录18个关键点的坐标，以及测量改点的时间结点
        timescale[k] = temppoint;
    }
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        int delay = 33;

        // read input (video) frame
        cv::Mat curr_frame; cap >> curr_frame;
        cv::Mat next_frame;
        if (!cap.grab()) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        estimator.reshape(curr_frame);  // Do not measure network reshape, if it happened

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key" << std::endl;
            std::cout << "To pause execution, switch to the output window and press 'p' key" << std::endl;
        }
        std::cout << std::endl;

        cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
        Presenter presenter(FLAGS_u, static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) - graphSize.height - 10, graphSize);
        std::vector<HumanPose> poses;
        bool isLastFrame = false;
        bool isAsyncMode = false; // execution is always started in SYNC mode
        bool isModeChanged = false; // set to true when execution mode is changed (SYNC<->ASYNC)
        bool blackBackground = FLAGS_black;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double render_time = 0;

        //while循环中处理视频数据 
        while (true) {
            //记录开始工作的时刻点
            auto t0 = std::chrono::high_resolution_clock::now();
            //here is the first asynchronus point:
            //in the async mode（异步模式） we capture frame to populate（填入） the NEXT infer request
            //一般都不为异步模式，所以 isAsyncMode==0
            //in the regular mode（一般模式） we capture frame to the current infer request
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true; //end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }

            //非异步，此段不执行
            if (isAsyncMode) {
                std::cout<<"some"<<std::endl;
                if (isModeChanged) {
                    estimator.frameToBlobCurr(curr_frame);
                }
                if (!isLastFrame) {
                    estimator.frameToBlobNext(next_frame);
                }
            } else if (!isModeChanged) {
                estimator.frameToBlobCurr(curr_frame);
            }

            //记录时间点，推算出解码时间
            auto t1 = std::chrono::high_resolution_clock::now();
            double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            
            //记录解码完成后的时间点
            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the trully Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            
            //非异步，此段不执行
            if (isAsyncMode) {
                if (isModeChanged) {
                    estimator.startCurr();
                }
                if (!isLastFrame) {
                    estimator.startNext();
                }
            } else if (!isModeChanged) {
                estimator.startCurr();
            }
            
            //下面这一大段主要是计算性能，并打印出来
            if (estimator.readyCurr()) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);
                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();

                //显示性能面板，执行以下打印代码
                if (!FLAGS_no_show) {
                    if (blackBackground) {
                        curr_frame = cv::Mat::zeros(curr_frame.size(), curr_frame.type());
                    }
                    std::ostringstream out;
                    //模型推断所花时间
                    out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                        << (decode_time + render_time) << " ms";

                    cv::putText(curr_frame, out.str(), cv::Point2f(0, 25),
                                cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                    out.str("");
                    out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                    out << std::fixed << std::setprecision(2) << wall.count()
                        << " ms (" << 1000.f / wall.count() << " fps)";
                    cv::putText(curr_frame, out.str(), cv::Point2f(0, 50),
                                cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                    if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                        out.str("");
                        //模型检测所花时间
                        out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                        cv::putText(curr_frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                            cv::Scalar(255, 0, 0));
                    }
                }
		        //此处调用骨架推断模型
                poses = estimator.postprocessCurr();

                //FLAGS_r==false,此段代码不执行
                if (FLAGS_r) {
                    if (!poses.empty()) {
                        std::time_t result = std::time(nullptr);
                        char timeString[sizeof("2020-01-01 00:00:00: ")];
                        std::strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S: ", std::localtime(&result));
                        std::cout << timeString;
                     }

                    for (HumanPose const& pose : poses) {
                        std::stringstream rawPose;
                        rawPose << std::fixed << std::setprecision(0);
                        for (auto const& keypoint : pose.keypoints) {
                            rawPose << keypoint.x << "," << keypoint.y << " ";
                        }
                        rawPose << pose.score;
                        std::cout << rawPose.str() << std::endl;
                    }
                }

                //此处根据模型推断结果绘制图像，并在视频上显示
                if (!FLAGS_no_show) {
                    presenter.drawGraphs(curr_frame);
                    auto nowtime = std::chrono::high_resolution_clock::now();
                    ms curtime = std::chrono::duration_cast<ms>(nowtime - total_t0);
                    //此处记录关键点
                    double* temppoint = timescale[timelength];
                    timelength++;
                    timelength = timelength%20;
                    for(int k=0;k<36;k++){
                        temppoint[k]=pointarr[k];
                    }
                    temppoint[36]=curtime.count();//最后记录时间结点
                    //std::cout << "Detection time  : " << std::fixed << std::setprecision(2) << .count()<<std::endl;
                    renderHumanPose(poses, curr_frame,curtime.count(),pointarr);
                    cv::imshow("Human Pose Estimation on " + FLAGS_d, curr_frame);
                    t1 = std::chrono::high_resolution_clock::now();
                    render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
                }
                // weightmovetest();//重心移动检测
                falltest();//静态图像检测
                //takemeddetect();//服药检测
            }

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                estimator.swapRequest();
            }

	        //键盘控制函数
            const int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (27 == key) { // Esc
                break;
            } else if (9 == key) { // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            } else if (32 == key) { // Space
                blackBackground ^= true;
            }
            presenter.handleKey(key);
        }

        //计算并打印总视频推断时间（摄像头不执行）
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;
        std::cout << presenter.reportMeans() << '\n';
    }
    //错误处理
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
//将数字翻译成对应的部位
char* translate(int k){
    switch(k){
        case 0:
        strcpy(globaltemp, "鼻子");
        break;
        case 1:
        strcpy(globaltemp, "脖子");
        break;
        case 2:
        strcpy(globaltemp, "右肩");
        break;
        case 3:
        strcpy(globaltemp, "右肘");
        break;
        case 4:
        strcpy(globaltemp, "左腕");
        break;
        case 5:
        strcpy(globaltemp, "左肩");
        break;
        case 6:
        strcpy(globaltemp, "右肘");
        break;
        case 7:
        strcpy(globaltemp, "左腕");
        break;
        case 8:
        strcpy(globaltemp, "右臀");
        break;
        case 9:
        strcpy(globaltemp, "右膝");
        break;
        case 10:
        strcpy(globaltemp, "右踝");
        break;
        case 11:
        strcpy(globaltemp, "左臀");
        break;
        case 12:
        strcpy(globaltemp, "左膝");
        break;
        case 13:
        strcpy(globaltemp, "左踝");
        break;
        case 14:
        strcpy(globaltemp, "右眼");
        break;
        case 15:
        strcpy(globaltemp, "左眼");
        break;
        case 16:
        strcpy(globaltemp, "右耳");
        break;
        case 17:
        strcpy(globaltemp, "左耳");
        break;
    }
    return globaltemp;
}

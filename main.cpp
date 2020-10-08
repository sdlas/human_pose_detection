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
#include <iostream>
#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;
//全局变量
int timelength = 0;
double* pointarr = new double[36];
double*** timescale = new double**[20];//时空循环指针数组
double* timearr = new double[20];//记录与时空循环指针数组对应的时间节点
double** persons = new double*[5];//最多同时读取三人的信息
const int personnum = 5;//同时读取的人的数量
char globaltemp[10];//用于返回关节编号对应的数据
int arrlength=0;//记录读取的数组数
//函数声明
char* translate(int k);//将关节编号翻译成数据
double distances(double x1,double y1,double x2,double y2);//计算两点间的距离
double getdeg(double x1,double y1,double x2,double y2);//获取与地面之间的夹角
void takemeddetect();//服药检测
void weightmovetest();//重心移动检测
bool getweightpoint(double* temppointarr);//获取重心坐标
double weightpoint[2];//重心坐标
double getscale(double* temppointarr);//获取scale值
bool isusable(double* arr);//判断一组数据是否无效
bool isusable(double* arr){
    for(int k=0;k<36;k++){
        if(arr[k]!=-1) return true;
    }
    return false;
}
//获取scale值
double getscale(double* temppointarr){
    return temppointarr[36];
}
//flags
bool speedfallflag[5];//按重心下降速度判断是否摔倒
bool posefallflag[5];//按姿势判断是否摔倒
bool falldownflag=false;//判断是否摔倒了
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
//获取重心坐标
bool getweightpoint(double* temppointarr){
    //重心代表
    if(temppointarr[11*2]>0&&temppointarr[8*2]>0){//如果左右臀都在同时使用
        weightpoint[0] = (temppointarr[11*2]+temppointarr[8*2])/2;
        weightpoint[1] = (temppointarr[11*2+1]+temppointarr[8*2+1])/2;
    }else if(temppointarr[11*2]>0){//只有左臀
        weightpoint[0] = temppointarr[11*2];
        weightpoint[1] = temppointarr[11*2+1];
    }else if(temppointarr[8*2]>0){//只有右臀
        weightpoint[0] = temppointarr[8*2];
        weightpoint[1] = temppointarr[8*2+1];
    }else{
    //std::cout<<"检测中......."<<std::endl;
        return false;
    }
    return true;
}
//静态姿势跌倒判断
void falltest(double* temppointarr,int id){
    //获取关键点
    double top_x;
    double top_y;
    double weight_x;
    double weight_y;
    double foot_x;
    double foot_y;
    bool useknee=false;
    //头部代表
    if(temppointarr[0]>0){//优先用鼻子
        top_x = temppointarr[0];
        top_y = temppointarr[1];
    }else if(temppointarr[28]>0&&temppointarr[30]>0){//鼻子没有就用眼睛的对称点
        top_x = (temppointarr[28]+temppointarr[30])/2;
        top_y = (temppointarr[29]+temppointarr[31])/2;
    }else if(temppointarr[32]>0&&temppointarr[34]>0){//鼻子眼睛都没有就用耳朵
        top_x = (temppointarr[32]+temppointarr[34])/2;
        top_y = (temppointarr[33]+temppointarr[35])/2;
    }else if(temppointarr[10]>0&&temppointarr[4]>0){//在没有就用肩膀
        top_x = (temppointarr[10]+temppointarr[4])/2;
        top_y = (temppointarr[11]+temppointarr[5])/2;
    }else{//都没有就没得判断
        //std::cout<<"检测中......."<<std::endl;
        return;
    }
    //获取重心
    if(!getweightpoint(temppointarr)) return;
    else{
        weight_x = weightpoint[0];
        weight_y = weightpoint[1];
    }
    //脚部代表
    if(temppointarr[13*2]>0&&temppointarr[10*2]>0){
        foot_x = (temppointarr[13*2]+temppointarr[10*2])/2;
        foot_y = (temppointarr[13*2+1]+temppointarr[10*2]+1)/2;
    }else if(temppointarr[13*2]>0){
        foot_x = temppointarr[13*2];
        foot_y = temppointarr[13*2+1];
    }else if(temppointarr[10*2]>0){
        foot_x = temppointarr[10*2];
        foot_y = temppointarr[10*2+1];
    }else if(temppointarr[24]>0&&temppointarr[18]>0){//没有脚，用膝盖
        foot_x = (temppointarr[24]+temppointarr[18])/2;
        foot_y = (temppointarr[25]+temppointarr[19])/2;
        useknee = true;
    }else if(temppointarr[24]>0){
        foot_x = temppointarr[24];
        foot_y = temppointarr[25];
        useknee = true;
    }else if(temppointarr[18]>0){
        foot_x = temppointarr[18];
        foot_y = temppointarr[19];
        useknee = true;
    }else{
        //std::cout<<"脚部数据不足，静态无法判断"<<std::endl;
        //std::cout<<"检测中......."<<std::endl;
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
    if(deg1<25&&deg2<25){
        posefallflag[id] = true;
    }else{
        posefallflag[id] = false;
    }
}
//多人检测，现在最多三个
void falltests(){
    for(int k=0;k<personnum;k++){
        falltest(persons[k],k);
    }
}
//重心移动检测
void weightmovetest(){
    if(arrlength<6) return; 
    double** nowpoint = timescale[(timelength+19)%20];
    double** agopoint = timescale[(timelength+14)%20];
    double timetin = timearr[(timelength+14)%20]-timearr[(timelength+19)%20];
    //每个人单独判断
    for(int k=0;k<personnum;k++){
        if(!getweightpoint(nowpoint[k])){
            std::cout<<"并没有获取现在的重心"<<tempheight<<std::endl;
        }
        double now_y=weightpoint[1];
        if(!getweightpoint(agopoint[k])){
            std::cout<<"并没有获取10帧前的重心"<<tempheight<<std::endl;
        }
        double ago_y=weightpoint[1];
        double y_move = ago_y*getscale(agopoint[k])-now_y*getscale(nowpoint[k]);
        double speed = y_move/100/(timetin/1000);
        if(speed>1.37){
            speedfallflag[k] = true;
        }else{
            speedfallflag[k] = false;
        }
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
    //数据初始化
    for(int k=0;k<20;k++){
        double** temppoint = new double*[5];
        //再给每个人申请空间
        for(int j=0;j<personnum;j++){
            double* temppersonpoint = new double[37];
            temppoint[j] = temppersonpoint;
        }
        timescale[k] = temppoint;
    }
    for (int k=0;k<personnum;k++){
        posefallflag[k] = false;
        speedfallflag[k] = false;
        double* temppoint = new double[37];//一个包含37个浮点数的数组，记录18个关键点的坐标，以及,画面比例转化比，测量改点的时间结点
        persons[k] = temppoint;
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
                    double** personpoint = timescale[timelength];
                    timearr[timelength] = curtime.count();//最后记录时间结点
                    timelength++;
                    timelength = timelength%20;
                    arrlength++;
                    
                    //std::cout << "Detection time  : " << std::fixed << std::setprecision(2) << .count()<<std::endl;
                    renderHumanPose(poses, curr_frame,curtime.count(),persons);
                    int usablenum=0;//可用数据数量
                    for(int k=0;k<personnum;k++){
                        double* temppointarr = persons[k];
                        if(!isusable(temppointarr)) break;
                        usablenum++;
                    }
                    //将persons里的数据传进timescale的一个时刻里
                    for(int k=0;k<usablenum;k++){
                        double* temppointarr = personpoint[k];
                        double* tempperson = persons[k];
                        for(int j=0;j<37;j++){
                            temppointarr[j] = tempperson[j];
                        }
                    }

                    //判断是否跌倒
                    bool tempposefallflag = false;
                    bool tempspeedfallflag = false;
                    for(int k=0;k<usablenum;k++){
                        if(posefallflag[k]){
                            std::cout<<"姿势确认摔倒了!"<<std::endl;
                            tempposefallflag = true;
                        } 
                        if(speedfallflag[k]){
                            std::cout<<"重心下降确认摔倒了!"<<std::endl;
                            tempposefallflag = true;
                        } 
                    }
                    if(tempposefallflag||tempspeedfallflag){
                        falldownflag = true;
                        //向txt写入跌倒
                        std::cout<<"确认摔倒了!"<<std::endl;
                        std::ofstream OpenFile("flag.txt",std::ios::out);
                        OpenFile<<"1"<<'\n';
                        OpenFile.close();
                    }else{
                        //向txt写入正常
                        falldownflag = false;
                        std::ofstream OpenFile("flag.txt",std::ios::out);
                        OpenFile<<"0"<<'\n';
                        OpenFile.close();
                    }
                    cv::imshow("Human Pose Estimation on " + FLAGS_d, curr_frame);
                    t1 = std::chrono::high_resolution_clock::now();
                    render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
                }
                // weightmovetest();//重心移动检测
                falltests();//静态图像检测
                weightmovetest();
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

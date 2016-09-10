#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
using std::string;

#include "caffe_thread_learn.hpp"

class VideoCaptureTest : public InternalThread {
public:
    string video;
    explicit VideoCaptureTest(string v) : video(v) { fun(); StartInternalThread(); }
protected:
    virtual void InternalThreadEntry();
    virtual void fun() { std::cout << "child" << std::endl;}
};

void VideoCaptureTest::InternalThreadEntry(){
    std::cout << "video child" << std::endl;
}

int main(){
    
    InternalThread* vt = new VideoCaptureTest("/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    delete vt;

    return 0;
}
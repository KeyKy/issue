//
//  caffe_thread_learn.cpp
//  c++learn
//
//  Created by meitu on 16/9/10.
//  Copyright © 2016年 meitu. All rights reserved.
//

#include "caffe_thread_learn.hpp"


InternalThread::~InternalThread() {
    StopInternalThread();
}

bool InternalThread::is_started() const {
    return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
    return thread_ && thread_->interruption_requested();
}

void InternalThread::StopInternalThread() {
    if (is_started()) {
        thread_->interrupt();
        try {
            std::cout << "join" << std::endl;
            thread_->join();
        } catch (boost::thread_interrupted&) {
        } catch (std::exception& e) {
            std::cout << "Thread exception: " << e.what();
        }
    }
}


void InternalThread::StartInternalThread() {
    thread_.reset(new boost::thread(&InternalThread::entry, this));
}

void InternalThread::entry() {
    InternalThreadEntry();
}
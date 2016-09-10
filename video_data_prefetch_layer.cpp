
#include <opencv2/core/core.hpp>

#include <fstream>
#include <vector>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "boost/algorithm/string.hpp"

#include <opencv2/opencv.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/data_transformer.hpp"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/layers/video_data_prefetch_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    VideoDataPrefetchLayer<Dtype>::~VideoDataPrefetchLayer<Dtype>(){
        this->StopInternalThread();
    }
    
    template <typename Dtype>
    void VideoDataPrefetchLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
        batch_size_ = this->layer_param_.video_data_prefetch_param().batch_size();
        channels_ = this->layer_param_.video_data_prefetch_param().channels();
        video_folder = this->layer_param_.video_data_prefetch_param().video_folder();
        target_size = this->layer_param_.video_data_prefetch_param().target_size();
        clips_per_video = this->layer_param_.video_data_prefetch_param().clips_per_video();
        
        LOG(INFO) << "batch_size_:" << batch_size_;
        LOG(INFO) << "channels_:" << channels_;
        
        const string& source = this->layer_param_.video_data_prefetch_param().source();
        LOG(INFO) << "Opening file " << source;
        
        std::ifstream infile(source.c_str());
        string line; vector<string> spl; int count = 0;
        while(std::getline(infile, line)){
            boost::filesystem::path video_folder(this->layer_param_.video_data_prefetch_param().video_folder());
            perm.push_back(count);
            boost::split(spl, line, boost::is_any_of(" "));
            vector<string> tmp;
            boost::split(tmp, spl[1], boost::is_any_of(","));
            lines_.push_back(std::make_pair(video_folder.append(spl[0]), boost::lexical_cast<int>(tmp[0])));
            count += 1;
        }
        infile.close();
        
        CHECK(!lines_.empty()) << "File is empty";
        
        if(this->layer_param_.video_data_prefetch_param().shuffle()){
            LOG(INFO) << "Shuffling data";
            const unsigned int prefetch_rng_seed = caffe_rng_rand();
            prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
            ShuffleVideos();
        }
        LOG(INFO) << "A total of " << lines_.size() << " videos.";
        
        lines_id_ = 0;
        
        data_shape.push_back(batch_size_*clips_per_video);
        data_shape.push_back(channels_);
        data_shape.push_back(target_size);
        data_shape.push_back(target_size);
        
        label_shape.push_back(batch_size_*clips_per_video);
        label_shape.push_back(1);
        label_shape.push_back(1);
        label_shape.push_back(1);
        
        this->transformed_data_.Reshape(data_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(data_shape);
        }
        top[0]->Reshape(data_shape);
        LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();
        
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }
    
    template <typename Dtype>
    void VideoDataPrefetchLayer<Dtype>::ShuffleVideos(){
        caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(perm.begin(), perm.end(), prefetch_rng);
    }
    
    template <typename Dtype>
    void VideoDataPrefetchLayer<Dtype>::load_batch(Batch<Dtype> *batch){
        CPUTimer batch_timer;
        double read_time = 0;
        double trans_time = 0;
        double capture_frame_time = 0;
        batch_timer.Start();
        
        CPUTimer timer, capture_timer;
        
        CHECK(batch->data_.count());
        CHECK(this->transformed_data_.count());
        
        //Dtype* prefetch_data = batch->data_.mutable_cpu_data(); // Blob<Dtype> data_, label_;
        //Dtype* prefetch_label = batch->label_.mutable_cpu_data();
        
        const int lines_size = lines_.size();
        if (this->lines_id_ + batch_size_ > lines_size){
            DLOG(INFO) << "Restarting data prefetching from start.";
            this->lines_id_ = 0;
            if (this->layer_param_.video_data_prefetch_param().shuffle()){
                LOG(INFO) << "Shuffling data";
                ShuffleVideos();
            }
        }
        vector<cv::Mat> mat_vector;
        vector<int> ds_inds(perm.begin()+this->lines_id_, perm.begin()+this->lines_id_+batch_size_);
        for (int i = 0; i < ds_inds.size(); i++){
            boost::filesystem::path video_path = lines_[ds_inds[i]].first;
            LOG(INFO) << "opening video file" << video_path.c_str();
            
            cv::VideoCapture capture(video_path.c_str());
            CHECK(capture.isOpened()) << "Could not open " << video_path;
            
            int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
            CHECK_GE(frame_count, clips_per_video) << "frame_count " << frame_count << " must >= clips_per_video " << clips_per_video;
            LOG(INFO) << "frame_count " << frame_count;
            
            int step = frame_count / clips_per_video;
            vector<int> frame_inds; vector<int> frame_picks;
            for (int j = 0; j < frame_count; j++){
                frame_inds.push_back(j);
            }
            for (int j = 0, start = 0; j < clips_per_video; j++, start+=step){
                shuffle(frame_inds.begin()+start, frame_inds.begin()+start+step);
                frame_picks.push_back(*(frame_inds.begin()+start));
            }
            timer.Start();
            for (int j = 0; j < clips_per_video; j++){
                LOG(INFO) << "extract frames in clips " << j;
                cv::Mat frame;
                int frame_current = frame_picks[j];
                capture_timer.Start();
                capture.set(cv::CAP_PROP_POS_FRAMES, frame_current);
                LOG(INFO) << "read frame";
                capture >> frame;
                LOG(INFO) << "read finish";
                capture_frame_time += capture_timer.MicroSeconds();
                CHECK(!frame.empty()) << "Could not read video " << video_path << " frame " << frame_current;
                int frame_height = frame.rows; int frame_width = frame.cols;
                int im_size_min = frame_height > frame_width ? frame_width : frame_height;
                float im_scale = float(target_size) / float(im_size_min);
                cv::Mat resized_frame;
                //LOG(INFO) << "unresized frame" << frame.rows << "_" << frame.cols;
                cv::resize(frame, resized_frame, cv::Size(), im_scale, im_scale);
                //LOG(INFO) << "resize frame " << resized_frame.rows << "_" << resized_frame.cols;
                mat_vector.push_back(resized_frame);
            }
            read_time += timer.MicroSeconds();
            capture.release();
        }
        timer.Start();
        this->data_transformer_->Transform(mat_vector, &batch->data_);
        //LOG(INFO) << "Tranformed data";
        trans_time += timer.MicroSeconds();
        
        Dtype* prefetch_label = batch->label_.mutable_cpu_data();
        for(int j = 0; j < ds_inds.size(); j++){
            int label_ = lines_[ds_inds[j]].second;
            for(int k = 0; k < clips_per_video; k++){
                prefetch_label[j*clips_per_video + k] = label_;
            }
        }
        //LOG(INFO) << "Tranformed label";
        this->lines_id_ += batch_size_;
        
        
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
        DLOG(INFO) << "  Capture time: " << capture_frame_time / 1000 << " ms.";
    }
    
    INSTANTIATE_CLASS(VideoDataPrefetchLayer);
    REGISTER_LAYER_CLASS(VideoDataPrefetch);
}
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        



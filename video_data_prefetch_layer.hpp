

#ifndef CAFFE_VIDEO_DATA_PREFETCH_LAYER_HPP_
#define CAFFE_VIDEO_DATA_PREFETCH_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template <typename Dtype>
    class VideoDataPrefetchLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit VideoDataPrefetchLayer(const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype>(param) {}
        virtual ~VideoDataPrefetchLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const { return "VideoDataPrefetchLayer"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 2; }
        
    protected:
        shared_ptr<Caffe::RNG> prefetch_rng_;
        vector<int> perm;
        virtual void ShuffleVideos();
        virtual void load_batch(Batch<Dtype>* batch);
        
        vector<std::pair<boost::filesystem::path, int> > lines_;
        int lines_id_;
        int batch_size_, channels_, clips_per_video, target_size;
        string video_folder, source;
        
        vector<int> label_shape;
        vector<int> data_shape;
    };
}


#endif /* CAFFE_VIDEO_DATA_PREFETCH_LAYER_HPP_ */

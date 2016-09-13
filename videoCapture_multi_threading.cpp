#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
using std::string;
void fun(string path){
    cv::VideoCapture cap(path);
    if(cap.isOpened()){
        while(true){
            cv::Mat frame;
            cap.read(frame);
            if(frame.empty())
                break;
            
        }
    }
    cap.release();
}
int main(){
    boost::thread trd1(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    boost::thread trd2(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    boost::thread trd3(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    boost::thread trd4(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    boost::thread trd5(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    boost::thread trd6(fun, "/Users/zj-db0655/Documents/data/528100078_5768b1b1764438418.mp4");
    trd1.join();
    trd2.join();
    trd3.join();
    trd4.join();
    trd5.join();
    trd6.join();

    return 0;
}


#ifndef caffe_thread_learn_hpp
#define caffe_thread_learn_hpp

#include <stdio.h>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

namespace boost { class thread; }

class InternalThread {
public:
    InternalThread() : thread_() {}
    virtual ~InternalThread();
    
    /**
     * Caffe's thread local state will be initialized using the current
     * thread values, e.g. device id, solver index etc. The random seed
     * is initialized using caffe_rng_rand.
     */
    void StartInternalThread();
    
    /** Will not return until the internal thread has exited. */
    void StopInternalThread();
    
    bool is_started() const;
    
protected:
    /* Implement this method in your subclass
     with the code you want your thread to run. */
    virtual void InternalThreadEntry() { std::cout << "parant" << std::endl; }
    virtual void fun() {}
    
    /* Should be tested when running loops to exit when requested. */
    bool must_stop();
    
private:
    void entry();
    
    boost::shared_ptr<boost::thread> thread_;
};

#endif /* caffe_thread_learn_hpp */

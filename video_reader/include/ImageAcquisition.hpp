#ifndef IMAGE_ACQUISITION_HPP
#define IMAGE_ACQUISITION_HPP

#include <chrono>
#include <cmath>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <atomic>
#include <iostream>
#include <condition_variable>
#include <iomanip>
#include <queue>
#include <sstream>
#include <ctime>
#include <memory>

#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

// Use shared_ptr to avoid deep copying of frames
using FramePtr = std::shared_ptr<cv::Mat>;

class ImageAcquisition
{
private:
    // color for the terminals
    std::string green = "\033[1;32m";
    std::string red = "\033[1;31m";
    std::string blue = "\033[1;34m";
    std::string yellow = "\033[1;33m";
    std::string purple = "\033[1;35m";
    std::string reset = "\033[0m";

    // camera parameters
    int fps_;
    std::string rtsp_url_;
    bool debug_mode_;
    bool use_video_file_;
    std::string video_file_;
    bool headless_mode_ = false;

    // Optimized threading architecture
    std::thread capture_thread_;
    std::thread display_thread_; // Removed separate processing thread
    std::atomic<bool> capturing_{false};

    // Performance monitoring
    std::atomic<int> frames_dropped_{0};
    std::atomic<int> frames_processed_{0};

    // Reduced queue sizes and using shared_ptr to avoid copying
    std::queue<FramePtr> frame_queue_; // Reduced to single queue
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_available_;

    // Frame reuse pool to avoid constant allocation/deallocation
    std::queue<FramePtr> frame_pool_;
    std::mutex pool_mutex_;

    // Image processing parameters
    cv::Size target_size_{512, 512};
    double target_ratio_;
    double x_offset_{0.0};
    double y_offset_{0.0};

    // Performance optimization settings
    static constexpr size_t MAX_QUEUE_SIZE = 3; // Reduced from 3
    static constexpr size_t POOL_SIZE = 5;      // Pre-allocated frames
    static constexpr int SKIP_FRAMES = 1;       // Process every N frames

    // Adaptive frame skipping
    std::atomic<int> frame_counter_{0};
    std::chrono::steady_clock::time_point last_process_time_;

    std::function<void(std::shared_ptr<cv::Mat>)> frame_callback_;

    std::atomic<double> source_fps_{30.0};

public:
    ImageAcquisition(const std::string &rtsp_url = "",
                     int fps = 30,
                     bool debug_mode = false,
                     bool use_video_file = false,
                     const std::string &video_file = "");
    ~ImageAcquisition();

    static std::string maskPassword(const std::string &url);
    void displayLoop();
    void captureLoop();
    void waitForCompletion();
    void init();
    void stop();

    // Optimized helper functions
    FramePtr getPooledFrame();
    void returnFrameToPool(FramePtr frame);
    void initializeFramePool();

    // Image processing functions
    cv::Mat centerCropResize(const cv::Mat &img, double x_offset = 0.0, double y_offset = 0.0) const;
    void setTargetSize(const cv::Size &size);
    void setOffsets(double x_offset, double y_offset);

    // Performance monitoring
    void printStats() const;

    void registerFrameCallback(std::function<void(std::shared_ptr<cv::Mat>)> cb)
    {
        frame_callback_ = std::move(cb);
    }

    double getSourceFps() const { return source_fps_.load(); }
};

#endif // IMAGE_ACQUISITION_HPP
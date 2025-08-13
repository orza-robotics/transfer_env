#include "ImageAcquisition.hpp"

ImageAcquisition::ImageAcquisition(const std::string &rtsp_url,
                                   int fps,
                                   bool debug_mode,
                                   bool use_video_file,
                                   const std::string &video_file)
    : fps_(std::clamp<int>(fps, 0, 120)),
      rtsp_url_(rtsp_url),
      debug_mode_(debug_mode),
      use_video_file_(use_video_file),
      video_file_(video_file),
      last_process_time_(std::chrono::steady_clock::now())
{
    headless_mode_ = (getenv("DISPLAY") == nullptr);
    if (headless_mode_)
    {
        std::cout << yellow << "Running in headless mode (no display)" << reset << std::endl;
    }

    target_ratio_ = static_cast<double>(target_size_.height) / target_size_.width;
    initializeFramePool();

    std::cout << green << "Optimized ImageAcquisition created (reduced memory/CPU)" << reset << std::endl;
}

void ImageAcquisition::initializeFramePool()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Pre-allocate frames to avoid runtime allocation
    for (size_t i = 0; i < POOL_SIZE; ++i)
    {
        auto frame = std::make_shared<cv::Mat>();
        frame_pool_.push(frame);
    }

    if (debug_mode_)
    {
        std::cout << blue << "Frame pool initialized with " << POOL_SIZE << " frames" << reset << std::endl;
    }
}

FramePtr ImageAcquisition::getPooledFrame()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!frame_pool_.empty())
    {
        auto frame = frame_pool_.front();
        frame_pool_.pop();
        return frame;
    }

    // Pool exhausted, create new frame (should be rare)
    return std::make_shared<cv::Mat>();
}

void ImageAcquisition::returnFrameToPool(FramePtr frame)
{
    if (!frame || frame.use_count() > 1)
        return; // Still in use elsewhere

    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (frame_pool_.size() < POOL_SIZE)
    {
        frame_pool_.push(frame);
    }
    // If pool is full, let the frame be destroyed naturally
}

void ImageAcquisition::init()
{
    // Optimized RTSP settings with lower buffer sizes
    ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
             "rtsp_transport;tcp|fifo_size;1|buffer_size;256000", 1);

    capturing_ = true;

    // Start only 2 threads instead of 3 for lower resource usage
    capture_thread_ = std::thread(&ImageAcquisition::captureLoop, this);
    display_thread_ = std::thread(&ImageAcquisition::displayLoop, this);

    std::cout << green << "Optimized 2-thread acquisition started" << reset << std::endl;
}

ImageAcquisition::~ImageAcquisition()
{
    stop();
}

void ImageAcquisition::stop()
{
    capturing_ = false;
    frame_available_.notify_all();
    waitForCompletion();

    if (debug_mode_)
    {
        printStats();
    }

    std::cout << red << "Optimized acquisition stopped" << reset << std::endl;
}

void ImageAcquisition::waitForCompletion()
{
    if (capture_thread_.joinable())
        capture_thread_.join();
    if (display_thread_.joinable())
        display_thread_.join();
}

void ImageAcquisition::printStats() const
{
    std::cout << blue << "Performance Stats:" << reset << std::endl;
    std::cout << "  Frames processed: " << frames_processed_.load() << std::endl;
    std::cout << "  Frames dropped: " << frames_dropped_.load() << std::endl;

    int total = frames_processed_.load() + frames_dropped_.load();
    if (total > 0)
    {
        double drop_rate = (static_cast<double>(frames_dropped_.load()) / total) * 100.0;
        std::cout << "  Drop rate: " << std::fixed << std::setprecision(1) << drop_rate << "%" << std::endl;
    }
}

// =============== THREAD 1: OPTIMIZED CAPTURE ===============
void ImageAcquisition::captureLoop()
{
    std::cout << blue << "Starting optimized capture thread..." << reset << std::endl;

    int frames_captured = 0;
    auto stats_time = std::chrono::steady_clock::now();

    while (capturing_)
    {
        cv::VideoCapture cap;
        double video_fps = 30.0;                      // Default FPS
        std::chrono::microseconds frame_delay(33333); // Default ~30 FPS

        // Initialize capture with optimal settings
        if (use_video_file_)
        {
            cap.open(video_file_);
            if (!cap.isOpened())
            {
                std::cerr << red << "Could not open video file '" << video_file_ << "'" << reset << std::endl;
                capturing_ = false;
                frame_available_.notify_all();
                return;
            }

            // Robust FPS read
            double f = cap.get(cv::CAP_PROP_FPS);
            if (!(std::isfinite(f)) || f <= 0.0 || f > 240.0)
                f = 30.0;
            video_fps = f;
            frame_delay = std::chrono::microseconds(static_cast<long>(1000000.0 / video_fps));

            source_fps_.store(video_fps); // expose true source FPS to ROS node

            std::cout << blue << "Playing video: " << video_file_
                      << " at " << video_fps << " FPS" << reset << std::endl;
        }
        else
        {
            const std::size_t max_attempts = 3;
            std::size_t attempt = 0;
            while (capturing_ && attempt < max_attempts)
            {
                cap.open(rtsp_url_, cv::CAP_FFMPEG);

                // Ultra-low latency settings
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                cap.set(cv::CAP_PROP_FPS, fps_ > 0 ? fps_ : 30);

                if (cap.isOpened())
                {
                    std::cout << blue << "Connected to RTSP: " << maskPassword(rtsp_url_) << reset << std::endl;
                    break;
                }
                ++attempt;
                std::cerr << yellow << "RTSP failed (" << attempt << "/" << max_attempts << ")" << reset << std::endl;
                std::this_thread::sleep_for(1s);
            }
            if (!cap.isOpened())
            {
                std::cerr << red << "Failed to connect after " << max_attempts << " attempts" << reset << std::endl;
                capturing_ = false;
                frame_available_.notify_all();
                return;
            }

            // Initial guess; will be measured below
            source_fps_.store(fps_ > 0 ? static_cast<double>(fps_) : 30.0);
        }

        auto last_frame_time = std::chrono::steady_clock::now();

        const int fps_probe_window = 120;
        int window_count = 0;
        auto window_start = std::chrono::steady_clock::now();

        // OPTIMIZED CAPTURE LOOP
        while (capturing_)
        {
            // Get a pooled frame to avoid allocation
            auto frame_ptr = getPooledFrame();

            if (!cap.read(*frame_ptr) || frame_ptr->empty())
            {
                returnFrameToPool(frame_ptr); // Return unused frame

                if (!use_video_file_)
                {
                    std::cout << yellow << "Frame read failed - reconnecting" << reset << std::endl;
                    break;
                }
                else
                {
                    std::cout << yellow << "Video file ended" << reset << std::endl;
                    capturing_ = false;
                    break;
                }
            }

            frames_captured++;

            // Aggressive queue management with frame dropping
            {
                std::unique_lock<std::mutex> lock(frame_queue_mutex_);

                // If queue is full, drop the oldest frame AND current frame to prevent buildup
                if (frame_queue_.size() >= MAX_QUEUE_SIZE)
                {
                    // Return dropped frames to pool
                    while (!frame_queue_.empty())
                    {
                        returnFrameToPool(frame_queue_.front());
                        frame_queue_.pop();
                        frames_dropped_++;
                    }
                }

                frame_queue_.push(frame_ptr);
            }
            frame_available_.notify_one();

            // Performance stats (less frequent to reduce overhead)
            if (debug_mode_ && frames_captured % 300 == 0)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stats_time);
                if (elapsed.count() >= 10) // Every 10 seconds instead of 5
                {
                    double capture_fps = 300.0 / elapsed.count();
                    std::cout << green << "Capture: " << std::fixed << std::setprecision(1)
                              << capture_fps << " FPS" << reset << std::endl;
                    stats_time = now;
                }
            }

            // FIXED: Proper frame timing for both video files and RTSP
            if (use_video_file_)
            {
                // For video files: maintain original frame rate
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time);

                if (elapsed < frame_delay)
                {
                    std::this_thread::sleep_for(frame_delay - elapsed);
                }
                last_frame_time = std::chrono::steady_clock::now();
            }
            else
            {
                // For RTSP: adaptive delay based on system load (original logic)
                auto sleep_time = frame_queue_.size() > 1 ? std::chrono::milliseconds(2) : std::chrono::microseconds(500);
                std::this_thread::sleep_for(sleep_time);

                // Measure source FPS over a window and low-pass it
                ++window_count;
                if (window_count >= fps_probe_window)
                {
                    auto now = std::chrono::steady_clock::now();
                    double secs = std::chrono::duration<double>(now - window_start).count();
                    if (secs > 0.0)
                    {
                        double measured = window_count / secs;
                        double prev = source_fps_.load();
                        double blended = (prev > 0.0) ? (0.7 * prev + 0.3 * measured) : measured;
                        source_fps_.store(std::clamp(blended, 1.0, 240.0));
                    }
                    window_count = 0;
                    window_start = std::chrono::steady_clock::now();
                }
            }
        }

        cap.release();
        if (capturing_ && !use_video_file_)
        {
            frame_available_.notify_all();
            std::this_thread::sleep_for(500ms);
        }
    }

    std::cout << blue << "Capture thread finished (" << frames_captured << " frames)" << reset << std::endl;
}

// =============== THREAD 2: COMBINED PROCESSING + DISPLAY ===============
void ImageAcquisition::displayLoop()
{
    std::cout << green << "Starting combined processing+display thread..." << reset << std::endl;

    int frames_displayed = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto stats_time = start_time;

    while (capturing_)
    {
        FramePtr frame_ptr;

        // Get frame from capture queue
        {
            std::unique_lock<std::mutex> lock(frame_queue_mutex_);
            if (!frame_available_.wait_for(lock, std::chrono::milliseconds(100),
                                           [this]
                                           { return !frame_queue_.empty() || !capturing_; }))
            {
                continue; // Timeout
            }

            if (!capturing_)
                break;

            if (!frame_queue_.empty())
            {
                frame_ptr = frame_queue_.front();
                frame_queue_.pop();
            }
        }

        if (!frame_ptr || frame_ptr->empty())
            continue;

        // Adaptive frame skipping to reduce CPU load
        frame_counter_++;
        if (frame_counter_ % (SKIP_FRAMES + 1) != 0)
        {
            returnFrameToPool(frame_ptr);
            continue; // Skip this frame
        }

        // Return original frame to pool ASAP
        returnFrameToPool(frame_ptr);

        frames_displayed++;
        frames_processed_++;

        if (headless_mode_)
        {
            // Headless: performance stats only
            if (frames_displayed % 200 == 0)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stats_time);
                if (elapsed.count() >= 10)
                {
                    double display_fps = 200.0 / elapsed.count();
                    std::cout << green << "Processing: " << std::fixed << std::setprecision(1)
                              << display_fps << " FPS (total: " << frames_displayed << ")" << reset << std::endl;
                    stats_time = now;
                }
            }

            // Auto-stop for demo (increased threshold)
            if (frames_displayed > 2000)
            {
                std::cout << yellow << "Auto-stopping after " << frames_displayed << " frames" << reset << std::endl;
                capturing_ = false;
                break;
            }
        }
        else
        {
            // GUI mode: display frame
            try
            {

                auto out_ptr = std::make_shared<cv::Mat>(std::move(*frame_ptr));
                if (frame_callback_)
                {
                    frame_callback_(out_ptr);
                }
                else
                {
                    // Fallback: just show it here
                    cv::imshow("Optimized RTSP Viewer", *out_ptr);
                }

                if (!frame_callback_)
                {
                    char key = cv::waitKey(1) & 0xFF;
                    if (key == 27 || key == 'q')
                    {
                        capturing_ = false;
                        break;
                    }
                }
            }
            catch (const cv::Exception &e)
            {
                std::cerr << red << "Display error: " << e.what() << reset << std::endl;
                headless_mode_ = true;
                continue;
            }
        }

        // Update processing time for adaptive behavior
        last_process_time_ = std::chrono::steady_clock::now();
    }

    if (!headless_mode_)
    {
        cv::destroyAllWindows();
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double avg_fps = total_duration.count() > 0 ? static_cast<double>(frames_displayed) / total_duration.count() : 0;

    std::cout << green << "Display finished: " << frames_displayed << " frames in "
              << total_duration.count() << "s (avg " << std::fixed << std::setprecision(1)
              << avg_fps << " FPS)" << reset << std::endl;
}

std::string ImageAcquisition::maskPassword(const std::string &url)
{
    return std::regex_replace(url, std::regex(R"(:(.*?)@)"), ":******@");
}

// =============== OPTIMIZED IMAGE PROCESSING FUNCTIONS ===============

cv::Mat ImageAcquisition::centerCropResize(const cv::Mat &img, double x_offset, double y_offset) const
{
    if (img.empty())
        return img;

    const int h = img.rows;
    const int w = img.cols;
    const double current_ratio = static_cast<double>(h) / w;

    // Early return optimization
    if (std::abs(current_ratio - target_ratio_) < 1e-6)
    {
        cv::Mat result;
        cv::resize(img, result, target_size_, 0, 0, cv::INTER_LINEAR);
        return result;
    }

    cv::Mat cropped;

    if (current_ratio > target_ratio_)
    {
        // Crop height
        const int new_h = static_cast<int>(w * target_ratio_);
        const double scale_factor = static_cast<double>(target_size_.height) / new_h;
        const int original_y_offset = static_cast<int>(y_offset / scale_factor);
        const int center_y = h / 2;
        int start_y = std::clamp(center_y - new_h / 2 + original_y_offset, 0, h - new_h);

        cropped = img(cv::Rect(0, start_y, w, new_h));
    }
    else
    {
        // Crop width
        const int new_w = static_cast<int>(h / target_ratio_);
        const double scale_factor = static_cast<double>(target_size_.width) / new_w;
        const int original_x_offset = static_cast<int>(x_offset / scale_factor);
        const int center_x = w / 2;
        int start_x = std::clamp(center_x - new_w / 2 + original_x_offset, 0, w - new_w);

        cropped = img(cv::Rect(start_x, 0, new_w, h));
    }

    // Resize to target size
    cv::Mat result;
    cv::resize(cropped, result, target_size_, 0, 0, cv::INTER_LINEAR);
    return result;
}

void ImageAcquisition::setTargetSize(const cv::Size &size)
{
    target_size_ = size;
    target_ratio_ = static_cast<double>(target_size_.height) / target_size_.width;

    if (debug_mode_)
    {
        std::cout << blue << "Updated target size: " << target_size_.width << "x" << target_size_.height
                  << " (ratio: " << std::fixed << std::setprecision(2) << target_ratio_ << ")" << reset << std::endl;
    }
}

void ImageAcquisition::setOffsets(double x_offset, double y_offset)
{
    x_offset_ = x_offset;
    y_offset_ = y_offset;

    if (debug_mode_)
    {
        std::cout << blue << "Updated offsets: x=" << x_offset_ << ", y=" << y_offset_ << reset << std::endl;
    }
}
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
#include <chrono>

#include "ImageAcquisition.hpp"

using namespace std::chrono_literals;

class VideoReaderNode : public rclcpp::Node
{
public:
    VideoReaderNode()
        : Node("video_reader_node")
    {
        // --- Parameters (declare + get) ---
        source_url_ = this->declare_parameter<std::string>("source_url", "rtsp://admin:cC3aQxNEAQyee6k!@192.168.15.134:554/profile2/media.smp");
        fps_param_ = this->declare_parameter<int>("fps", 30);
        use_video_file_ = this->declare_parameter<bool>("use_video_file", false);
        video_path_ = this->declare_parameter<std::string>("video_path", "/workspace/videos/nucor_west_3.mkv");
        debug_mode_ = this->declare_parameter<bool>("debug_mode", false);
        frame_id_ = this->declare_parameter<std::string>("frame_id", "camera_optical_frame");
        qos_depth_ = this->declare_parameter<int>("qos_depth", 5);

        // Publisher (SensorDataQoS is often a good default for images)
        rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        qos.keep_last(qos_depth_);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);

        // Construct acquisition
        image_acq_ = std::make_unique<ImageAcquisition>(
            source_url_,
            fps_param_,
            debug_mode_,
            use_video_file_,
            video_path_);

        // Register frame callback: store latest frame thread-safely
        image_acq_->registerFrameCallback(
            [this](std::shared_ptr<cv::Mat> frame_ptr)
            {
                if (!frame_ptr || frame_ptr->empty())
                    return;

                std::lock_guard<std::mutex> lk(last_frame_mtx_);
                // store a clone so the pool can safely recycle original
                last_frame_ = frame_ptr->clone();
                last_frame_time_ = this->now();
            });

        // Start acquisition
        image_acq_->init();

        // Create publishing timer with initial period; will be updated dynamically
        double initial_fps = clampFps(image_acq_->getSourceFps());
        setTimerFromFps(initial_fps);

        // A small watcher to adjust the timer period if source FPS drifts
        fps_check_timer_ = this->create_wall_timer(
            500ms, [this]()
            { this->maybeUpdateTimer(); });

        RCLCPP_INFO(this->get_logger(), "video_reader_node started. Initial source FPS: %.2f", initial_fps);
    }

    ~VideoReaderNode() override
    {
        try
        {
            if (image_acq_)
                image_acq_->stop();
        }
        catch (...)
        {
        }
    }

private:
    // ---- Helpers ----
    static double clampFps(double f)
    {
        if (!std::isfinite(f))
            return 30.0;
        if (f < 1.0)
            return 1.0;
        if (f > 240.0)
            return 240.0;
        return f;
    }

    void setTimerFromFps(double fps)
    {
        fps = clampFps(fps);
        auto period = std::chrono::duration<double>(1.0 / fps);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(period);

        // (Re)create timer
        publish_timer_ = this->create_wall_timer(ns, [this]()
                                                 { this->publishOnce(); });

        current_timer_fps_.store(fps);
        RCLCPP_INFO(this->get_logger(), "Set publish timer to %.2f FPS (period %.3f ms)",
                    fps, 1000.0 / fps);
    }

    void maybeUpdateTimer()
    {
        double src = clampFps(image_acq_->getSourceFps());
        double cur = current_timer_fps_.load();

        // Update if difference is meaningful (> ~1%)
        if (std::fabs(src - cur) / std::max(src, 1.0) > 0.01)
        {
            setTimerFromFps(src);
        }
    }

    void publishOnce()
    {
        cv::Mat frame_copy;
        rclcpp::Time stamp;

        {
            std::lock_guard<std::mutex> lk(last_frame_mtx_);
            if (last_frame_.empty())
                return;
            frame_copy = last_frame_; // shallow copy ok
            stamp = last_frame_time_; // stamp when we received the frame
        }

        // Convert to ROS Image (assumes BGR cv::Mat)
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame_copy).toImageMsg();
        msg->header.stamp = stamp;
        msg->header.frame_id = frame_id_;

        image_pub_->publish(*msg);
    }

private:
    // Acquisition
    std::unique_ptr<ImageAcquisition> image_acq_;

    // Latest frame cache
    std::mutex last_frame_mtx_;
    cv::Mat last_frame_;
    rclcpp::Time last_frame_time_{0, 0, RCL_ROS_TIME};

    // ROS I/O
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr fps_check_timer_;

    // Params / state
    std::string source_url_;
    int fps_param_;
    bool use_video_file_;
    std::string video_path_;
    bool debug_mode_;
    std::string frame_id_;
    int qos_depth_;

    std::atomic<double> current_timer_fps_{30.0};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoReaderNode>());
    rclcpp::shutdown();
    return 0;
}

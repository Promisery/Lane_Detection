// LaneDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include <array>
#include <utility>
#include <string>
#include <vector>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <boost/progress.hpp>

#include "readerwriterqueue.h"

#pragma comment(lib, "Bcrypt.lib")
//#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Secur32.lib")
#pragma comment(lib, "ws2_32.lib")

#define DEBUG_IMSHOW(name, image) \
if (DEBUG) {                      \
    cv::imshow(#name, image);     \
}

const bool DEBUG = false;

namespace lane {
    const cv::Scalar HSV_LOWER_BOUND{ 0, 0, 132 }, HSV_UPPER_BOUND{ 255, 255, 255 };
    
    const double CANNY_THRESHOLD1 = 50, CANNY_THRESHOLD2 = 150;

    const double HOUGH_RHO = 1, HOUGH_THETA = CV_PI / 180, HOUGH_MIN_LINE_LENGTH = 20, HOUGH_MAX_LINE_GAP = 40;
    const int HOUGH_THRESHOLD = 60;

    const int LINE_THICKNESS = 2;

    const cv::Point2f PERSPECTIVE_SRC[4]{ cv::Point2f{ 330, 719 }, cv::Point2f{ 1018, 719 }, cv::Point2f{ 477, 456 }, cv::Point2f{ 748, 456 } },
        PERSPECTIVE_DST[4]{ cv::Point2f{ 330, 719 }, cv::Point2f{ 1018, 719 }, cv::Point2f{ 330, 0 }, cv::Point2f{ 1018, 0 } };
    const cv::Mat PERSPECTIVE_MT = cv::getPerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST),
        PERSPECTIVE_MR = cv::getPerspectiveTransform(PERSPECTIVE_DST, PERSPECTIVE_SRC);

    std::pair<double, double> get_line_slope_and_intercept(cv::Vec4i line) {
        cv::Point start{ line[0], line[1] };
        cv::Point end{ line[2], line[3] };
        const double slope = ((double)(line[1]) - line[3]) / ((double)(line[0]) - line[2]);
        const double intercept = line[1] - slope * line[0];
        return { slope, intercept };
    }

    std::pair<cv::Point, cv::Point> toPoints(int sy, std::pair<double, double> si, double frac) {
        int sx = (int)((sy - si.second) / si.first);
        int ey = (int)(frac * sy);
        int ex = (int)((ey - si.second) / si.first);
        cv::Point start{ sx, sy };
        cv::Point end{ ex, ey };
        return { start, end };
    }

    inline void plot_si_line(cv::Mat img, std::pair<double, double> si, const cv::Scalar& color, int thickness = 1, int lineType = cv::LINE_8, int shift = 0) {
        auto se = toPoints(img.rows - 1, si, 0.4);
        cv::line(img, se.first, se.second, color, thickness, lineType, shift);
    }

    inline void plot_road(cv::Mat img, std::pair<double, double> left_si, std::pair<double, double> right_si, const cv::Scalar & color, int lineType = cv::LINE_8, int shift = 0) {
        auto lse = toPoints(img.rows - 1, left_si, 0.5);
        auto rse = toPoints(img.rows - 1, right_si, 0.5);
        std::array<cv::Point, 4> region{ lse.first, lse.second, rse.second, rse.first };
        cv::fillConvexPoly(img, region, color, lineType, shift);
    }

    cv::Mat frame_pipeline(cv::Mat frame) {
        cv::Mat binary, pti, vis; // binarized, perspective-transformed, final
        std::vector<cv::Vec4i> lines, lines2; // hough lines in pti, hough lines in original
        
        // s means slope, i means intercept, used in categorizing hough lines
        double left_s_sum = 0, left_i_sum = 0, right_s_sum = 0, right_i_sum = 0;
        int left_count = 0, right_count = 0;

        // HSV threshold
        {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND, binary);
            DEBUG_IMSHOW(HSV_threshold, binary)
        }

        // Gaussian, UNUSED
        {
            // cv::GaussianBlur(binary, binary, GAUSSIAN_SIZE, 0);
            // DEBUG_IMSHOW(GaussianBlur, binary)
        }

        // ROI
        {
            cv::Mat mask(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
            // trapezoid
            std::array<cv::Point, 4> region{ cv::Point{275, 719}, cv::Point{1050, 719}, cv::Point{675, 719 - 375}, cv::Point{500, 719 - 375} };
            cv::Scalar color{ 255 };
            cv::fillConvexPoly(mask, region, color);
            binary &= mask;
            DEBUG_IMSHOW(ROI, binary)
        }

        // Canny
        {
            cv::Canny(binary, binary, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
            DEBUG_IMSHOW(Canny, binary)
        }

        // Perspective transform
        {
            cv::warpPerspective(binary, pti, PERSPECTIVE_MT, cv::Size{ frame.cols, frame.rows });
            DEBUG_IMSHOW(PerspectiveTransform, pti)
        }

        // Hough
        {
            cv::HoughLinesP(pti, lines, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
        }

        // Transform back hough points
        {
            std::vector<cv::Point2f> points, points2;
            points.reserve(lines.size() * 2);
            points2.reserve(lines.size() * 2);
            for (auto&& line : lines) {
                cv::Point2f start{ (float)line[0], (float)line[1] };
                cv::Point2f end{ (float)line[2], (float)line[3] };
                points.push_back(start);
                points.push_back(end);
            }

            cv::perspectiveTransform(points, points2, PERSPECTIVE_MR);

            for (size_t i = 0; i < lines.size(); i++)
            {
                cv::Point2f start = points2[2 * i];
                cv::Point2f end = points2[2 * i + 1];

                lines2.push_back({ (int)start.x, (int)start.y, (int)end.x, (int)end.y });
            }

            if (DEBUG) {
                // visualize hough
                cv::Mat lineGraph(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0));
                for (auto&& line : lines2) {
                    cv::Point start{ line[0], line[1] };
                    cv::Point end{ line[2], line[3] };
                    cv::Scalar color{ 0, 0, 255 };
                    cv::line(lineGraph, start, end, color, LINE_THICKNESS);
                }
                cv::imshow("HoughVisualization", lineGraph);
            }
        }

        // Categorize left & right line
        {
            for (auto&& line : lines2) {
                auto si = get_line_slope_and_intercept(line);
                if (si.first < 0) {
                    // left
                    left_s_sum += si.first;
                    left_i_sum += si.second;
                    left_count += 1;
                }
                else {
                    right_s_sum += si.first;
                    right_i_sum += si.second;
                    right_count += 1;
                }
            }
        }
        
        // Copy raw picture
        {
            frame.copyTo(vis);
        }

        // Draw left & right line
        {
            cv::Scalar color{ 0, 0, 255 };
            if (left_count > 0) {
                std::pair<double, double> left_si_mean{ left_s_sum / left_count, left_i_sum / left_count };
                plot_si_line(vis, left_si_mean, color, LINE_THICKNESS * 3);
            }
            else {
                std::cerr << "Fail to detect left lane!\n";
            }
            if (right_count > 0) {
                std::pair<double, double> right_si_mean{ right_s_sum / right_count, right_i_sum / right_count };
                plot_si_line(vis, right_si_mean, color, LINE_THICKNESS * 3);
            }
            else {
                std::cerr << "Fail to detect right lane!\n";
            }
        }
            
        // Draw area between line
        {
            cv::Scalar color{ 255, 255, 0 };
            if (left_count > 0 && right_count > 0) {
                std::pair<double, double> left_si_mean{ left_s_sum / left_count, left_i_sum / left_count };
                std::pair<double, double> right_si_mean{ right_s_sum / right_count, right_i_sum / right_count };
                plot_road(vis, left_si_mean, right_si_mean, color);
            }
        }
            
        return vis;
    }

    const char* PROLOG =
        "-------------------------------------------\n"
        "| Traffic line detector                   |\n"
        "| Author: Jiang Yuxuan                    |\n"
        "-------------------------------------------\n";

    void process(std::string src, std::string dst) {
        std::cout << PROLOG << std::endl;

        cv::VideoCapture vc;
        vc.open(src, cv::CAP_FFMPEG);
        if (!vc.isOpened()) {
            std::cerr << "ERROR! Unable to read source: " << src << '\n';
            return;
        }
        
        double fps = vc.get(cv::CAP_PROP_FPS);
        int width = (int)vc.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)vc.get(cv::CAP_PROP_FRAME_HEIGHT);
        int frames = (int)vc.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "Video metadata:\n" << "  FPS: " << fps << "\n  Total frames: " << frames << "\n  Size: " << width << " * " << height << "\n" << std::endl;

        cv::VideoWriter vw;
        vw.open(dst, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size{ width, height }, true);
        if (!vw.isOpened()) {
            std::cerr << "ERROR! Unable to write to destination: " << dst << '\n';
            return;
        }
        
        std::cout << "Start decoding and encoding, please wait\n" << std::endl;
        boost::progress_display progress(frames);
        cv::Mat frame;

        for (;;)
        {
            vc >> frame;
            ++progress;
            if (frame.empty()) {
                break;
            }
            frame = frame_pipeline(frame);
            vw << frame;
        }
    }

    namespace parallel {
        class VideoReaderStream
        {
        public:
            //using PAIR = std::pair<int, cv::Mat>;
            VideoReaderStream() = delete;
            VideoReaderStream(std::unique_ptr<cv::VideoCapture> vc): vc(std::move(vc)), queue(128), reader([&]() {
                while (true) {
                    *(this->vc) >> tmp;
                    this->queue.enqueue(tmp);
                    if (tmp.empty()) {
                        this->vc.reset();
                        return;
                    }
                }
            })
            {
                
            }

            inline double get(int propId) { return vc->get(propId); }
            inline void join() { reader.join(); }

            ~VideoReaderStream()
            {
            }

            VideoReaderStream& operator >> (cv::Mat& image) {
                queue.wait_dequeue(image);
                return *this;
            }
        private:
            cv::Mat tmp;
            std::thread reader;
            std::unique_ptr<cv::VideoCapture> vc;
            moodycamel::BlockingReaderWriterQueue<cv::Mat> queue;
        };

        class VideoWriterStream
        {
        public:
            //using PAIR = std::pair<int, cv::Mat>;
            VideoWriterStream() = delete;
            VideoWriterStream(std::unique_ptr<cv::VideoWriter> vw) : vw(std::move(vw)), queue(128), writer([&]() {
                while (true) {
                    this->queue.wait_dequeue(tmp);
                    if (tmp.empty()) {
                        this->vw.reset();
                        return;
                    }
                    *(this->vw) << tmp;
                }
                })
            {
            }
            inline void join() { writer.join(); }

            ~VideoWriterStream()
            {
            }

            VideoWriterStream& operator << (cv::Mat& image) {
                queue.enqueue(image);
                return *this;
            }
        private:
            cv::Mat tmp;
            std::thread writer;
            std::unique_ptr<cv::VideoWriter> vw;
            moodycamel::BlockingReaderWriterQueue<cv::Mat> queue;
        };

        void process(std::string src, std::string dst) {
            auto vc = std::make_unique<cv::VideoCapture>();
            vc->open(src, cv::CAP_FFMPEG);
            if (!vc->isOpened()) {
                std::cerr << "ERROR! Unable to read source\n";
                return;
            }
            VideoReaderStream vrs{ std::move(vc) };

            double fps = vrs.get(cv::CAP_PROP_FPS);
            int width = (int)vrs.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = (int)vrs.get(cv::CAP_PROP_FRAME_HEIGHT);
            auto vw = std::make_unique<cv::VideoWriter>();
            vw->open(dst, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size{ width, height }, true);
            VideoWriterStream vws{ std::move(vw) };

            cv::Mat frame;

            for (;;)
            {
                vrs >> frame;
                vws << frame;
                if (frame.empty()) {
                    vrs.join();
                    vws.join();
                    break;
                }

                //frame = frame_pipeline(frame);
                //
                //if (DEBUG) {
                //    cv::imshow("Live", frame);
                //    if (cv::waitKey(0) >= 0)
                //        break;
                //}

            }
        }

    }
}

int main()
{
    lane::process("./input.mp4", "./output.mp4");
    return 0;
}

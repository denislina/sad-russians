#pragma once

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <mutex>
#include <algorithm>

class FaceHandler {
public:
    FaceHandler(const cv::Mat& image) : image_(image) { CreateKernel_();}
    void ProcessFace(const cv::Rect&, torch::jit::script::Module*);
private:
    cv::Mat image_;
    std::mutex mutex_;
    cv::Mat kernel_;

    void CreateKernel_();
    cv::Mat ApplyModel_(const cv::Mat&, torch::jit::script::Module*);
    void InsertFace_(const cv::Rect&, cv::Mat);

    static constexpr int rows_ = 256;
    static constexpr int cols_ = 256;
};
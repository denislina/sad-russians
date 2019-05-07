#include "FaceHandler.h"
#include "utils.h"

void FaceHandler::ProcessFace(const cv::Rect& face, torch::jit::script::Module* model) {
    auto face_img = cv::Mat(rows_, cols_, CV_32F, 0.);
    auto face_img_full = image_(face);
    resize(face_img_full, face_img, cv::Size(rows_, cols_));
    auto face_img_transformed = ApplyModel_(face_img, model);
    resize(face_img_transformed, face_img_full, face.size());
    InsertFace_(face, face_img_full);
}

void FaceHandler::InsertFace_(const cv::Rect& face, cv::Mat face_img) {
    for (size_t row = 0; row < face_img.rows; ++row) {
        for (size_t col = 0; col  < face_img.cols; ++col) {
            image_.at<cv::Vec3b>(row + face.tl().y, col + face.tl().x) = face_img.at<cv::Vec3b>(row, col);
        }
    }
}

void FaceHandler::CreateKernel_() {
    auto gauss_x = cv::getGaussianKernel(rows_, 500, CV_64F);
    auto gauss_y = cv::getGaussianKernel(cols_, 500, CV_64F).t();
    kernel_ = cv::Mat(rows_, cols_, CV_32F, 0.);
    cv::pow(gauss_x * gauss_y, 30, kernel_);
    double max_val = *std::max_element(kernel_.begin<double>(), kernel_.end<double>());
    kernel_ = kernel_ / max_val;
}

cv::Mat FaceHandler::ApplyModel_(const cv::Mat& face_img, torch::jit::script::Module* model) {
    auto input = torch::from_blob(face_img.data, {1, rows_, cols_, 3}, at::kByte).to(at::kFloat).permute({0, 3, 1, 2});
    input /= 255;
    input -= 0.5;
    input *= 2;

    mutex_.lock();
    auto output = model->forward({input}).toTensor();
    mutex_.unlock();

    input /= 2;
    input += 0.5;

    //sub mean and divide by std
    output[0][0] -= torch::mean(output[0][0]);
    output[0][1] -= torch::mean(output[0][1]);
    output[0][2] -= torch::mean(output[0][2]);
    output[0][0] /= torch::std(output[0][0]);
    output[0][1] /= torch::std(output[0][1]);
    output[0][2] /= torch::std(output[0][2]);


    //add input mean and mul by std
    output[0][0] *= torch::std(input[0][0]);
    output[0][1] *= torch::std(input[0][1]);
    output[0][2] *= torch::std(input[0][2]);
    output[0][0] += torch::mean(input[0][0]);
    output[0][1] += torch::mean(input[0][1]);
    output[0][2] += torch::mean(input[0][2]);
    output *= 255;
    output += 0.5;
    input *= 255;

    HandleEdges(kernel_, input, &output);
    TensortoMat(output, face_img);
    return face_img;
}
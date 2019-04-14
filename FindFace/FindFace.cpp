
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>

std::vector<std::string> HandleArguments(int, char**);

std::vector<cv::Rect> FindFrontFaces(const cv::Mat);

std::string GetShortName(const std::string&);

bool ProcessFile(const std::string&);

void TensortoMat(const at::Tensor&, cv::Mat);

void InsertFace(const cv::Rect&, cv::Mat, cv::Mat);

int clamp(float* data_ptr);

int main(int argc, char** argv) {
    auto file_names = HandleArguments(argc, argv);

    if (file_names.empty()) {
      return 0;
    }

    try {
        for (auto&& file_name : file_names) {
            bool success = ProcessFile(file_name);

            if (! success) {
                throw std::runtime_error("Not succes");
            }
        }
    } catch (std::runtime_error& exception) {
        std::cerr << "Something go wrong. " << exception.what() << '\n';  
    }
    return 0;
}


std::string GetShortName(const std::string& filename) {
  size_t split_index = filename.find_last_of('/');
  size_t point_index = filename.find_last_of('.');
  return filename.substr(split_index + 1, point_index - split_index - 1);
}

std::vector<std::string> HandleArguments(int argc, char** argv) {

    /*
    if (argc != 2 && argc != 3) {
        std::cerr << "You have to specify a photo file or dir\n";
        return {};
    }
    */
  
    if (argc == 1) {
        std::cerr << "You have to specify a photo files";
        return {};
    }

    std::vector<std::string> file_names;
    file_names.reserve(argc - 1);
    for (size_t index = 1; index < argc; ++index) {
      file_names.emplace_back(argv[index]);
    }

    return file_names;
}

std::vector<cv::Rect> FindFrontFaces(cv::Mat image) {
    //path to cascade weights
    static const std::string cascade_weights =
        "../haarcascade_frontalface_alt.xml";

    cv::CascadeClassifier face_detector;
    if (!face_detector.load(cascade_weights)) {
        throw std::runtime_error("Can't load cascade");
    }

    //translate to gray
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    //normalize
    cv::equalizeHist(image, image);

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(image, faces, 1.6, 5, 0, cv::Size(10, 10));
    return faces;
}

bool ProcessFile(const std::string& file_name) {
    auto short_name = GetShortName(file_name);

    cv::Mat image = cv::imread(file_name);
    if (image.data == NULL) {
        throw std::runtime_error("Can't open file " + file_name + " with image");
    }

    auto faces = FindFrontFaces(image.clone());
    auto smile_to_sad = torch::jit::load("../../models/netG_B2A_model.pth");

    for (auto&& face : faces) {
        cv::Mat face_img = image(face);
        resize(image(face), face_img, cv::Size(256, 256));

        auto input = torch::from_blob(face_img.data, {1, face_img.rows, face_img.cols, 3}, at::kByte).to(at::kFloat);
        input = input.permute({0, 3, 1, 2});
        input /= 255;
        auto old_input = input.clone();

        input -= 0.5;
        input *= 2;

        auto output = smile_to_sad->forward({input}).toTensor();
        output[0][0] -= torch::mean(output[0][0]);
        output[0][1] -= torch::mean(output[0][1]);
        output[0][2] -= torch::mean(output[0][2]);
        output[0][0] /= torch::std(output[0][0]);
        output[0][1] /= torch::std(output[0][1]);
        output[0][2] /= torch::std(output[0][2]);
        output[0][0] *= torch::std(old_input[0][0]);
        output[0][1] *= torch::std(old_input[0][1]);
        output[0][2] *= torch::std(old_input[0][2]);
        output[0][0] += torch::mean(old_input[0][0]);
        output[0][1] += torch::mean(old_input[0][1]);
        output[0][2] += torch::mean(old_input[0][2]);


        output *= 255;
        output += 0.5;

        TensortoMat(output, face_img);
        resize(face_img, face_img, face.size());
        InsertFace(face, face_img, image);
    }

    cv::imwrite("../../out/" + short_name + "_transformed.jpeg", image);
    return true;
}

void TensortoMat(const at::Tensor& tensor, cv::Mat mat) {
    for (size_t row = 0; row < tensor.size(2); ++row) {
        for (size_t col = 0; col < tensor.size(3); ++col) {
            mat.at<cv::Vec3b>(row, col) = cv::Vec3b(clamp(tensor[0][0][row][col].data<float>()),
                                                    clamp(tensor[0][1][row][col].data<float>()),
                                                    clamp(tensor[0][2][row][col].data<float>()));
        }
    }
}

void InsertFace(const cv::Rect& face, cv::Mat face_img, cv::Mat image) {
    for (size_t row = 0; row < face_img.rows; ++row) {
        for (size_t col = 0; col < face_img.cols; ++col) {
            image.at<cv::Vec3b>(row + face.tl().y, col + face.tl().x) = face_img.at<cv::Vec3b>(row, col);
        }
    }
}


int clamp(float* data_ptr) {
    float data = *data_ptr;
    if (data < 0) {
        return 0;
    }
    if (data > 255) {
        return 255;
    }
    return static_cast<int>(round(data));
}
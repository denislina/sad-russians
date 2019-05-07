#include "utils.h"

std::string HandleArguments(int argc, char** argv, std::vector<std::string>* file_names_ptr) {
    if (argc == 1) {
        std::cerr << "You must transfer the name of at least one image file" << std::endl;
        return "";
    }
    std::string mode = "happy";
    size_t start_index = 1;
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--happy") == 0) {
        mode = "happy";
        start_index = 2;
    } else if (strcmp(argv[1], "-s") == 0 || strcmp(argv[1], "--sad") == 0) {
        mode = "sad";
        start_index = 2;
    }

    if (start_index == argc) {
        std::cerr << "You must transfer the name of at least one image file" << std::endl;
        return "";
    }

    file_names_ptr->reserve(argc - start_index);
    for (size_t index = start_index; index < argc; ++index) {
      file_names_ptr->emplace_back(argv[index]);
    }

    return mode;
}

std::string GetShortName(const std::string& filename) {
  size_t split_index = filename.find_last_of('/');
  size_t point_index = filename.find_last_of('.');
  return filename.substr(split_index + 1, point_index - split_index - 1);
}

std::string GetPath(const std::string& filename) {
    size_t split_index = filename.find_last_of('/');
    return filename.substr(0, split_index + 1);
}

int clamp(float data) {
    if (data < 0) {
        return 0;
    }
    if (data > 255) {
        return 255;
    }
    return static_cast<int>(round(data));
}

void TensortoMat(const at::Tensor& tensor, cv::Mat mat) {
    for (size_t row = 0; row < tensor.size(2); ++row) {
        for (size_t col = 0; col < tensor.size(3); ++col) {
            mat.at<cv::Vec3b>(row, col) = cv::Vec3b(clamp(*tensor[0][0][row][col].data<float>()),
                                                    clamp(*tensor[0][1][row][col].data<float>()),
                                                    clamp(*tensor[0][2][row][col].data<float>()));
        }
    }
}

void HandleEdges(const cv::Mat& mat, const torch::Tensor& input, torch::Tensor* output_ptr) {
    constexpr float max_limit = 0.6;
    constexpr float min_limit = 0.4;
    int rows = mat.rows;
    int cols = mat.cols;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float mask_val = mat.at<double>(row, col);
            if (mask_val >= max_limit) {
                mask_val = 1;
            } else if (mask_val >= min_limit) {
                mask_val -= min_limit;
                mask_val /= (max_limit - min_limit);
            } else {
                mask_val = 0;
            }
            for (int colour = 0; colour < 3; ++colour) {
                float input_val = *input[0][colour][row][col].data<float>();
                *(*output_ptr)[0][colour][row][col].data<float>() *= mask_val;
                *(*output_ptr)[0][colour][row][col].data<float>() += (1 - mask_val) * input_val;
            }
        }
    }
}

std::vector<cv::Rect> FindFrontFaces(cv::Mat image) {
    //path to cascade weights
    static const std::string cascade_weights = "../haarcascade_frontalface_default.xml";

    cv::CascadeClassifier face_detector;
    if (!face_detector.load(cascade_weights)) {
        throw std::runtime_error("Can't load cascade");
    }

    //translate to gray
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    //normalize
    cv::equalizeHist(image, image);

    //find faces
    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(image, faces, 1.3, 5);
    return faces;
}
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <experimental/filesystem>
#include <cstring>

namespace fs = std::experimental::filesystem;

std::vector<std::string> HandleArguments(int, char**);

//std::vector<cv::Rect> FindFrontFaces(const cv::Mat);

int main(int argc, char** argv) {
    auto file_names = HandleArguments(argc, argv);

    if (file_names.empty()) {
      return 0;
    }

    auto x = cv::String(argv[1]);

    cv::Mat image = cv::imread(x);
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    return 0;
    try {
        for (auto&& file_name : file_names) {
        //cv::Mat image = cv::imread("abcd");

        if (image.data == NULL) {
          throw std::runtime_error("Can't open file " + file_name + " with image");
        }
        /*
        cv::Mat image_gray;
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

        cv::CascadeClassifier face_detector;
        if (!face_detector.load( "/Users/r.britkov/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")) {
          throw std::runtime_error("Can't load cascade");
        }
        std::vector<cv::Rect> faces;
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(image_gray, image_gray);
        face_detector.detectMultiScale(image_gray, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20));

        for (int i = 0; i < faces.size(); i++) {
            cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            cv::ellipse(image, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(100, 0, 255), 4, 8, 0);
        }

        cv::imwrite("../tests/test_out.jpeg", image);
        */
        }
    } catch (std::runtime_error& exception) {
        std::cerr << "Something go wrong. " << exception.what() << '\n';  
    }
    return 0;
}


std::vector<std::string> HandleArguments(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "You have to specify a photo file or dir\n";
        return {};
    }

    std::vector<std::string> file_names;
    if (argc == 2) {
        file_names.emplace_back(argv[1]);
    } else {
        if (strcmp(argv[1], "-f") == 0) {
            file_names.emplace_back(argv[2]);
        } else if (strcmp(argv[1], "-d") == 0) {
            size_t length_of_path = strlen(argv[2]);
            for (auto&& file : fs::directory_iterator(argv[2])) {
                std::string path = file.path();

                //don't add hidden files
                if (path[length_of_path + 1] != '.') { 
                    file_names.emplace_back(std::move(path));
                }
            }
        } else {
          std::cerr << "Unknown option. You use only -f for file or -d for dir\n";
          return {};
        }
    }

    return file_names;
}

/*
std::vector<cv::Rect> FindFrontFaces(cv::Mat image) {
    static constexpr char* cascade_weights =
        "/Users/r.britkov/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier face_detector;
    if (!face_detector.load(cascade_weights)) {
        throw std::runtime_error("Can't load cascade");
    }

    //translate to gray
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    //normalize
    cv::equalizeHist(image, image);

    std::vector<cv::Rect> faces;
    //face_detector.detectMultiScale(image, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20));
    return faces;
}
*/
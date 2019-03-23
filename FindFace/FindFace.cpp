#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cstring>

std::vector<std::string> HandleArguments(int, char**);

std::vector<cv::Rect> FindFrontFaces(const cv::Mat);

std::string GetShortName(const std::string&);

bool ProcessFile(const std::string&);

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
    face_detector.detectMultiScale(image, faces, 1.1, 5, 0, cv::Size(10, 10));
    return faces;
}

bool ProcessFile(const std::string& file_name) {
    auto short_name = GetShortName(file_name);

    cv::Mat image = cv::imread(file_name);
    if (image.data == NULL) {
        throw std::runtime_error("Can't open file " + file_name + " with image");
    }

    auto faces = FindFrontFaces(image);

    char number = 'a';
    for (auto&& face : faces) {
        cv::Mat face_img = image(face);
        std::string name = "../out/" + short_name + "_";
        name += number;
        name += ".jpeg";
        number += 1;
        cv::imwrite(name, face_img);
    }

    return true;
}
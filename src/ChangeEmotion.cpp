#include "ChangeEmotion.h"
#include "utils.h"
#include "FaceHandler.h"

int main(int argc, char** argv) {
    std::vector<std::string> file_names;
    auto mode = HandleArguments(argc, argv, &file_names);
    if (file_names.empty()) {
      return 0;
    }

    std::string path =  argv[0];
    path = GetPath(path);
    static const std::string cascade_weights = path + "../haarcascade_frontalface_default.xml";
    cv::CascadeClassifier face_detector;

    if (!face_detector.load(cascade_weights)) {
        throw std::runtime_error("Can't load cascade");
    }

    std::shared_ptr<torch::jit::script::Module> model;
    auto clf = torch::jit::load(path + "../models/emotion_recognition.pt");
    if (mode == "happy") {
        model = torch::jit::load(path + "../models/netG_sad2smile-final.pt");
    } else {
        model = torch::jit::load(path + "../models/netG_smile2sad-final.pt");
    }

    for (auto&& file_name : file_names) {
        try {
            ProcessFile(file_name, model.get(), clf.get(), face_detector, mode);
        } catch (std::runtime_error& exception) {
            std::cerr << "Something go wrong. " << exception.what() << '\n';
        }
    }
    return 0;
}

void ProcessFile(const std::string& file_name, torch::jit::script::Module* model, torch::jit::script::Module* clf,
                        cv::CascadeClassifier& face_detector, const std::string& mode) {
    auto short_name = GetShortName(file_name);
    auto path = GetPath(file_name);

    auto image = cv::imread(file_name);
    if (image.data == NULL) {
        throw std::runtime_error("Can't open file " + file_name + " with image");
    }
    FaceHandler face_handler(image);
    auto faces = FindFrontFaces(image.clone(), face_detector);

    if (faces.size() == 1) {
        face_handler.ProcessFace(faces[0], model, clf, mode);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(faces.size());
        for (auto&& face : faces) {
            workers.emplace_back([&face_handler, face, model, clf, mode]() {
                face_handler.ProcessFace(face, model, clf, mode);
            });
        }

        for (auto&& worker : workers) {
            worker.join();
        }
    }

    cv::imwrite(path + short_name + "_" + mode + ".jpeg", image);
}


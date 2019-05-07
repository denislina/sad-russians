#include "ChangeEmotion.h"
#include "utils.h"
#include "FaceHandler.h"

int main(int argc, char** argv) {
    std::vector<std::string> file_names;
    auto mode = HandleArguments(argc, argv, &file_names);
    if (file_names.empty()) {
      return 0;
    }
    std::shared_ptr<torch::jit::script::Module> model;
    if (mode == "happy") {
        model = torch::jit::load("../models/netG_sad2smile-final.pt");
    } else {
        model = torch::jit::load("../models/netG_smile2sad-final.pt");
    }

    for (auto&& file_name : file_names) {
        try {
            ProcessFile(file_name, model.get(), mode);
        } catch (std::runtime_error& exception) {
            std::cerr << "Something go wrong. " << exception.what() << '\n';
        }
    }
    return 0;
}

void ProcessFile(const std::string& file_name, torch::jit::script::Module* model, const std::string& mode) {
    auto short_name = GetShortName(file_name);
    auto path = GetPath(file_name);

    auto image = cv::imread(file_name);
    if (image.data == NULL) {
        throw std::runtime_error("Can't open file " + file_name + " with image");
    }
    FaceHandler face_handler(image);
    auto faces = FindFrontFaces(image.clone());

    if (faces.size() == 1) {
        face_handler.ProcessFace(faces[0], model);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(faces.size());
        for (auto&& face : faces) {
            workers.emplace_back([&face_handler, face, model]() {
                face_handler.ProcessFace(face, model);
            });
        }

        for (auto&& worker : workers) {
            worker.join();
        }
    }

    cv::imwrite(path + short_name + "_" + mode + ".jpeg", image);
}


#pragma once

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

std::string HandleArguments(int, char**, std::vector<std::string>*);

std::string GetShortName(const std::string&);

std::string GetPath(const std::string&);

int clamp(float);

void TensortoMat(const at::Tensor&, cv::Mat);

void HandleEdges(const cv::Mat&, const torch::Tensor&, torch::Tensor*);

std::vector<cv::Rect> FindFrontFaces(const cv::Mat, cv::CascadeClassifier&);
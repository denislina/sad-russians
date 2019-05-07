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
#include <thread>
#include <cmath>

void ProcessFile(const std::string&, torch::jit::script::Module*, const std::string&);


cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(model_executor model_executor.cpp)
target_link_libraries(model_executor "${TORCH_LIBRARIES}")
set_property(TARGET model_executor PROPERTY CXX_STANDARD 11)

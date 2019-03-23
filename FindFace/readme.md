I Install opencv on macos by this tutorial
https://www.learnopencv.com/install-opencv-4-on-macos/

How to build
rm -r ./build
mkdir ./build && cd ./build
cmake ..
cmake --build . --config Release

How to run from build directory
./FindFace ../tests/test_1.jpeg
After name of program specify photo files
Faces appear in dir out

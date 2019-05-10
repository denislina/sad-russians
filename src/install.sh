# Создаем чистую папку libs
rm -r libs
mkdir libs
cd libs

# Устанавливаем OpenCV
wget -O opencv.tar.gz https://sourceforge.net/projects/opencvlibrary/files/4.1.0/OpenCV%204.1.0.tar.gz
tar -xvf opencv.tar.gz
rm opencv.tar.gz
mv opencv-opencv-0399435/ opencv/
cd opencv
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make -j8
sudo make install
cd ../..

# Устанавливаем LibTorch
wget -O libtorch.zip https://www.dropbox.com/s/9qky4jlu7tcjtof/libtorch.zip
unzip libtorch.zip
rm -r __MACOSX
rm libtorch.zip
cd ..

# Собираем код проекта в исполняемый файл
mkdir build
cd build
cmake ..
make


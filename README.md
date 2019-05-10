# Проект "Россия для грустных"

The sad, The happy, The ugly, The creepy.

Утилита для изменения эмоций всех лиц, найденных на фотографии.
С ней вы можете превратить грустные лица в весёлые и наоборот.

Возможности:
- загрузка одного или нескольких фото
- выделение нескольких лиц одновременно на одной фотографии
- классификатор весёлости выдает значение от 0 (грусть) до 1 (радость).
- изменение лица на (грустное/счастливое/иногда криповое)
- загрузка преобразованной фотографии

<img src="https://pp.userapi.com/c850136/v850136493/13f0cc/bHQNApSwdrg.jpg" width="700" height="400">

<img src="https://pp.userapi.com/c844417/v844417493/1fde54/JUI0kEWwXIU.jpg" width="700" height="400">

## Начало работы

### Зависимости

- LibTorch
- OpenCV

Установочный скрипт (см. ниже) скачает и соберёт зависимости. Сборка OpenCV достаточно долгая. Ближе к концу работы скрипт попросит пароль.

### Сборка на Mac OS

- `git clone https://github.com/denislina/sad-russians.git`
- `cd sad-russians/src`
- `bash install.sh`

### Пример запуска

#### Из консоли

В папке `src/build`
```
./ChangeEmotion --happy <path/to/pic.jpg>
```
Программа положит в папку с исходной картинкой `pic.jpg` новый файл `pic_happy.jpeg`. Примеры картинок можно найти в папке examples.

#### С помощью красивого сайта

В папке `py_web`
- запустить `python loader.py`
- загрузить файл из examples (например, 8.jpg)
- ..подождать, пока отработает модель..
- наслаждаться!

### Обучение моделей

Disclaimer: Обучение моделей производилось на Python. При этом сама утилита от Python не пострадала.

- emotion_recognition.ipynb - обучение модели классификации на веселых и грустных.
- CycleGanSmile.ipynb - обучение модели CycleGan.
- CycleGanApply.ipynb - применение модели CycleGan для изменения эмоции.

CycleGan обучался на датасете [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
. Небольшая выборка представлена в datasets/img_align_celeba_50/

## Авторы

* **Алина Денисова** - *Классификация улыбки, Веб приложение* - [Gitlab](https://github.com/denislina)
* **Павел Губко** - *C++* - [Gitlab](https://github.com/gubkopaul)
* **Радомир Бритков** - *C++* - [Gitlab](https://github.com/Radi4)
* **Виктория Ходырева** - *CycleGan* - [Gitlab](https://github.com/Khodyrevavk)

## Благодарности

* [идея CycleGan](https://hardikbansal.github.io/CycleGANBlog/), [Код](https://github.com/aitorzip/PyTorch-CycleGAN)
* [модель классификации Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)

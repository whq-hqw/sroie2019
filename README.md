# SROIE 2019 Task 1 & 3

## Installation
Clone this repo under ~/Documents:
```
git clone --recurse-submodules https://github.com/loveorchids/sroie2019 ~/Documents/sroie2019
```

## Requirement
Python:  3.5.2 or higher
```
pip install -r requirements.txt
```

## Prepare Data:
1. Create path: ~/Downloads/dataset/ocr/SROIE2019_test
2. Put the images(jpg, png, etc.) you wish to try under above folder, or you can download dataset from SROIE 2019.

## Download Model for Task1:
1. Create path: ~/Downloads/dataset/ocr/_text_detection/
2. Download the model and put it under the above folder.
```
https://drive.google.com/open?id=1uZnBuLm_DLKwNukYhwbrbDxUmLa2XtPe
```

## Testing:
Run tb_test.py in terminal.
There are some parameters that you could change, please see the definition under tb_test.py
Feel free to adjust them.
```
python3 ~/Documents/sroie2019/researches/ocr/textbox/tb_test.py --square_size xxx
```
If everything is prepared, then you will see:
```
Load model form: /home/USERNAME/Downloads/dataset/ocr/_text_detection/ft_003_3_epoch_0203.pth
Above model loaded with out a problem
0 th image cost 1.07 seconds
1 th image cost 0.38 seconds
...
```

## Commercial Usage:
It is OK to use this repo for commercial usage. 
If you have your other special needs on different commercial tasks, feel free to contact us and we could prepare customization services for your products.
Please contact: hqwang@icloud.com for more information.

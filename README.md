Readme

### 1.**How to train**

For training, you can refer to the YOLOv8 training tutorial
You can use deafult.yaml or run it directly in the terminal.

### 2.**image pre-processing**

1.From [Astropedia - Moon LRO LROC WAC Global Morphology Mosaic 100m](https://astrogeology.usgs.gov/search/map/moon_lro_lroc_wac_global_morphology_mosaic_100m) downloading test image.

2.In the MoonCode folder, there is a tif4s.py file. Run the following command. You can modify the path to split the original picture by row and column and set the number of columns and rows in the text. In this article, there are 3 rows and 6 columns (the original picture has a pixel length of width: 2:1, and it is recommended that the number of columns be twice the number of rows, so that the study area after splitting is square).

```
python MoonCode/tif4s.py
```

3.Run the study region.py file in MoonCode. The purpose is to randomly select a 200*200km^2 area from the just-segmented image, or you can set another area.

```
python MoonCode/study region.py
```

4.In the spilt processing folder, there is a split.py file containing input, which outputs a few rows and columns in the terminal. The input is the 200*200km^2 tif image that was just randomly split, and the text is the output of eight rows and eight columns.

```
python spilt processing/split.py
```

5.Find the split processing/GAN/GAN.py file. First, randomly split a 200*200km^2 random area once for GAN processing and output to the folder REF_PIC. Then input the split small photo into GAN as the starting point for recognition.

### 3.**recognition**

Then it can be directly called in the main.py file.
It requires: the weight path, the path of the small image after GAN processing, and the tif map of the corresponding area in REF_PIC.

```
python main.py
```

The output is the folder t/{test_num} (where test_num is a parameter) with a csv file containing the coordinates and diameter of each identified pit, and a tif image of the identified pit with the same pixel value as the research area in REF_PIC.

### 4.**age estimation**

There are five classic chronological equations in the Age directory. By entering the csv file and directly entering the code of the chronological equation, you can directly obtain the age corresponding to each pit.

According to the proposed time: Neukum<Hartmann<Marchi<Robbins<ChangE-5, for details, please read the paper.


# PESMOD
**PESMOD** (**PE**xels **S**mall **M**oving **O**bject **D**etection) dataset consists of high resolution aerial images in which moving objects are labelled manually. The aim of this work is to provide a different and challenging dataset for moving object detection methods evaluation. Each moving object is labelled for each frame with PASCAL VOC format in a XML file. Dataset consists of 8 sequence detailed below.


|   Sequence name  | Number of frames | Number of moving objects |
|:----------------:|:----------------:|:------------------------:|
| Elliot-road      | 664              | 3416                     |
| Miksanskiy       | 729              | 189                      |
| Shuraev-trekking | 400              | 800                      |
| Welton           | 470              | 1129                     |
| Marian           | 622              | 2791                     |
| Grisha-snow      | 115              | 1150                     |
| Zaborski         | 582              | 3290                     |
| Wolfgang         | 525              | 1069                     |
| Total            |       4107       |           13834          |


# Evaluations for different motion detection methods on PESMOD

| IOU | Method | P | R | F1 |
|----------------------------------|-----------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|----------------------------|
| 0.5    | MCD                                           | 0\.3928                                                | 0\.4163                                                | 0\.2856                    |
|                                  | SCBU                                          | 0\.3248                                                | 0\.3127                                                | 0\.3072                    |
|                                  | BSDOF                                         | 0\.4890                                                | 0\.4061                                                | 0\.3898                    |
|                                  | RTBS                                          | 0\.5442                                                | **0\.4636**                                    | 0\.4538                    |
|                                  | RTBS\*                                        | **0\.6023**                                    | 0\.4315                                                | **0\.4618**        |
| 0.25  | MCD                                           | 0\.5133                                                | 0\.5266                                                | 0\.3717                    |
|                                  | SCBU                                          | 0\.4846                                                | 0\.4490                                                | 0\.4373                    |
|                                  | BSDOF                                         | 0\.7309                                                | 0\.5681                                                | 0\.5670                    |
|                                  | RTBS                                          | 0\.7958                                                | **0\.6093**                                    | 0\.6177                    |
|                                  | RTBS\*                                        | **0\.8629**                                    | 0\.5697                                                | **0\.6240**        |

[MCD](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2013/W03/html/Yi_Detection_of_Moving_2013_CVPR_paper.html)\
[SCBU](https://www.sciencedirect.com/science/article/pii/S0167865517300260)\
[BSDOF](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-30/issue-6/063027/Real-time-motion-detection-with-candidate-masks-and-region-growing/10.1117/1.JEI.30.6.063027.short)\
RTBS: under review

# Download

Click [here](https://drive.google.com/file/d/153fLcf4F33G3oKWYUkggBWJRP5LVHV60/view?usp=sharing) to download the dataset

# Citing PESMOD Dataset
If you find this dataset or method (proposed in the paper) useful in your work, please cite the paper:

Preprint paper on [arxiv](https://arxiv.org/abs/2103.11460) 

# Contributions
If you find any mistakes in the labels, you can report it in the issues section.

## Script to view dataset, build and run performance code to evaluate your own method with foreground mask 

To view dataset after downloading: 

```
python view-dataset.py --path "/home/ibrahim/PESMOD/Pexels-Welton/"
```

Build performance code with following commands: 
```
cd performance
mkdir build
cmake ..
make .
```
Run with (-d for dataset main folder, -m for masks main folder, -f for sequence name, -o if you apply morphological opening):
```
./performance -d "/home/ibrahim/PESMOD/" -m "/home/ibrahim/SCBU-PESMOD-results/" -f "Pexels-Marian"
```

# Dataset sample frames
![Example frames from each sequence in the dataset](images/dataset.png)


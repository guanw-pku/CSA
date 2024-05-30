# CSA
## Dataset
### Data preperation
We use [PLA](https://github.com/zjucsq/PLA) as baseline and we aims to refine its detection results.
Please obtain its detection results as described in their repo and put it into `CSA/data/action-genome` folder.

Besides, you can download our preprocessed detection results, pretrained checkpoint and other necessary files in this [link](https://drive.google.com/drive/folders/1IDFe4GOd321h4FogG1MnzTAYl1fTz488?usp=sharing), and then put them in corresponding location.

The directories of CSA should look like:
```
|-- data
    |-- action-genome
        |-- frames        # sampled frames
        |-- videos        # original videos
        |-- AG_detection_results # detection results as PLA described
        |-- annotations   # downloaded gt annotations
        |-- AG_detection_results_refine # downloaded preprocessed refine detection results
|-- refine
    |-- output # downloaded checkpoint
    |-- ...
|-- PLA
    |-- model # downloaded checkpoint
    |-- ...
```

## Evaluation
+ For refine model
```
cd ~/CSA/refine
python scripts/evaluate.py # evaluate the performance of object detection
```
|AP@1|AP@10|AR@1|AR@10|weight|
|:----:|:----:|:----:|:----:|:----:|
| 13.8 | 14.2 | 35.2 | 40.6 |[link](https://drive.google.com/file/d/1KIBicIMgJ2GvJ4L-T_2n5wnSnxTTe792/view?usp=sharing)|

+ For preprocess (optional)

You can obtain the detection results with trained model.
The split to be inferred should be given in `preprocess.py`, which is in ['test', 'train', 'total'].
These three splits should all be processed.
```
cd ~/CSA/refine
python scripts/preprocess.py
```

+ For PLA model
```
cd ~/CSA/PLA
python test.py --cfg configs/oneframe.yml # for image SGG model
python test.py --cfg configs/final.yml # for video SGG model
```
| Model |W/R@10|W/R@20|W/R@50|N/R@10|N/R@20|N/R@50|weight|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Image | 16.52 | 21.42 | 26.68 | 17.57 | 23.35 | 32.27 |[link](https://drive.google.com/file/d/19asLLW8NLgjuZUuXPsnFBmiSN1mXLLV7/view?usp=sharing)|
| Video | 16.83 | 21.72 | 26.96 | 17.95 | 23.98 | 32.98 |[link](https://drive.google.com/file/d/1DRm2W3cshzfQfoKdhUnD9JEn5wnpTFzC/view?usp=sharing)|

## Train
+ For refine model
```
cd ~/CSA/refine
python scripts/train.py # evaluate object detection results
```
+ For PLA model
```
cd ~/CSA/PLA
python train.py --cfg configs/oneframe.yml # for image SGG model
python train.py --cfg configs/final.yml # for video SGG model
```

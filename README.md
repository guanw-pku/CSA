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

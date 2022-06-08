# Prioritizing Stable Images Polyp Semi-supervised segmentation 
Original code for paper ***"Improve polyp semi-supervised segmentation with prioritizing the reliability of unlabeled images"***

# How to use this source code 

## Requirement
* **Python 3.7**
* **CUDA 11.1**
* **Wandb** account for visualization. Register at [https://wandb.ai/](https://wandb.ai/)
## Installation 

```beam.assembly
pip install -r requirements.txt 
```

## Data Prepare 

Dataset training including Kvasir-SEG, CVC-ColonDB, EndoScene, ETIS-Larib Polyp DB and CVC-Clinic DB from PraNet: Parallel Reverse Attention Network for Polyp Segmentation

* Download training dataset and move it into your `./data` folder which can be found in this [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view)
* Download testing dataset and move it into your `./data` folder which can be found in this [Google Drive](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)

## Training 
After download all data and setting hyper params. We run this command to train models
```beam.assembly
python train.py
```
## Testing 

* Download pretrained model from this [Google Drive](https://drive.google.com/drive/folders/15ekYgeLrqFHKiYQxNYe29z1KuINGh8ya?usp=sharing)
and put all the models downloaded to the `/pretrained` folder.
* Change the relative path of model in ``test.py`` file.
* Run test script 

```beam.assembly
python test.py
```

## Visualization 
for test samples visualization, we use wandb website. You can use online service at [https://wandb.ai/](https://wandb.ai/) 

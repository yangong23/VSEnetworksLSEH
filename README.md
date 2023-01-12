# VSEnetworksLSEH
This is the PyTorch code for using LSEH of VSE networks, which is described in the paper " Improving Visual-Semantic Embeddings by Learning Semantically-Enhanced Hard Negatives for Cross-modal Information Retrieval".

## Requirements and installation
We recommend the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/) 

* [Pandas](https://pandas.pydata.org/)

* [scikit-learn](https://scikit-learn.org/stable/)

* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger) 

* [pycocotools](https://github.com/cocodataset/cocoapi) 

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 
We used [anaconda](https://www.anaconda.com/) to manage the dependencies.

## Download Data
The datasets of Flickr30K and MS-COCO can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC) or follow [VSRN](https://github.com/KunpengLi1994/VSRN). The dataset of IAPR TC-12 can be downloaded from [here](https://drive.google.com/drive/folders/1wXY4nOqopy4H_9Rb6RByjCrK81WQ5Fnl). For VSEinfty_LSEH project change the folder names of all datasets as 'precomp', for other projects change the folder names of the datasets as 'f30k_precomp', 'coco_precomp', and 'iaprtc12_precomp' respectively.

## 1 SVD Descriptions for Datasets
### SVDdescriptions
Modify "$DATA_PATH" in file "train_caps.txt" and run below.
``` 
python SVDdescriptions.py --data_path $DATA_PATH
``` 
Or directly download from [our data](https://drive.google.com/drive/folders/1wXY4nOqopy4H_9Rb6RByjCrK81WQ5Fnl).

See the result file "train_svd.txt" in the root folder "output". Copy "train_svd.txt" into the dataset (e.g. f30k_precomp and coco_precomp).

## 2 VSRN_LSEH
### Training
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --learning_rate 0.0004 --lr_update 5 --max_violation --max_len 60
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name coco_precomp --logger_name runs/coco_VSRN --learning_rate 0.0004 --lr_update 6 --max_violation --max_len 60
``` 
* For IAPR TC-12:
``` 
python IAPRTC12/trainIAPRTC12.py --data_path $DATA_PATH --data_name IAPRTC12_precomp --logger_name runs/IAPR_VSRN --learning_rate 0.0005 --lr_update 20 --max_violation --max_len 60
```

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py/evaluationIAPRTC12.py
``` 

## 3 VSEinfty_LSEH
### Training
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k --logger_name runs/f30k --model_name runs/f30k --learning_rate 0.0008 --lr_update 10
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name coco --logger_name runs/coco --model_name runs/coco --learning_rate 0.0008 --lr_update 10
``` 
* For IAPR TC-12:
``` 
python IAPRTC12/trainIAPRTC12.py --data_path $DATA_PATH --data_name iaprtc12 --logger_name runs/IAPR --model_name runs/IAPR
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run eval.py:
``` 
python eval.py/evalIAPRTC12.py
``` 
## 4 SGRAF_LSEH
### Training
* For Flickr30K:
``` 
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 30 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 30 --lr_update 20 --module_name SAF
```
* For MS-COCO:
``` 
(For SGR) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SGR
(For SAF) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SAF
``` 
* For IAPR TC-12:
``` 
(For SGR) python IAPRTC12/trainIAPRTC12.py --data_name IAPRTC_precomp --module_name SGR
(For SAF) python IAPRTC12/trainIAPRTC12.py --data_name IAPRTC_precomp --module_name SAF
```

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py/evaluationIAPRTC12.py
``` 

## 5 VSEpp_LSEH
### Training
* For Flickr30K:
``` 
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --logger_name runs/f30k_vse++ --max_violation
``` 
* For MS-COCO:
``` 
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --logger_name runs/coco_vse++ --max_violation
``` 
* For IAPR TC-12:
``` 
python IAPRTC12/trainIAPRTC12.py --data_path "$DATA_PATH" --data_name IAPRTC_precomp --logger_name runs/IAPRTC_vse++ --max_violation
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py/evaluationIAPRTC12.py
``` 

## LMH
Comment "ContrastiveLossLSEH" and uncomment "ContrastiveLoss" in the model file of each project. Or follow the offical repositories as follows:

Projects using LMH follow [VSRN](https://pytorch.org/), [VSE∞](https://github.com/woodfrog/vse_infty), [SGRAF](https://github.com/Paranioar/SGRAF), [VSE++](https://github.com/fartashf/vsepp).

# Citation
If you find this code useful for your research, please consider citing:
``` 
@article{gong2022improving,
  title={Improving Visual-Semantic Embeddings by Learning Semantically-Enhanced Hard Negatives for Cross-modal Information Retrieval},
  author={Gong, Yan and Cosma, Georgina},
  journal={Pattern Recognition},
  pages={109272},
  year={2022},
  publisher={Elsevier}
}
``` 


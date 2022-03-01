# VSEnetworksLSEH
This is the PyTorch code for using LSEH of VSE networks, which is described in the paper "Efficient Learning: Semantically Enhanced Hard Negatives for Visual Semantic Embeddings".

## Requirements and installation
We recommended the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/) (>=1.12.1)

* [Pandas](https://pandas.pydata.org/) (>=1.2.3)

* [scikit-learn](https://scikit-learn.org/stable/) (>=0.24.1)

* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger) 

* [pycocotools](https://github.com/cocodataset/cocoapi) 

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 
We used [anaconda](https://www.anaconda.com/) to manage the dependencies, and the code was runing on a NVIDIA TITAN or RTX 3080 GPU.

## Download data
The precomputed image features of Flickr30K and MS-COCO can be downloaded from [VSRN](https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC).

Or follow [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

## 1 SVD descriptions Project
### SVDdescriptions
Modify the "$DATA_PATH" (dataset "train_caps.txt"), then run:
``` 
python SVDdescriptions.py --data_path $DATA_PATH
``` 
Or directly download from [our data](https://drive.google.com/drive/folders/1wXY4nOqopy4H_9Rb6RByjCrK81WQ5Fnl).

Obtain the result file "train_svd.txt" in the folder of root "output", and copy it into the dataset (e.g. f30k_precomp and coco_precomp).

## 2 VSRN_LSEH Projects
### Training
#### VSRN_LSEH
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --learning_rate 0.0004 --lr_update 5 --max_len 60
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --learning_rate 0.0004 --lr_update 6 --max_len 60
``` 

### Evaluation
#### VSRN_LSEH: 
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 

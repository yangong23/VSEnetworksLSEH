# VSEnetworksLSEH
This is the PyTorch code for using LSEH of VSE networks, which is described in the paper " Semantically Enhanced Hard Negatives Learning for Cross-Modal Information Retrieval".

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
Modify "$DATA_PATH" of dataset file "train_caps.txt" and run below. See the result file "train_svd.txt" in the root folder "output"
``` 
python SVDdescriptions.py --data_path $DATA_PATH
``` 
Or directly download from [our data](https://drive.google.com/drive/folders/1wXY4nOqopy4H_9Rb6RByjCrK81WQ5Fnl).

Put "train_svd.txt" into the dataset (e.g. f30k_precomp and coco_precomp).

## 2 VSRN_LSEH Project
### Training
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --learning_rate 0.0004 --lr_update 5 --max_len 60
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name coco_precomp --logger_name runs/coco_VSRN --max_violation --learning_rate 0.0004 --lr_update 6 --max_len 60
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 

## 3 VSEinfty_LSEH Project
### Training
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/f30k --model_name runs/f30k --learning_rate 0.0008 --lr_update 10
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name coco_precomp --logger_name runs/coco --model_name runs/coco --learning_rate 0.001 --lr_update 10
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run eval.py:
``` 
python eval.py
``` 
## 3 SGRAF_LSEH Project
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

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 

## 4 VSEpp_LSEH Project
### Training
* For Flickr30K:
``` 
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --logger_name runs/f30k_vse++ --max_violation
``` 
* For MS-COCO:
``` 
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --logger_name runs/coco_vse++ --max_violation
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 

## 2 GSMN_LSEH Project
### Training
* For Flickr30K:
``` 
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --model_name "$MODEL_PATH" --bi_gru
``` 
* For MS-COCO:
``` 
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --model_name "$MODEL_PATH" --bi_gru
``` 

### Evaluation
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python test.py
``` 



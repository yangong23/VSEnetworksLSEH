# VSEnetworksLSEH

### Requirements and installation
We recommended the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/) (>1.12.1)
* 
* [Pandas](https://pandas.pydata.org/) (>1.2.3)

* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger) 

* [pycocotools](https://github.com/cocodataset/cocoapi) 

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 
We used [anaconda](https://www.anaconda.com/) to manage the dependencies, and the code was runing on a NVIDIA RTX 3080 GPU.

### Download data
The precomputed image features of Flickr30K and MS-COCO can be downloaded from [VSRN](https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC).
Or follow [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

## Run VSRN Projects
### Training new models
#### VSRN
* For Flickr30K:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --learning_rate 0.0004 --lr_update 5 --max_len 60
``` 
* For MS-COCO:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --learning_rate 0.0004 --lr_update 6 --max_len 60
``` 

### Evaluation
* For VSRN: 
Modify the "$MODEL_PATH" and "$DATA_PATH" in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 

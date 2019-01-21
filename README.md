# ulmfit-multilingual
Temporary repository used for collaboration on application of for multiple languages.  
This branch (japanese) contains the code and the pretrained models (sentencepiece model and language model) for Japanese.

## data directory strucutre

Directory structure of data
```
data
├── Aozora
│   ├── csv
│   └── tmp
│       ├── clas
│       └── lm
├── MedWeb
├── wiki
│   ├── ja
│   │   └── models      <- sentencepiece model
│   ├── ja-100
│   │   ├── figs
│   │   └── models      <- pretrained lm
│   ├── ja-2
│   └── ja-all
├── wiki_dumps
└── wiki_extr
    └── ja
        ├── AA
        ├── AB
...
        └── BB
```

## how to contribute
We have a fork of fastai to propose changes to fastai.text, with a branch for this project:
 https://github.com/n-waves/fastai/tree/ulmfit_multilingual  

Let us know that you want to start collaboration on fastai forum thread: [Multilingual ULMFIT](https://forums.fast.ai/t/multilingual-ulmfit/28117)
and you will get access to both repositories.

- Follow the [developer installation of fastai](https://github.com/fastai/fastai#developer-install)


## Repo structure

- `fastai_contrib`  -- anything that can be ported to fastai once we finish the project like:  NLI models, Sentence Piece tok.


## Usage

### 1. Prepare wikipedia data and train a sentencepiece model

``` 
$ sh prepare_wiki.sh ja
```

### 2. Pretrain a language model (lm)

```
$ python -m pretrain_lm --dir_path data/wiki/ja-100/ --lang ja --qrnn True --subword True --max_vocab 8000 --name wt-100 --num_epochs 10 --spm_dir data/wiki/ja/
```

### 3. Finetuning of lm and training of classifier

#### Aozora Bunko

First, you need to download the texts from Aozora Bunko.
To do so, clone the following repository and run aozora_scrape.py  
https://github.com/shibuiwilliam/aozora_classification  
(As of Dec. 2018, the config file (target_author.csv) has a minor bug.
Change http to https and everything will be fine)

Then, run the following lines of code. 

1. prepare data
```
$ python prepare_Aozora_data.py prepare
```

2. fine tune the lm & train a classifier
```
$ python finetune_lm_classify_Aozora.py experiment
```

#### MedWeb

First, you need to download the data from MedWeb website.
To do so, fill in the application form follwing the links from below.
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

Project website: NTCIR-13 MedWeb
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

Place the downloaded files in the following directory like this:  
data/MedWeb/  
L NTCIR-13_MedWeb_ja_training.xlsx  
L NTCIR-13_MedWeb_ja_test.xlsx

1. prepare data
```
$ python prepare_MedWeb_data.py prepare
```

2. fine tune the lm & train a classifier
```
$ python finetune_lm_classify_MedWeb.py experiment
```

### Footnote

To see all available input arguments for finetuning and classifer training (step 3), run `python <name of python file> -- --help` .  
For example, running the following line
```
$ python finetune_lm_classify_MedWeb.py -- --help
```

will return 
```
Type:        type
String form: <class '__main__.MedWeb'>
File:        ~/dev/repo/ulmfit-multilingual/finetune_lm_classify_MedWeb.py
Line:        33
Docstring:   Class for pretraining lm and classifiying lines using MedWeb Bunko data

Usage:       finetune_lm_classify_MedWeb.py [DATA_DIR] [LANG] [CUDA_ID] [PRETRAIN_NAME] [MODEL_DIR] [FIG_DIR] [MAX_VOCAB] [NAME] [DATASET] [SPM_DIR] [QRNN] [BS] [BPTT] [PAD_TOKEN] [FINE_TUNE_LM] [LM_LR] [LM_DROP_MULT] [USE_PRETRAINED_LM] [LM_EPOCHS] [LM_LR_DISCOUNT] [CLAS_LR] [CLAS_DROP_MULT] [USE_LM] [CLAS_LR_DISCOUNT] [LIN_FTRS] [CLS_WEIGHTS] [FRAC_DS] [DETERMINISTIC] [EXPERIMENT_NAME]
             finetune_lm_classify_MedWeb.py [--data-dir DATA_DIR] [--lang LANG] [--cuda-id CUDA_ID] [--pretrain-name PRETRAIN_NAME] [--model-dir MODEL_DIR] [--fig-dir FIG_DIR] [--max-vocab MAX_VOCAB] [--name NAME] [--dataset DATASET] [--spm-dir SPM_DIR] [--qrnn QRNN] [--bs BS] [--bptt BPTT] [--pad-token PAD_TOKEN] [--fine-tune-lm FINE_TUNE_LM] [--lm-lr LM_LR] [--lm-drop-mult LM_DROP_MULT] [--use-pretrained-lm USE_PRETRAINED_LM] [--lm-epochs LM_EPOCHS] [--lm-lr-discount LM_LR_DISCOUNT] [--clas-lr CLAS_LR] [--clas-drop-mult CLAS_DROP_MULT] [--use-lm USE_LM] [--clas-lr-discount CLAS_LR_DISCOUNT] [--lin-ftrs LIN_FTRS] [--cls-weights CLS_WEIGHTS] [--frac-ds FRAC_DS] [--deterministic DETERMINISTIC] [--experiment-name EXPERIMENT_NAME]

```

# coding: utf-8
"""
# # 青空文庫の著作切れ作品の作者を当てる
# 
# - 作品著作権フラグのない作品のうち11176件を形態素解析した結果が公開されている。新仮名データをダウンロードした。
# - http://aozora-word.hahasoha.net/index.html
# - この中から以下の著者のデータをダウンロードして、どの著者のデータか予測するモデルを作る(multi class)
#     - akutagawa  
#     - mori 
#     - natsume 
#     - sakaguchi 
#     - yoshikawa 
"""

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from fastai.text import TextLMDataBunch, TextClasDataBunch, language_model_learner, text_classifier_learner, LanguageLearner
from fastai import fit_one_cycle
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists, get_sentencepiece
from fastai.text.transform import Vocab
from fastai.metrics import accuracy, accuracy_thresh, fbeta
from pathlib import Path
import pandas as pd
import csv
from functools import partial
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow
import glob
import shutil
import dask.dataframe as dd
import dask_ml as dm
from matplotlib import pyplot as plt
import itertools


def copy_csv(dir_orig="../aozora_classification/aozora_data/", dir_dest="data/Aozora/csv/"):
    "copy csvs from original directory"

    Path(dir_dest).mkdir(parents=True, exist_ok=True)
    csvs = glob.glob(dir_orig+"*/csv/*.csv", recursive=True)

    for f in csvs:
        shutil.copy(f,dir_dest)


def split_data(dir_csv="data/Aozora/csv/"):
    "collect lines from all authors and split them into train/valid/test"

    # ## read data
    ddf = dd.read_csv(dir_csv+'*.csv')
    ddf = ddf.rename(columns={'Unnamed: 0':'line_num'})

    # filter lines: remove headers and short lines
    # line number >=20 and number of letters >= 20
    ddf_fitered = ddf.loc[(ddf['line_num']>=20) & (ddf['line'].str.len()>=20),['auth','line']]

    # ### train test split
    df = ddf_fitered.compute()
    trn_val, tst = train_test_split(df,test_size=0.05,random_state=0)
    trn, val = train_test_split(trn_val,test_size=5/95,random_state=0)

    # print data size
    print('Data size')
    print('Train: {}'.format(trn.shape[0]))
    print('Validate: {}'.format(val.shape[0]))
    print('Test: {}'.format(tst.shape[0]))

    # ### save
    trn.to_csv('data/Aozora/train.csv',header=None,index=None)
    val.to_csv('data/Aozora/valid.csv',header=None,index=None)
    tst.to_csv('data/Aozora/test.csv',header=None,index=None)


def prepare_databunch(spm_dir, max_vocab, dataset_dir, num_trains):
    "prepare databunch for lm and classification using different sizes of train data"
    # here we're just loading the trained spm model
    sp = get_sentencepiece(spm_dir, None, 'wt-all', vocab_size=max_vocab)


    # load train, valid in df
    train_df = pd.read_csv(dataset_dir/'train.csv',header=None)
    valid_df = pd.read_csv(dataset_dir/'valid.csv',header=None)
    test_df = pd.read_csv(dataset_dir/'test.csv',header=None)

    # create databunch for different sizes of train data
    print("Creating databunch for different sizes of train data")
    for num_train in num_trains:
        num_train = int(num_train)
        print("Size: {}".format(num_train))
        
        # sample train data
        tmp_train_df = train_df.sample(n=num_train,random_state=42)
        
        # create data bunches
        tmp_lm = TextLMDataBunch.from_df(path=dataset_dir,train_df=tmp_train_df,valid_df=valid_df,test_df=test_df,**sp)
        tmp_clas = TextClasDataBunch.from_df(path=dataset_dir,train_df=tmp_train_df,valid_df=valid_df,test_df=test_df,**sp)
        
        # name of tmp cache
        tmp_cache_lm = Path(f'tmp/lm/{num_train}')
        tmp_cache_clas = Path(f'tmp/clas/{num_train}')
        
        # save
        tmp_lm.save(tmp_cache_lm)
        tmp_clas.save(tmp_cache_clas)


def set_deterministic(deterministic=True):
    "reset states of torch and numpy for reproducibility of results"

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)


def lm_fine_tuning( data_lm, lm_lr, bptt, emb_sz, nh, nl, qrnn,
                pad_token,
                pretrained_fnames, 
                model_dir,
                lm_enc_finetuned,
                lm_drop_mult, use_pretrained_lm,fine_tune_lm,deterministic):
    
    
    if not use_pretrained_lm:
        pretrained_fnames = None
    
    learn = language_model_learner(
        data_lm, bptt=bptt, emb_sz=emb_sz, nh=nh, nl=nl, qrnn=qrnn,
        pad_token=pad_token,
        pretrained_fnames=pretrained_fnames, 
        path=model_dir.parent, model_dir=model_dir.name,
        drop_mult=lm_drop_mult
    )
    
    set_deterministic(deterministic)
    
    if fine_tune_lm:
        print('Fine-tuning the language model...')
        learn.unfreeze()
        learn.fit(4, slice(lm_lr/(10**2), lm_lr))
    #     learn.fit(10, slice(1e-3, 1e-1))
    else:
        print('Skipping fine tuning')

    # save fine tuned lm
    print(f"Saving models at {learn.path / learn.model_dir}")
    learn.save_encoder(lm_enc_finetuned)
    
    return learn


def classify_multiclas(data_clas, clas_lr, bptt, 
                        pad_token,
                        model_dir,
                        qrnn, emb_sz, nh, nl,clas_drop_mult,
                        fine_tune_lm, use_lm,deterministic):
    
    
    print("Starting classifier training")
    learn = text_classifier_learner(data_clas, bptt=bptt, pad_token=pad_token,
                                  path=model_dir.parent, model_dir=model_dir.name,
                                  qrnn=qrnn, emb_sz=emb_sz, nh=nh, nl=nl,drop_mult=clas_drop_mult
                                    )
    learn.model.reset()
    
    if use_lm:
        print('Loading language model')
        lm_enc = lm_enc_finetuned # if fine_tune_lm else lm_name
        learn.load_encoder(lm_enc)
    else:
        print('Training from scratch without language model')

    # train
    set_deterministic(deterministic)
    
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, clas_lr, moms=(0.8, 0.7), wd=1e-7)

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(clas_lr / (2.6 ** 4), clas_lr), moms=(0.8, 0.7), wd=1e-7)

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(clas_lr / (2.6 ** 4), clas_lr), moms=(0.8, 0.7), wd=1e-7)

    learn.unfreeze()
    learn.fit_one_cycle(2, slice(clas_lr / (2.6 ** 4), clas_lr), moms=(0.8, 0.7), wd=1e-7)
    
    # results
    results={}
    results['accuracy'] = learn.validate()[1]
    print(results)
    
    # save classifier
    print(f"Saving models at {learn.path / learn.model_dir}")
    learn.save(f'{model_name}_{name}')
    
    # prediction on validation data
    preds=learn.get_preds()

    # prediction
    p=preds[0]
    p = p.tolist()
    p = np.asarray(p)

    # binarize
    y_pred = p.argmax(axis=1)


    # target
    t = preds[1].tolist()
    y_true = np.asarray(t)
    
    return y_true, y_pred


def evaluation(y_true,y_pred):
    ## metrics
    f1_micro = f1_score(y_true,y_pred,average='micro')
    f1_macro = f1_score(y_true,y_pred,average='macro')

    print('F1 score')
    print('micro: {}'.format(f1_micro))
    print('macro: {}'.format(f1_macro))

    precision_micro = precision_score(y_true,y_pred,average='micro')
    precision_macro = precision_score(y_true,y_pred,average='macro')

    print('\nPrecision score')
    print('micro: {}'.format(precision_micro))
    print('macro: {}'.format(precision_macro))

    recall_micro = recall_score(y_true,y_pred,average='micro')
    recall_macro = recall_score(y_true,y_pred,average='macro')

    print('\nRecall score')
    print('micro: {}'.format(recall_micro))
    print('macro: {}'.format(recall_macro))
    
    acc = accuracy_score(y_true,y_pred)

    print('\nAccuracy score: {}'.format(acc))
    
    results = dict(
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        accuracy=acc
    )
    
    return results


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def save_confusion_matrix(y_true,y_pred,class_names,fn):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix')
    plt.gcf().savefig(fn)


# Create Params dictionary
class Params(object):
    def __init__(self, qrnn,bs,bptt,emb_sz, nh, nl,fine_tune_lm,lm_lr,lm_drop_mult,
                 use_pretrained_lm,clas_lr,clas_drop_mult,use_lm,num_train,deterministic):
        self.qrnn = qrnn
        self.fine_tune_lm = fine_tune_lm
        self.bs = bs
        self.bptt = bptt
        self.emb_sz = emb_sz
        self.nh = nh
        self.nl = nl
        self.lm_lr = lm_lr
        self.lm_drop_mult = lm_drop_mult
        self.use_pretrained_lm = use_pretrained_lm
        self.clas_lr = clas_lr
        self.clas_drop_mult = clas_drop_mult
        self.use_lm = use_lm
        self.num_train = num_train
        self.deterministic = deterministic        


def main():
    # ###  Parameters
    # ## prepare data for lm fine tuning and classification
    # basically copying from train_cls.py

    data_dir='data'
    lang='ja' 
    cuda_id=0 
    pretrain_name='wt-100' 
    model_dir='data/wiki/ja-100/models'
    fig_dir='data/wiki/ja-100/figs'
    max_vocab=8000
    name='Aozora-clas'
    dataset='Aozora' 
    frac_ds=1.0
    spm_dir = 'data/wiki/ja/'

    # full train data has about 400k items so these numbers of train data will be saved for experiments
    num_trains = [1e2,1e3,1e4,1e5]

    data_dir = Path(data_dir)
    assert data_dir.name == 'data',    f'Error: Name of data directory should be data, not {data_dir.name}.'
    dataset_dir = data_dir / dataset
    model_dir = Path(model_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)

    if not torch.cuda.is_available():
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    print(f'Dataset: {dataset}. Language: {lang}.')

    # common to lm fine tuning and classification
    qrnn=True
    bs=20 
    bptt=70
    pad_token=PAD_TOKEN_ID

    if qrnn:
        emb_sz, nh, nl = 400, 1550, 3
    else:
        emb_sz, nh, nl = 400, 1150, 3
        
    # lm fine tuning
    fine_tune_lm=True 
    lm_lr=1e-2
    lm_drop_mult=0.5
    use_pretrained_lm=False

    # classification
    clas_lr=1e-2
    clas_drop_mult=0.1
    use_lm=True

    # number of training data
    num_train = 1000

    # reduce fluctuation between runs
    deterministic = True


    mlflow.set_experiment('20181213_Aozora_vocab8k')


    with mlflow.start_run():
        set_deterministic(deterministic)
        
        ## Log our parameters into mlflow
        num_train = int(num_train)
        args = Params(qrnn,bs,bptt,emb_sz, nh, nl,fine_tune_lm,lm_lr,lm_drop_mult,
                    use_pretrained_lm,clas_lr,clas_drop_mult,use_lm,num_train,deterministic)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        
        ## create data set
        data_lm = TextLMDataBunch.load(path=Path(data_dir/dataset),cache_name=Path(f'tmp/lm/{num_train}'))
        data_clas = TextClasDataBunch.load(path=Path(data_dir/dataset),cache_name=Path(f'tmp/clas/{num_train}'))
        
        ## preprocess
        if qrnn:
            print('Using QRNNs...')
        model_name = 'qrnn' if qrnn else 'lstm'
        lm_name = f'{model_name}_{pretrain_name}'
        pretrained_fnames = (lm_name, f'itos_{pretrain_name}')

        ensure_paths_exists(data_dir,
                            dataset_dir,
                            model_dir,
                            model_dir/f"{pretrained_fnames[0]}.pth",
                            model_dir/f"{pretrained_fnames[1]}.pkl")
        lm_enc_finetuned  = f"{lm_name}_{dataset}_enc"
        cm_name = fig_dir / f'{lm_name}_{dataset}_confusionmatrix.png'

        ## fine tune lm 
        if use_lm:
            lm=lm_fine_tuning( data_lm, lm_lr, bptt, emb_sz, nh, nl, qrnn,
                        pad_token,
                        pretrained_fnames, 
                        model_dir,
                        lm_enc_finetuned,
                        lm_drop_mult, use_pretrained_lm,fine_tune_lm,deterministic)


        ## create classifier
        y_true, y_pred = classify_multiclas(data_clas, clas_lr, bptt, 
                            pad_token,
                            model_dir,
                            qrnn, emb_sz, nh, nl,clas_drop_mult,
                            fine_tune_lm, use_lm,deterministic)

        ## Evaluation
        results = evaluation(y_true,y_pred)
        class_names = data_clas.valid_ds.y.classes
        save_confusion_matrix(y_true,y_pred,class_names,cm_name)
        
        ## Log metrics
        for key, value in results.items():
            mlflow.log_metric(key, value)
            
        ## Save artifacts
        # pretrained lm
        mlflow.log_artifact(model_dir/f"{pretrained_fnames[0]}.pth")
        mlflow.log_artifact(model_dir/f"{pretrained_fnames[1]}.pkl")
        
        # fine tuned lm
        if fine_tune_lm & use_lm:
            mlflow.log_artifact(model_dir / f'{lm_enc_finetuned}.pth')
        
        # classifier
        mlflow.log_artifact(model_dir / f'{model_name}_{name}.pth')
        
        # confusion matrix
        mlflow.log_artifact(cm_name)


if __name__ == '__main__':
    main()

    



# coding: utf-8

# In[2]:


"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
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

import fire
from collections import Counter
from pathlib import Path

import pandas as pd
import csv
from functools import partial
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import os


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # NTCIR-13 MedWeb
# http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-ja-MedWeb.html

# ## copy data

# In[4]:


get_ipython().system('ls /home/ubuntu/dev/download/data/MedWeb_TestCollection/')


# In[5]:


get_ipython().system('cat /home/ubuntu/dev/download/data/MedWeb_TestCollection/readme.txt')


# In[6]:


get_ipython().system('mkdir -p data/MedWeb')


# In[7]:


get_ipython().system('cp /home/ubuntu/dev/download/data/MedWeb_TestCollection/*ja* data/MedWeb/')


# In[8]:


get_ipython().system('ls data/MedWeb/')


# ## read data

# ### train

# In[9]:


trn = pd.read_excel('data/MedWeb/NTCIR-13_MedWeb_ja_training.xlsx',sheet_name='ja_train')


# In[10]:


trn.head(100)


# In[11]:


# name of labels
lbl_names = trn.columns[2:]
lbl_names


# In[12]:


# save
pickle.dump(lbl_names, open('data/MedWeb/lbl_names.pkl','wb'))


# 3列目以降がpの列名を列挙して、ラベルとする

# trn_lbl = ['_'.join(row) for row in ((trn.iloc[:,2:] == 'p')*1).values.astype(str)]

# trn_lbl = [','.join(row) for row in trn.columns[2:] + '_' +trn.iloc[:,2:]]

# for i, row in trn.iterrows():
#     print(','.join(lbl_names[row[2:]=='p']))
#     if i > 5:
#         break

# trn_lbl = []
# for i, row in trn.iterrows():
#     trn_lbl.append(lbl_idx[(row[2:]=='p').values])

# In[13]:


lbl_idx = np.asarray([i for i in range(len(lbl_names))])


# In[14]:


lbl_idx


# In[15]:


trn_lbl = []
for i, row in trn.iterrows():
    trn_lbl.append(lbl_names[(row[2:]=='p')].tolist())


# In[16]:


trn_lbl[:5]


# In[17]:


# put data into df and save
trn_df = pd.DataFrame({'text':trn['Tweet'].values, 'labels':trn_lbl},columns=['labels', 'text'])
trn_df.to_csv('data/MedWeb/train.csv', header=False, index=False)


# In[18]:


trn_df


# ### test

# In[19]:


tst = pd.read_excel('data/MedWeb/NTCIR-13_MedWeb_ja_test.xlsx',sheet_name='ja_test')


# In[20]:


tst.head(100)


# In[21]:


tst_lbl = []
for i, row in tst.iterrows():
    tst_lbl.append(lbl_names[(row[2:]=='p')].tolist())


# In[22]:


tst_lbl[:5]


# In[23]:


# name of labels
lbl_names = tst.columns[2:]
lbl_names


# In[24]:


# put data into df and save
tst_df = pd.DataFrame({'text':tst['Tweet'].values, 'labels':tst_lbl},columns=['labels', 'text'])
# change name from test to valid (fastai lm expects valid)
tst_df.to_csv('data/MedWeb/valid.csv', header=False, index=False)


# In[25]:


tst_df.head()


# ## prepare data for lm fine tuning and classification
# basically copying from train_cls.py

# In[26]:


data_dir='data'
lang='ja' 
cuda_id=0 
pretrain_name='wt-100' 
model_dir='data/wiki/ja-100/models'
max_vocab=8000
name='MedWeb-clas'
dataset='MedWeb' 
frac_ds=1.0
spm_dir = 'data/wiki/ja/'


# In[27]:


data_dir = Path(data_dir)
assert data_dir.name == 'data',    f'Error: Name of data directory should be data, not {data_dir.name}.'
dataset_dir = data_dir / dataset
model_dir = Path(model_dir)


# In[28]:


if not torch.cuda.is_available():
    print('CUDA not available. Setting device=-1.')
    cuda_id = -1
torch.cuda.set_device(cuda_id)

print(f'Dataset: {dataset}. Language: {lang}.')


# In[29]:


# here we're just loading the trained spm model
sp = get_sentencepiece(spm_dir, None, 'wt-all', vocab_size=max_vocab)


# In[30]:


# load train, valid in df
train_df = pd.read_csv(dataset_dir/'train.csv',header=None)
valid_df = pd.read_csv(dataset_dir/'valid.csv',header=None)


# In[31]:


train_df.head()


# ## tokenization

# In[32]:


t=sp['tokenizer']


# In[33]:


trn_tok = np.asarray(t.process_all(train_df[1]))


# In[34]:


val_tok = np.asarray(t.process_all(valid_df[1]))


# In[35]:


trn_lbls = train_df[0].apply(eval).values


# In[36]:


val_lbls = valid_df[0].apply(eval).values


# # LM fine tuning & Classification

# In[37]:


def set_deterministic(deterministic=True):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)


# In[38]:


def lm_fine_tuning( data_lm, lm_lr, bptt, emb_sz, nh, nl, qrnn,
                   pad_token,
                   pretrained_fnames, 
                   model_dir,
                   lm_enc_finetuned,
                   lm_drop_mult,
                   use_pretrained_lm,
                   fine_tune_lm,
                   deterministic,
                   lm_epochs=10,
                   lm_lr_discount=10**2
                  ):
    
    
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
        learn.fit(lm_epochs, slice(lm_lr/lm_lr_discount, lm_lr))
    #     learn.fit(10, slice(1e-3, 1e-1))
    else:
        print('Skipping fine tuning')

    # save fine tuned lm
    print(f"Saving models at {learn.path / learn.model_dir}")
    learn.save_encoder(lm_enc_finetuned)
    
    return learn


# In[39]:


def classify_multilabel(data_clas, clas_lr, bptt, 
                        pad_token,
                        model_dir,
                        qrnn, emb_sz, nh, nl,
                        clas_drop_mult,
                        fine_tune_lm, use_lm,
                        cls_weights,
                        deterministic,
                        clas_lr_discount=2.6**4,
                        lin_ftrs=[50]
                       ):
    
    
    # change metric from accuracy to accuarcy_thresh and f1
    accuracy_multi =  partial(accuracy_thresh,thresh=0.5,sigmoid=True)
    f1 = partial(fbeta,thresh=0.5,beta=1,sigmoid=True) # corresponds to f1 score in sklearn with 'samples' option?

    print("Starting classifier training")
    learn = text_classifier_learner(data_clas, bptt=bptt, pad_token=pad_token,
                                  path=model_dir.parent, model_dir=model_dir.name,
                                  qrnn=qrnn, emb_sz=emb_sz, nh=nh, nl=nl,drop_mult=clas_drop_mult,
                                    lin_ftrs=lin_ftrs
                                    )
#     learn.model.reset()
    
    if use_lm:
        print('Loading language model')
        lm_enc = lm_enc_finetuned # if fine_tune_lm else lm_name
        learn.load_encoder(lm_enc)
    else:
        print('Training from scratch without language model')

    # change metric
    learn.metrics = [accuracy_multi,f1]

    # CRITICAL STEP
    #- need to adjust pos_weight of loss function to get the model working
    if cls_weights is not None:
        pos_weight = torch.cuda.FloatTensor(cls_weights[data_clas.train_ds.y.classes].values)
        bce_logits_weighted = partial(F.binary_cross_entropy_with_logits,  pos_weight=pos_weight)
        learn.loss_func = bce_logits_weighted

    # train
    set_deterministic(deterministic)
    
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, clas_lr, moms=(0.8, 0.7), wd=1e-7)

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(clas_lr / clas_lr_discount, clas_lr), moms=(0.8, 0.7), wd=1e-7)

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(clas_lr / clas_lr_discount, clas_lr), moms=(0.8, 0.7), wd=1e-7)

    learn.unfreeze()
    learn.fit_one_cycle(2, slice(clas_lr /clas_lr_discount, clas_lr), moms=(0.8, 0.7), wd=1e-7)
    
    # results
    results={}
    results['accuracy_multi'] = learn.validate()[1]
    results['F1'] = learn.validate()[2]
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
    y_pred = (p>=0.5)*1.0


    # target
    t = preds[1].tolist()
    y_true = np.asarray(t)
    
    return y_true, y_pred


# In[40]:


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
    
    results = dict(
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        recall_micro=recall_micro,
        recall_macro=recall_macro
    )
    
    return results


# In[41]:


# Create Params dictionary
class Params(object):
    def __init__(self, qrnn,bs,bptt,emb_sz, nh, nl,fine_tune_lm,lm_lr,lm_drop_mult,
                 use_pretrained_lm,clas_lr,clas_drop_mult,use_lm,cls_weights,frac_ds,deterministic,lm_epochs,lm_lr_discount,clas_lr_discount,
                lin_ftrs):
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
        self.cls_weights = cls_weights.values.tolist()
        self.frac_ds = frac_ds
        self.deterministic = deterministic 
        self.lm_epochs = lm_epochs 
        self.lm_lr_discount = lm_lr_discount     
        self.clas_lr_discount = clas_lr_discount 
        self.lin_ftrs = lin_ftrs                         


# ###  Parameters

# In[56]:


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
lm_lr=5e-3
lm_lr_discount=10
lm_drop_mult=0.5
use_pretrained_lm=True
lm_epochs=10

# classification
clas_lr=1e-2
clas_lr_discount=2.6**4
clas_drop_mult=0.1
use_lm=True
lin_ftrs=[50*2]

# reduction of training data
frac_ds = 1.0

# reduce fluctuation between runs
deterministic = True


# In[57]:


# class weights for classification loss function
# cls_weights = trn.shape[0]/(trn.iloc[:,2:]=='p').sum()/5
cls_weights = pd.Series(np.ones(8,)*1.0,index=trn.columns[2:])
cls_weights


# In[58]:


mlflow.set_experiment('20181214_MedWeb_vocab8k')


# In[59]:


with mlflow.start_run():
    set_deterministic(deterministic)
    
    ## Log our parameters into mlflow
    args = Params(qrnn,bs,bptt,emb_sz, nh, nl,fine_tune_lm,lm_lr,lm_drop_mult,
                 use_pretrained_lm,clas_lr,clas_drop_mult,use_lm,cls_weights,frac_ds,deterministic,lm_epochs,lm_lr_discount,clas_lr_discount,
                 lin_ftrs)
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    
    ## create data set
    n_trn = trn_tok.shape[0]
    size = int(n_trn*frac_ds)
    idx = np.random.choice(n_trn,size=size,replace=False)
    data_lm = TextLMDataBunch.from_tokens(path=dataset_dir,trn_tok=trn_tok[idx],val_tok=val_tok,trn_lbls=[],val_lbls=[],vocab=sp['vocab'])
    data_clas = TextClasDataBunch.from_tokens(path=dataset_dir,trn_tok=trn_tok[idx],val_tok=val_tok,
                                              trn_lbls=trn_lbls[idx],
                                              val_lbls=val_lbls,
                                              vocab=sp['vocab']
                                             )
    
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

    ## fine tune lm
    p = None
    
    set_deterministic(deterministic)
    if use_lm:
        lm=lm_fine_tuning( data_lm, lm_lr, bptt, emb_sz, nh, nl, qrnn,
                          pad_token,
                          pretrained_fnames, 
                          model_dir,
                          lm_enc_finetuned,
                          lm_drop_mult,
                          use_pretrained_lm,
                          fine_tune_lm,
                          deterministic,
                          lm_epochs,
                          lm_lr_discount
                         )
        # check lm quality by predicting words after some text
        os.makedirs('tmp',exist_ok=True)
        p = lm.predict('料金が高い',n_words=200,no_unk=False)
        print(p)
        with open('tmp/lm_predicted.txt','w') as f:
            f.write(p)


    ## create classifier
    set_deterministic(deterministic)
    y_true, y_pred = classify_multilabel(data_clas, clas_lr, bptt, 
                                         pad_token,
                                         model_dir,
                                         qrnn, 
                                         emb_sz, 
                                         nh, nl,
                                         clas_drop_mult,
                                         fine_tune_lm,
                                         use_lm,cls_weights,
                                         deterministic,
                                         clas_lr_discount,
                                         lin_ftrs
                                        )

    ## Evaluation
    results = evaluation(y_true,y_pred)
    
    ## Log metrics
    for key, value in results.items():
        mlflow.log_metric(key, value)
        
    ## Save artifacts
    # pretrained lm
    mlflow.log_artifact(model_dir/f"{pretrained_fnames[0]}.pth")
    mlflow.log_artifact(model_dir/f"{pretrained_fnames[1]}.pkl")
    
    # fine tuned lm
    if fine_tune_lm:
         mlflow.log_artifact(model_dir / f'{lm_enc_finetuned}.pth')
    
    # classifier
    mlflow.log_artifact(model_dir / f'{model_name}_{name}.pth')
    
    # predicted text
    if p:
        mlflow.log_artifact('tmp/lm_predicted.txt')
        print('\n'+p)


# coding: utf-8
"""
Pretrain lm and classify the texts (tweets) using MedWeb data

First, you need to prepare the data using prepare_MedWeb_data.py
After doing so, run this code to pretrain the lm and classify the tweets.
Please be aware that this is a multilabel task, not a multiclass task
(i.e. Each tweet can have multiple labels)

For usage instructions execute the following lines:
>>> python pretrain_lm_classify_MedWeb.py -- --help
>>> python pretrain_lm_classify_MedWeb.py experiment -- --help

"""

import fire
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
from functools import partial
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import mlflow
from matplotlib import pyplot as plt
import itertools

class MedWeb():
    """
    Class for pretraining lm and classifiying lines using MedWeb Bunko data
    """
    def __init__(
        self, 
        data_dir='data',
        lang='ja' ,
        cuda_id=0 ,
        pretrain_name='wt-100', 
        model_dir='data/wiki/ja-100/models',
        fig_dir='data/wiki/ja-100/figs',
        max_vocab=8000,
        name='MedWeb-clas',
        dataset='MedWeb' ,
        spm_dir = 'data/wiki/ja/',
        qrnn=True,
        bs=20, 
        bptt=70,
        pad_token=PAD_TOKEN_ID,
        fine_tune_lm=True ,
        lm_lr=1e-2,
        lm_drop_mult=0.5,
        use_pretrained_lm=True,
        lm_epochs=10,
        lm_lr_discount=10**2,
        clas_lr=1e-2,
        clas_drop_mult=0.1,
        use_lm=True,
        clas_lr_discount=2.6**4,
        lin_ftrs=[50],
        cls_weights=[1.0]*8,
        frac_ds = 1.0,
        deterministic = True,
        experiment_name = 'MedWeb_classification',
        ):

        # set parameters
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.dataset_dir = self.data_dir / self.dataset
        self.spm_dir = spm_dir
        self.pretrain_name = pretrain_name
        self.name = name
        self.model_dir = Path(model_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)
        self.max_vocab = max_vocab
        self.qrnn = qrnn
        self.bs = bs 
        self.bptt = bptt
        self.pad_token = pad_token
        self.fine_tune_lm = fine_tune_lm
        self.lm_lr = lm_lr
        self.lm_drop_mult = lm_drop_mult
        self.use_pretrained_lm = use_pretrained_lm
        self.lm_epochs = lm_epochs
        self.lm_lr_discount = lm_lr_discount
        self.clas_lr = clas_lr
        self.clas_drop_mult = clas_drop_mult
        self.use_lm = use_lm
        self.clas_lr_discount = clas_lr_discount
        self.lin_ftrs = lin_ftrs
        self.cls_weights = cls_weights
        self.frac_ds = frac_ds
        self.deterministic = deterministic
        self.experiment_name = experiment_name

        # set network parameters
        if self.qrnn:
            self.emb_sz, self.nh, self.nl = 400, 1550, 3
        else:
            self.emb_sz, self.nh, self.nl = 400, 1150, 3

        # use gpu if available
        if not torch.cuda.is_available():
            print('CUDA not available. Setting device=-1.')
            cuda_id = -1
        torch.cuda.set_device(cuda_id)

        # set names
        if self.qrnn:
            print('Using QRNNs...')
        self.model_name = 'qrnn' if self.qrnn else 'lstm'
        
        self.lm_name = f'{self.model_name}_{self.pretrain_name}'
        self.pretrained_fnames = (self.lm_name, f'itos_{self.pretrain_name}')
        self.lm_enc_finetuned  = f"{self.lm_name}_{self.dataset}_enc"
        self.cm_name = self.fig_dir / f'{self.lm_name}_{self.dataset}_confusionmatrix.png'

        # ensure path exists
        ensure_paths_exists(self.data_dir,
                            self.dataset_dir,
                            self.model_dir,
                            self.model_dir/f"{self.pretrained_fnames[0]}.pth",
                            self.model_dir/f"{self.pretrained_fnames[1]}.pkl")
        
        # show dataset and language
        print(f'Dataset: {dataset}. Language: {lang}.')


    def set_deterministic(self):
        "reset states of torch and numpy for reproducibility of results"

        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(0)
            np.random.seed(0)


    def lm_fine_tuning(self, data_lm):
        "fine tune lm"
        
        if not self.use_pretrained_lm:
            self.pretrained_fnames = None
        
        learn = language_model_learner(
            data_lm, 
            bptt=self.bptt,
            emb_sz=self.emb_sz, 
            nh=self.nh, 
            nl=self.nl, 
            qrnn=self.qrnn,
            pad_token=self.pad_token,
            pretrained_fnames=self.pretrained_fnames, 
            path=self.model_dir.parent, 
            model_dir=self.model_dir.name,
            drop_mult=self.lm_drop_mult
        )
        
        self.set_deterministic()
        
        if self.fine_tune_lm:
            print('Fine-tuning the language model...')
            learn.unfreeze()
            learn.fit(self.lm_epochs, slice(self.lm_lr/self.lm_lr_discount, self.lm_lr))
        
        else:
            print('Skipping fine tuning')

        # save fine tuned lm
        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save_encoder(self.lm_enc_finetuned)
        
        return learn


    def classify_multilabel(self, data_clas):
        """
        Create a classifier to predict the symptoms in each text (tweet)
        Please be aware that this is a multilabel task, not a multiclass task.
        """
        
        # change metric from accuracy to accuarcy_thresh and f1 to evaluate multilabel predictions
        accuracy_multi =  partial(accuracy_thresh,thresh=0.5,sigmoid=True)
        f1 = partial(fbeta,thresh=0.5,beta=1,sigmoid=True) # corresponds to f1 score in sklearn with 'samples' option?

        # train classifier
        print("Starting classifier training")
        learn = text_classifier_learner(
            data_clas, 
            bptt=self.bptt, 
            pad_token=self.pad_token,
            path=self.model_dir.parent, 
            model_dir=self.model_dir.name,
            qrnn=self.qrnn, 
            emb_sz=self.emb_sz, 
            nh=self.nh, 
            nl=self.nl,
            drop_mult=self.clas_drop_mult,
            lin_ftrs=self.lin_ftrs
            )
        learn.model.reset()
        
        if self.use_lm:
            print('Loading language model')
            lm_enc = self.lm_enc_finetuned # if fine_tune_lm else lm_name
            learn.load_encoder(lm_enc)
        else:
            print('Training from scratch without language model')

        # change metric
        learn.metrics = [accuracy_multi,f1]

        # CRITICAL STEP
        #- need to adjust pos_weight of loss function to get the model working
        if self.cls_weights is not None:
            pos_weight = torch.cuda.FloatTensor(np.asarray(self.cls_weights))
            bce_logits_weighted = partial(F.binary_cross_entropy_with_logits,  pos_weight=pos_weight)
            learn.loss_func = bce_logits_weighted

        # train
        self.set_deterministic()
        
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, self.clas_lr, moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(self.clas_lr / self.clas_lr_discount, self.clas_lr), moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(self.clas_lr / self.clas_lr_discount, self.clas_lr), moms=(0.8, 0.7), wd=1e-7)

        learn.unfreeze()
        learn.fit_one_cycle(2, slice(self.clas_lr / self.clas_lr_discount, self.clas_lr), moms=(0.8, 0.7), wd=1e-7)
        
        # results
        results={}
        results['accuracy_multi'] = learn.validate()[1]
        results['F1'] = learn.validate()[2]
        print(results)
        
        # save classifier
        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save(f'{self.model_name}_{self.name}')
        
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


    def evaluation(self,y_true,y_pred):
        "Evaluate the classification results"

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


    def experiment(self):
        """
        Performs an experiment with provided parameters
        Steps
        1. Pretrain lm 
        2. Classify tweets
        """

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            self.set_deterministic()
            
            ## Log our parameters into mlflow
            for key, value in vars(self).items():
                if isinstance(value, (int, float, list)):
                    mlflow.log_param(key, value)
            
            # load sentencepiece model
            sp = get_sentencepiece(self.spm_dir, None, 'wt-all', vocab_size=self.max_vocab)
        
            # load tokens and labels
            trn_tok = np.load(self.dataset_dir / 'trn_tok.npy')
            val_tok = np.load(self.dataset_dir / 'val_tok.npy')

            trn_lbls = np.load(self.dataset_dir / 'trn_lbl.npy')
            val_lbls = np.load(self.dataset_dir / 'val_lbl.npy')            

            ## create data set
            n_trn = trn_tok.shape[0]
            size = int(n_trn*self.frac_ds)
            idx = np.random.choice(n_trn,size=size,replace=False)
            data_lm = TextLMDataBunch.from_tokens(
                path=self.dataset_dir,
                trn_tok=trn_tok[idx],
                val_tok=val_tok,
                trn_lbls=[],
                val_lbls=[],
                vocab=sp['vocab']
                )
            data_clas = TextClasDataBunch.from_tokens(
                path=self.dataset_dir,
                trn_tok=trn_tok[idx],
                val_tok=val_tok,
                trn_lbls=trn_lbls[idx],
                val_lbls=val_lbls,
                vocab=sp['vocab']
                )
            

            ## fine tune lm 
            p = None
    
            self.set_deterministic()
   
            if self.use_lm:
                lm=self.lm_fine_tuning(data_lm)

                # check lm quality by predicting words after some text
                Path('tmp').mkdir(exist_ok=True)
                p = lm.predict('料金が高い',n_words=200,no_unk=False)
                with open('tmp/lm_predicted.txt','w') as f:
                    f.write(p)

            ## create classifier
            self.set_deterministic()
            y_true, y_pred = self.classify_multilabel(data_clas)

            ## Evaluation
            results = self.evaluation(y_true,y_pred)
            
            ## Log metrics
            for key, value in results.items():
                mlflow.log_metric(key, value)
                
            ## Save artifacts
            # pretrained lm
            if self.use_pretrained_lm:
                mlflow.log_artifact(self.model_dir/f"{self.pretrained_fnames[0]}.pth")
                mlflow.log_artifact(self.model_dir/f"{self.pretrained_fnames[1]}.pkl")
            
            # fine tuned lm
            if self.fine_tune_lm & self.use_lm:
                mlflow.log_artifact(self.model_dir / f'{self.lm_enc_finetuned}.pth')
            
            # classifier
            mlflow.log_artifact(self.model_dir / f'{self.model_name}_{self.name}.pth')
            
            # predicted text
            if p:
                mlflow.log_artifact('tmp/lm_predicted.txt')
                print('Predicted text following "料金が高い"\n'+p)

def main():
    fire.Fire(MedWeb)


if __name__ == '__main__':
    main()
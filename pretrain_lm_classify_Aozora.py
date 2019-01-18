# coding: utf-8
"""
Pretrain lm and classify the lines using Aozora Bunko data

First, you need to prepare the data using prepare_Aozora_data.py
After doing so, run this code to pretrain the lm and classify the lines.

For usage instructions execute the following lines:
>>> python pretrain_lm_classify_Aozora.py -- --help
>>> python pretrain_lm_classify_Aozora.py experiment -- --help

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

class Aozora():
    """
    Class for pretraining lm and classifiying lines using Aozora Bunko data
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
        name='Aozora-clas',
        dataset='Aozora' ,
        qrnn=True,
        bs=20, 
        bptt=70,
        pad_token=PAD_TOKEN_ID,
        fine_tune_lm=True ,
        lm_lr=1e-2,
        lm_drop_mult=0.5,
        use_pretrained_lm=True,
        clas_lr=1e-2,
        clas_drop_mult=0.1,
        num_train = 1000,
        use_lm=True,
        deterministic = True,
        experiment_name = 'Aozora_classification',
        ):

        # set parameters
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.dataset_dir = self.data_dir / self.dataset
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
        self.clas_lr = clas_lr
        self.clas_drop_mult = clas_drop_mult
        self.num_train = int(num_train)
        self.use_lm = use_lm
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
            learn.fit(4, slice(self.lm_lr/(10**2), self.lm_lr))
        
        else:
            print('Skipping fine tuning')

        # save fine tuned lm
        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save_encoder(self.lm_enc_finetuned)
        
        return learn


    def classify_multiclas(self, data_clas):
        "Create a classifier to predict the author of each line of text"
        
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
            drop_mult=self.clas_drop_mult
            )
        learn.model.reset()
        
        if self.use_lm:
            print('Loading language model')
            lm_enc = self.lm_enc_finetuned # if fine_tune_lm else lm_name
            learn.load_encoder(lm_enc)
        else:
            print('Training from scratch without language model')

        # train
        self.set_deterministic()
        
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, self.clas_lr, moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(self.clas_lr / (2.6 ** 4), self.clas_lr), moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(self.clas_lr / (2.6 ** 4), self.clas_lr), moms=(0.8, 0.7), wd=1e-7)

        learn.unfreeze()
        learn.fit_one_cycle(2, slice(self.clas_lr / (2.6 ** 4), self.clas_lr), moms=(0.8, 0.7), wd=1e-7)
        
        # results
        results={}
        results['accuracy'] = learn.validate()[1]
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
        y_pred = p.argmax(axis=1)


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


    def plot_confusion_matrix(self, cm, classes,
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


    def save_confusion_matrix(self,y_true,y_pred,class_names):
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)

        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix')
        plt.gcf().savefig(self.cm_name)
    

    def experiment(self):
        """
        Performs an experiment with provided parameters
        Steps
        1. Pretrain lm 
        2. Classify lines
        """

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            self.set_deterministic()
            
            ## Log our parameters into mlflow
            # args = Params(qrnn,bs,bptt,emb_sz, nh, nl,fine_tune_lm,lm_lr,lm_drop_mult,
            #             use_pretrained_lm,clas_lr,clas_drop_mult,use_lm,num_train,deterministic)
            for key, value in vars(self).items():
                if isinstance(value, (int, float)):
                    mlflow.log_param(key, value)
            
            ## create data set
            data_lm = TextLMDataBunch.load(path=Path(self.data_dir/self.dataset),cache_name=Path(f'tmp/lm/{self.num_train}'))
            data_clas = TextClasDataBunch.load(path=Path(self.data_dir/self.dataset),cache_name=Path(f'tmp/clas/{self.num_train}'))
            
            ## fine tune lm 
            if self.use_lm:
                lm=self.lm_fine_tuning(data_lm)

            ## create classifier
            y_true, y_pred = self.classify_multiclas(data_clas)

            ## Evaluation
            results = self.evaluation(y_true,y_pred)
            class_names = data_clas.valid_ds.y.classes
            self.save_confusion_matrix(y_true,y_pred,class_names)
            
            ## Log metrics
            for key, value in results.items():
                mlflow.log_metric(key, value)
                
            ## Save artifacts
            # pretrained lm
            mlflow.log_artifact(self.model_dir/f"{self.pretrained_fnames[0]}.pth")
            mlflow.log_artifact(self.model_dir/f"{self.pretrained_fnames[1]}.pkl")
            
            # fine tuned lm
            if self.fine_tune_lm & self.use_lm:
                mlflow.log_artifact(self.model_dir / f'{self.lm_enc_finetuned}.pth')
            
            # classifier
            mlflow.log_artifact(self.model_dir / f'{self.model_name}_{self.name}.pth')
            
            # confusion matrix
            mlflow.log_artifact(self.cm_name)

def main():
    fire.Fire(Aozora)


if __name__ == '__main__':
    main()
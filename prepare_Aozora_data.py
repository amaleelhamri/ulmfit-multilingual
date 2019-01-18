# coding: utf-8
"""
Prepare data for Aozora buko classification

First, you need to download the texts from Aozora Bunko.
To do so, clone the following awesome repository and run aozora_scrape.py
https://github.com/shibuiwilliam/aozora_classification
(As of Dec. 2018, the config file, target_author.csv has a minor bug.
Change http to https and everything will be fine)

After doing so, run this code to prepare data for lm pretraining
and the following classification.

For usage instructions execute the following lines:
>>> python prepare_Aozora_data.py -- --help
>>> python prepare_Aozora_data.py prepare -- --help

"""

import fire
from fastai.text import TextLMDataBunch, TextClasDataBunch, language_model_learner, text_classifier_learner, LanguageLearner
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists, get_sentencepiece
from fastai.text.transform import Vocab
from pathlib import Path
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import glob
import shutil
import dask.dataframe as dd

class PrepareAozora():
    """
    Class for preparing Aozora Bunko data
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
        frac_ds=1.0,
        spm_dir = 'data/wiki/ja/',
        num_trains = [1e2,1e3,1e4,1e5],
        dir_orig="../aozora_classification/aozora_data/",
        dir_dest="data/Aozora/csv/",
        ):

        self.data_dir = Path(data_dir)
        assert self.data_dir.name == 'data',    f'Error: Name of data directory should be data, not {self.data_dir.name}.'
        self.dataset = dataset
        self.dataset_dir = self.data_dir / self.dataset
        self.model_dir = Path(model_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)
        self.spm_dir = spm_dir
        self.num_trains = num_trains
        self.max_vocab = max_vocab

        self.dir_orig = dir_orig
        self.dir_dest = dir_dest

        print(f'Dataset: {dataset}. Language: {lang}.')


    def copy_csv(self):
        "copy csvs from original directory"

        Path(self.dir_dest).mkdir(parents=True, exist_ok=True)
        csvs = glob.glob(self.dir_orig+"*/csv/*.csv", recursive=True)

        for f in csvs:
            shutil.copy(f,self.dir_dest)


    def split_data(self):
        "collect lines from all authors and split them into train/valid/test"

        # ## read data
        ddf = dd.read_csv(self.dir_dest+'*.csv')
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


    def prepare_databunch(self):
        "prepare databunch for lm and classification using different sizes of train data"
        # here we're just loading the trained spm model
        sp = get_sentencepiece(self.spm_dir, None, 'wt-all', vocab_size=self.max_vocab)


        # load train, valid in df
        train_df = pd.read_csv(self.dataset_dir/'train.csv',header=None)
        valid_df = pd.read_csv(self.dataset_dir/'valid.csv',header=None)
        test_df = pd.read_csv(self.dataset_dir/'test.csv',header=None)

        # create databunch for different sizes of train data
        print("Creating databunch for different sizes of train data")
        for num_train in self.num_trains:
            num_train = int(num_train)
            print("Size: {}".format(num_train))
            
            # sample train data
            tmp_train_df = train_df.sample(n=num_train,random_state=42)
            
            # create data bunches
            tmp_lm = TextLMDataBunch.from_df(path=self.dataset_dir,train_df=tmp_train_df,valid_df=valid_df,test_df=test_df,**sp)
            tmp_clas = TextClasDataBunch.from_df(path=self.dataset_dir,train_df=tmp_train_df,valid_df=valid_df,test_df=test_df,**sp)
            
            # name of tmp cache
            tmp_cache_lm = Path(f'tmp/lm/{num_train}')
            tmp_cache_clas = Path(f'tmp/clas/{num_train}')
            
            # save
            tmp_lm.save(tmp_cache_lm)
            tmp_clas.save(tmp_cache_clas)
    
    
    def prepare(self):
        """
        Executes all the steps necessary to prepare data
        1. copy csv
        2. split data
        3. prepare databunch using different train size
        """
        
        print('Copying csv...')
        self.copy_csv()

        print('Splitting data...')
        self.split_data()

        print('Preparing databunch...')
        self.prepare_databunch()


def main():
    fire.Fire(PrepareAozora)


if __name__ == '__main__':
    main()
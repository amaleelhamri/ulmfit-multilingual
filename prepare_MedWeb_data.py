# coding: utf-8
"""
Prepare data for MedWeb classification

First, you need to download the data from MedWeb website.
To do so, fill in the application form follwing the links from below.
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

    Project website: NTCIR-13 MedWeb
    http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

Place the downloaded files in the following directory like this
data/MedWeb/
L NTCIR-13_MedWeb_ja_training.xlsx
L NTCIR-13_MedWeb_ja_test.xlsx

After doing so, run this code to prepare data for lm pretraining
and the following classification.
This code will create tokens and labels from the texts 
and saves them in files.

For usage instructions execute the following lines:
>>> python prepare_MedWeb_data.py -- --help
>>> python prepare_MedWeb_data.py prepare -- --help

"""

import fire
import numpy as np
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists, get_sentencepiece
from pathlib import Path
import pandas as pd

class PrepareMedWeb():
    """
    Class for preparing MedWeb data
    """
    def __init__(
        self, 
        data_dir='data',
        lang='ja' ,
        pretrain_name='wt-100', 
        model_dir='data/wiki/ja-100/models',
        fig_dir='data/wiki/ja-100/figs',
        max_vocab=8000,
        name='MedWeb-clas',
        dataset='MedWeb' ,
        spm_dir = 'data/wiki/ja/',
        ):

        self.data_dir = Path(data_dir)
        assert self.data_dir.name == 'data',    f'Error: Name of data directory should be data, not {self.data_dir.name}.'
        self.dataset = dataset
        self.dataset_dir = self.data_dir / self.dataset
        self.model_dir = Path(model_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(exist_ok=True)
        self.spm_dir = spm_dir
        self.max_vocab = max_vocab


        print(f'Dataset: {dataset}. Language: {lang}.')


    def get_text_label(self):
        """
        1. Read train and test files
        2. Reshape the data into text and labels
        """
        # read files
        trn = pd.read_excel(self.dataset_dir/'NTCIR-13_MedWeb_ja_training.xlsx',sheet_name='ja_train')
        tst = pd.read_excel(self.dataset_dir/'NTCIR-13_MedWeb_ja_test.xlsx',sheet_name='ja_test')

        # list of labels
        lbl_names = trn.columns[2:]

        # put labels into a list
        trn_lbl = []
        for i, row in trn.iterrows():
            trn_lbl.append(lbl_names[(row[2:]=='p')].tolist())

        tst_lbl = []
        for i, row in tst.iterrows():
            tst_lbl.append(lbl_names[(row[2:]=='p')].tolist())

        # reshape the data (rename tst to val because databunch expects val to exist)
        trn_df = pd.DataFrame({1:trn['Tweet'].values, 0:trn_lbl},columns=[0, 1])
        val_df = pd.DataFrame({1:tst['Tweet'].values, 0:tst_lbl},columns=[0, 1])

        return trn_df, val_df

    def tokenize(self, trn_df, val_df):
        "Tokenize the text"
        # here we're just loading the trained spm model
        sp = get_sentencepiece(self.spm_dir, None, 'wt-all', vocab_size=self.max_vocab)
        tok = sp['tokenizer']

        # tokenize
        trn_tok = np.asarray(tok.process_all(trn_df[1]))
        val_tok = np.asarray(tok.process_all(val_df[1]))
        
        return trn_tok, val_tok
    

    def save(self, trn_df, val_df, trn_tok, val_tok):
        "Save tokens and labels"

        # tokens
        np.save(self.dataset_dir/'trn_tok.npy', trn_tok)
        np.save(self.dataset_dir/'val_tok.npy', val_tok)

        # labels
        np.save(self.dataset_dir/'trn_lbl.npy', trn_df[0].values)
        np.save(self.dataset_dir/'val_lbl.npy', val_df[0].values)
    
    
    def prepare(self):
        """
        Executes all the steps necessary to prepare data
        1. read data from xlsx
        2. tokenize text
        3. save tokens and labels
        """
        
        print('Reading data...')
        trn_df, val_df = self.get_text_label()

        print('Tokenizing text...')
        trn_tok, val_tok = self.tokenize(trn_df, val_df)

        print('Saving tokens and labels...')
        self.save(trn_df, val_df, trn_tok, val_tok)


def main():
    fire.Fire(PrepareMedWeb)


if __name__ == '__main__':
    main()
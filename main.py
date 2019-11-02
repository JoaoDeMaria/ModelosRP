\'''
Project: Código principal
@author: João Victor
Created: 16/09/2019
'''

import time
import pandas as pd
import numpy as np

import utils
import Models
import re

from Perceptron import Perceptron 
from utils import label_encode, train_test, normalize, graphic_of_superficie, metrics
from collections import ChainMap

_bag_words_ = []
_list_series_ = []

def Main():
    global _bag_words_, _list_series_
   
    # Loading datas
    df_text = pd.read_csv("Data/chennai_reviews_edited.csv")
    df = pd.read_csv("Data/chennai_reviews_edited_100.csv")
    class_csv =  pd.read_csv("Data/chennai_reviews_edited.csv")
    #class_csv['Sentiment'] = pd.Series([class_encode(row, '3') for row in class_csv['Sentiment']])
    class_d = class_csv['Sentiment']
    df['Sentiment'] = class_d

    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # Pre-processing datas
    '''
    df = pd.read_csv("Data/chennai_reviews.csv")

    df.drop(
        ['Hotel_name','Review_Title','Rating_Percentage',
        'Unnamed: 5','Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'],
        axis=1, inplace=True
    )
    df.drop(2261, axis=0, inplace=True)
    df.to_csv('Data/chennai_reviews_edited.csv')
    '''
    '''
    stop_words = ['the', 'is', 'at', 'which', 'on', 'I', 'they','a']
    
    df['Review_Text'].apply(create_bag_word, stop_words=stop_words)
    _bag_words_ = sorted(_bag_words_)
    df['Review_Text'].apply(create_df)    

    new_df = pd.DataFrame(_list_series_)
    new_df.columns = _bag_words_
    new_df.to_csv('Data/chennai_reviews_edited_2.csv')
    '''
    '''
    count = 0
    for column in df.columns:
        if df[column].sum() <= 200:
            df.drop([column], axis=1,inplace=True)
            count +=1
    print(f'Foram deletas {count} colunas')
    df.to_csv('Data/chennai_reviews_edited_200.csv')
    '''
    '''perceptron = Perceptron()
    statistcs = metrics(perceptron, df, class_d, 20,100, 'error','p', True, 2)
    print('acc: {}, std: {}, var: {}'.format(statistcs[0],statistcs[1],statistcs[2]))
    '''
    '''
    train_x, test_x, train_d, test_d = train_test(df, class_d)
    nb.train(train_x, ['1','2','3'])
    print(nb.predict('bad hotel'))
    '''

    nb = Models.NaivyBayes()
    start = time.clock()
    statistic = nb.metrics(df, df_text, class_d, ['1','2','3'], 20, True, 2)
    print("acc: {}, std: {}, var: {}".format(statistic[0], statistic[1], statistic[2]))
    print("Durou {}s para realizar toda a operação".format(time.clock() - start))
    

def class_encode(row, column):
    if row == column:
        return 1 
    else:
        return 0

def create_bag_word(row, stop_words):
    list_words = clear_row(row)

    for word in list_words:
        if word not in stop_words and word not in _bag_words_:
            _bag_words_.append(word)   


def create_df(row):
    list_words = clear_row(row)
    series_row = []
    for word in _bag_words_:
        if word in list_words:
            series_row += [1]
        else:
            series_row += [0]

    _list_series_.append(pd.Series(data=series_row))

def clear_row(row):
    row = row.lower()
    row = re.sub('[!@#$%¨&*()*,.;~^}´`""\d]',' ',row)
    if '.' in row: 
        row = row.replace('.', '') 
    if ',' in row:
        row = row.replace(',', '')

    return row.split(" ")


if __name__ == "__main__":
    Main()
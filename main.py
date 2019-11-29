'''
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
    iris = pd.read_csv('Data/Iris_data.csv')
    class_d = iris['species'].apply(label_encode, collumn="setosa")
    
    iris.drop(['species'], axis = 1, inplace=True)
    iris = normalize(iris)
    rl = Models.RegressaoLogistica()
    rl.train(iris,class_d,50,0.1)
    for i in range(10):
        print(rl.predict(iris.iloc[100, :]))


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
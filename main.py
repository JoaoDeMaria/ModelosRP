'''
Project: Código principal
@author: João Victor
Created: 16/09/2019
'''

import time
import pandas as pd
import numpy as np
from Perceptron import Perceptron 
from utils import label_encode, train_test, normalize, metrics, graphic_of_superficie


def Main():
    # Pre-processing datas 
    df = pd.read_csv("Data/Iris_Data.csv")
    D = df.species.apply(label_encode, collumn = "setosa")
    df.drop(["species",'petal_width','petal_length'], axis = 1, inplace = True)
    df = normalize(df)

    #Artificial
    and_df = pd.read_csv('Data/AND.csv')
    and_class = and_df['class']
    and_df.drop(['Unnamed: 0','class'], axis = 1, inplace = True)

    or_df = pd.read_csv('Data/OR.csv')
    or_class = or_df['class']   
    or_df.drop(['Unnamed: 0','class'], axis = 1, inplace = True)


    # Split between train and test    
    train_x, _, train_d, _ = train_test(df, D)

    # Start Training process
    perceptron = Perceptron()
    perceptron.train(train_x,train_d,100,0.1,'erro')

    # Evaluating model
    statistics = metrics(perceptron, and_df, and_class, 20, 100, 'erro', True, 2)
    print('acc: {:.2}, std: {:.2}, var: {:.2}'.format(statistics[0], statistics[1], statistics[2]))

    #graphic_of_superficie(perceptron, or_df)
    
    print(perceptron.get_weights())
if __name__ == "__main__":
    Main()
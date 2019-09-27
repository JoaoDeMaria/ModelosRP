'''
Project: Perceptron Simples
Created: 15/04/2019
@author: João Victor
'''

from pandas.core.frame import DataFrame,Series
from rna import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Constant of expression w_new = w_old + e*N*x
N = 0.1

#Simple Perceptron
class Perceptron(object):
    __slots__ = ['vector_peso']

    def __init__(self):
        self.vector_peso = None

    def trainingModel(self,data,class_d, epocks = 100):
        self.vector_peso = np.random.random(1 + data.shape[1])

        for count in range(epocks):
            index_random = randomIndex(data,1)
            data = data.loc[index_random]
            class_d = class_d.loc[index_random]

            ee  = 0

            for index,row in data.iterrows():
                train_x = np.array([-1] + row.tolist())
                func_u = np.inner(train_x,self.vector_peso)
                class_y = 1 if func_u >= 0 else 0
                error = class_d[index] - class_y
                ee  = ee  +  abs(error)
                self.vector_peso = self.vector_peso + N * error * train_x

            if ee == 0:
                break

    def predict(self,vector_input):
        predict_list = []
        if type(vector_input) == list:
            vector = np.array([-1] + vector_input)
            func_u = np.inner(vector,self.vector_peso)
            return True if func_u >= 0 else False

        for index,row in vector_input.iterrows():
            vector = np.array([-1] + row.tolist())
            func_u = np.inner(vector,self.vector_peso)
            predict_list += [True if func_u >= 0 else False]

        return predict_list

    def getPesos(self):
        return self.vector_peso

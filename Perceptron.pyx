'''
Project: Perceptron Simples utilizando Cython
@author: João Victor
Created: 16/09/2019
'''

import numpy as np
cimport numpy as np

import math

cdef class Perceptron:
    cdef np.ndarray weigths
    cdef double rate_learning
    cdef object train_x
    cdef object train_d
    cdef list error 
    cdef str type_function

    def __init__(self):
        self.weigths = np.array([])
        self.rate_learning = 0.5
        self.train_x = None
        self.train_d = None
        self.error = []
        self.type_function = ''
    
    cpdef train(self,train_x, train_d, int epocks, double rate_learning_final, str rule, str type_function):
        # Initialize weigths
        self.weigths = np.random.random((1,train_x.shape[1] + 1))
        self.type_function = type_function

        cdef int epock
        for epock in range(epocks):
            self.error = []
            self.train_x = train_x.loc[np.random.choice(
                    train_x.index, train_x.shape[0], replace = False
            )] # Shuffle train x
            self.train_d = train_d.loc[self.train_x.index] # Shuffle train d
            
            # Otmizing rate learning
            self.rate_learning *= (self.rate_learning/rate_learning_final)**(epock/epocks)
            
            # Create a dataframe with x and d 
            dataset = train_x.join(train_d)

            # Activite function 
            dataset.apply(self.per_epock, axis = 1) 

            if len(self.error) > 0:
                if np.mean(self.error) == 0 and 'erro' in rule:
                    break
    
    cpdef per_epock(self, row):
        cdef np.ndarray x = row.values[:-1] 
        x = np.array(x.tolist() + [-1])
        cdef int d = row.values[-1]
        cdef double y_predicted = self.activate_function(x)
        cdef double erro = d - y_predicted
        self.error.append(erro)
        self.adjust_weigth(x, erro, y_predicted)

    cpdef activate_function(self, x):
        cdef double u = np.dot(x, self.weigths.T)  
        if 'b' in self.type_function:
            return 1 if u > 0 else 0
        elif 'l' in self.type_function:
            return 1/(1 + math.exp(-u))
        else:
            return (1-math.exp(-u))/(1+math.exp(-u))

    cpdef diff_function(self, y ):
        if 'b' in self.type_function:
            return 1
        elif 'l' in self.type_function:
            return y*(1-y)
        else:
            return 0.5*(1-(y**2))

    cpdef adjust_weigth(self, x,double error, y):
        self.weigths += self.rate_learning * x * self.diff_function(y) * error

    cpdef predict(self, x):
        x = np.array(x.tolist() + [-1])
        if 'b' in self.type_function:
            return self.activate_function(x)
        elif 'l' in self.type_function:
            return 1 if self.activate_function(x) >= 0.5 else 0
        else:
            return 1 if self.activate_function(x) >= 0 else 0
    
    cpdef get_weights(self):    
        return self.weigths
'''
Project: Perceptron Simples utilizando Cython
@author: João Victor
Created: 16/09/2019
'''

import numpy as np
cimport numpy as np


cdef class Perceptron:
    cdef np.ndarray weigths
    cdef double rate_learning
    cdef object train_x
    cdef object train_d
    cdef list error 

    def __init__(self):
        self.weigths = np.array([])
        self.rate_learning = 0.5
        self.train_x = None
        self.train_d = None
        self.error = []
    
    cpdef train(self,train_x, train_d, int epocks, double rate_learning_final, str rule):
        # Initialize weigths
        self.weigths = np.random.random((1,train_x.shape[1] + 1))
        
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

            if np.mean(self.error) == 0 and 'erro' in rule:
                break
    
    cpdef per_epock(self, row):
        cdef np.ndarray x = row.values[:-1] 
        x = np.array(x.tolist() + [-1])
        cdef int d = row.values[-1]
        cdef double y_predicted = self.activate_function(x)
        cdef double erro = d - y_predicted
        self.error.append(erro)
        self.adjust_weigth(x, erro)

    cpdef activate_function(self, x):
        cdef double u = np.dot(x, self.weigths.T)  
        return 1 if u > 0 else 0
    
    cpdef adjust_weigth(self, x,double error):
        self.weigths += self.rate_learning * x * error

    cpdef predict(self, x):
        x = np.array(x.tolist() + [-1])
        return self.activate_function(x)
    
    cpdef get_weights(self):
        return self.weigths
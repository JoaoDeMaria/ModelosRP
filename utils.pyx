'''
Project: Biblioteca para métodos auxiliares
@author: João Victor
Created: 16/09/2019
'''

import numpy as np
cimport numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

from threading import Event 
from queue import Queue
from collections import namedtuple
from Optimization import Worker


# Variáveis privadas
cdef list _HIT_RATE_LIST = []

# Métodos para avaliar o modelo
cpdef hold_out(args):
    # split between train and test
    train_x, test_x , train_d, test_d = train_test(args.data,args.class_d) 
    
    # Training model
    args.model.train(train_x,train_d,args.epocks,0.1,args.rule, args.type_function)
    
    # Test datas
    predicted_list = np.array([args.model.predict(row) for index,row in test_x.iterrows()])
    test_d.index = range(len(test_d))
    
    # Add hit rate
    cdef object hit_hate
    hit_hate = 1 - abs(test_d.values-predicted_list).mean()
    _HIT_RATE_LIST.append(hit_hate)


cpdef metrics(object model, object data, object class_d, 
    int realizations, int epocks, str rule, str type_function,with_threads, int num_threads):
    cdef object args = namedtuple('args', 'model data class_d epocks rule type_function')
    cdef object queue = Queue(maxsize=realizations+1)
    cdef object event = Event() 
    
    cdef int realization
    cdef list hit_rate_list 
    
    [queue.put(args(model, data, class_d, epocks, rule, type_function)) for i in range(realizations)]  
    
    if with_threads:
        # Initialize instances of the threads
        event.set()
        queue.put('Kill')
        
        workers = [Worker(target = hold_out, queue = queue, name = f'Worker{i}') for i in range(num_threads)]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    else:
        [hold_out(queue.get()) for realization in range(realizations)]

    return np.mean(_HIT_RATE_LIST), np.std(_HIT_RATE_LIST), np.var(_HIT_RATE_LIST)


# Métodos para dividir entre treino e teste
cpdef train_test(data, class_d):
    train_x, test_x = _split_data_(data)
    train_d = class_d.iloc[train_x.index]
    test_d = class_d.drop(train_d.index)
    return train_x,test_x,train_d,test_d

cdef _split_data_(data):
    index_random = _random_index_(data,0.8)
    return data.loc[index_random],data.drop(index_random)

cdef _random_index_(data,count):
    cdef int random_count = int(len(data) * count)
    return np.random.choice(data.index,random_count,replace = False)


# Métodos para tratar os dados 
cpdef label_encode( index, collumn = ""):
    if collumn in index:
        return 1
    else:
        return 0


cpdef normalize(data):
    for col in data.columns:
        min = np.min(data[col])
        max = np.max(data[col])
        data[col] = [(data.at[i,col] - min)/(max-min)
                        for i in range(len(data))]
    return data


# Métodos para plotar gráficos
cpdef graphic_of_superficie(object model, object data):
        
    df = pd.read_csv('Data/DatasetGraphicModel.csv')
    df.drop(['Unnamed: 0'], axis = 1, inplace = True)

    #--------Inicio---------
    df.apply(point, axis=1, model=model,color_one='#ff0000',color_two='#00ff00')
    data.apply(point, axis=1, model=model,color_one='#ffff00',color_two='#0000ff')
    #---------Fim-----------
    plt.show()

cpdef point(object index, object model, str color_one, str color_two):
    if model.predict(index) == 0:
        plt.scatter(index[0], index[1], c = color_one)
    else:
        plt.scatter(index[0], index[1], c = color_two)

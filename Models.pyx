'''
Project: KNN utilizando Cython
@author: João Victor
Created: 02/10/2019
'''

import numpy as np
cimport numpy as np 
import pandas as pd

from collections import Counter, namedtuple, defaultdict
from queue import Queue

from threading import Event 
from Optimization import Worker

from utils import train_test


cdef class KNN:
    cdef list _LIST_NN
    cdef object knn_tuple 
    # Variáveis privadas
    cdef list HIT_RATE_LIST 

    def __init__(self):
        self._LIST_NN = []
        self.knn_tuple = namedtuple('data', ('distance', 'class_i'))
        self.HIT_RATE_LIST = []

    cpdef predict(self, object data,object class_data, object x, int num_k):
        cdef list neighbor
        cdef list list_distance_sorted
        cdef list most_common
        cdef object counter
        self._LIST_NN = []

        # Juntando os atributos com os respectivos rótulos em um DataFrame só
        df = data.join(class_data)

        # Aplicando a função para calcular a distancia 
        df.apply(self.calc_distance, axis = 1, x=x) 
        
        # Ordenar a lista do menor para a maior distancia
        list_distance_sorted = sorted(self._LIST_NN) 
        
        # Pegar os rótulos dos vizinhos mais próximos, de acordo com o k
        neighbor = [nn.class_i for nn in list_distance_sorted[0:num_k]] 
        
        # Pegar o número de vezes que um rótulo aparece 
        counter = Counter(neighbor) 
        
        # Pega o rótulo que mais aparece
        most_common = counter.most_common(1)

        return most_common[0][0]



    cpdef calc_distance(self,index, x):
        x = np.array(x)
        # sqrt(sum((x-y)^2))
        distance = np.sqrt(np.sum((x - index[:-1])**2))
        class_i = index[-1]
        
        self._LIST_NN.append(self.knn_tuple(distance, class_i))


    # Métodos para avaliar o modelo
    cpdef hold_out(self, args):
        # split between train and test
        train_x, test_x , train_d, test_d = train_test(args.data,args.class_d) 
            
        # Test datas
        predicted_list = np.array([args.model.predict(train_x, train_d, row, args.num_k) for index,row in test_x.iterrows()])
        test_d.index = range(len(test_d))
        
        # Add hit rate
        cdef object hit_hate
        hit_hate = np.array(test_d.values==predicted_list).mean()
        self.HIT_RATE_LIST.append(hit_hate)


    cpdef metrics(self, object data, object class_d, int num_k,
        int realizations, with_threads, int num_threads):
        cdef object args = namedtuple('args', 'model data class_d num_k')
        cdef object queue = Queue(maxsize=realizations+1)
        cdef object event = Event() 
        
        cdef int realization
        cdef list hit_rate_list 
        
        [queue.put(args(data, class_d, num_k)) for i in range(realizations)]  
        
        if with_threads:
            # Initialize instances of the threads
            event.set()
            queue.put('Kill')
            
            workers = [Worker(target = self.hold_out, queue = queue, name = f'Worker{i}') for i in range(num_threads)]
            [worker.start() for worker in workers]
            [worker.join() for worker in workers]
        else:
            [self.hold_out(queue.get()) for realization in range(realizations)]

        return np.mean(self.HIT_RATE_LIST), np.std(self.HIT_RATE_LIST), np.var(self.HIT_RATE_LIST)


    cpdef grid_search(self, object data, object class_d,
        int realizations, with_threads, int num_threads, list num_param):
        cdef object hit_hate_dict = defaultdict()
        
        for num in num_param:
            hit_hate_dict[f'{num}'] = []  
        
        for realization in range(realizations):
            train_x, test_x , train_d, test_d = train_test(data,class_d) 
            
            for num in num_param:
                predicted_list = np.array([self.predict(train_x, train_d, row, num) for index,row in test_x.iterrows()])
                test_d.index = range(len(test_d))
                hit_hate_dict[f'{num}'] += [np.mean(predicted_list == test_d.values)]
        
        return [np.mean(hit_hate_dict[f'{num}']) for num in num_param]



cdef class NaivyBayes:
    cdef list list_p_x
    cdef list list_p_y
    cdef list vocabulary
    cdef list values_y
    cdef list HIT_RATE_LIST

    def __init__(self):
        self.list_p_x = []
        self.list_p_y = []
        self.vocabulary = []
        self.values_y = []
        self.HIT_RATE_LIST = []

    cpdef train(self, data, values_y):   
        cdef int i
        cdef list list_p_y 

        self.values_y = values_y

        for y in values_y:
            docs = data.loc[data['Sentiment'] == y]
            docs.drop(['Sentiment'], axis=1, inplace=True)

            p_y = docs.shape[0]/(data.shape[0] - 1)
            self.list_p_y += [p_y]

            list_p_y = []
            for word in data.columns:
                if word != 'Sentiment':
                    n_k = docs[word].sum()
                    list_p_y += [(n_k + 0.0001) / len(docs)]
                
            self.list_p_x += [list_p_y]

        data.drop(['Sentiment'], axis=1, inplace=True)
        self.vocabulary = list(data.columns)
       

    cpdef predict(self, doc):
        p_x_y = 1   
        cdef list_y = []
        
        for y in range(len(self.values_y)):
            sum_x_0 = 0
            sum_x_1 = 0
            for i in range(len(self.vocabulary)):
                if self.vocabulary[i] in doc[1] :
                    sum_x_1 += np.log(self.list_p_x[y][i]) 
                    
                else:
                    sum_x_0 += np.log(1 - self.list_p_x[y][i]) 
                    
            list_y += [np.log(self.list_p_y[y]) + sum_x_1 + sum_x_0]
        
        for i in range(len(list_y)):
            if list_y[i] == np.max(list_y):
                return self.values_y[i]  

        
    # Métodos para avaliar o modelo
    cpdef hold_out(self,args):
        # split between train and test
        train_x, test_x , train_d, test_d = train_test(args.data,args.class_d) 
        self.train(train_x,args.values_y)
        test_data = args.data_text.loc[test_x.index]
        # Test datas
        predicted_list = np.array([self.predict(row) for index,row in test_data.iterrows()])
        test_d.index = range(len(test_d))
        print(test_d)
        print(predicted_list)
        # Add hit rate
        cdef object hit_hate
        hit_hate = np.array(test_d.values==predicted_list).mean()
        print(hit_hate)
        self.HIT_RATE_LIST.append(hit_hate)


    cpdef metrics(self, object data,object data_text, object class_d, object values_y,
        int realizations, with_threads, int num_threads):
        cdef object args = namedtuple('args', ' data data_text class_d values_y')
        cdef object queue = Queue(maxsize=realizations+1)
        cdef object event = Event() 
        
        cdef int realization
        cdef list hit_rate_list 
        
        [queue.put(args(data,data_text, class_d, values_y)) for i in range(realizations)]  
        
        if with_threads:
            # Initialize instances of the threads
            event.set()
            queue.put('Kill')
            
            workers = [Worker(target = self.hold_out, queue = queue, name = f'Worker{i}') for i in range(num_threads)]
            [worker.start() for worker in workers]
            [worker.join() for worker in workers]
        else:
            [self.hold_out(queue.get()) for realization in range(realizations)]

        return np.mean(self.HIT_RATE_LIST), np.std(self.HIT_RATE_LIST), np.var(self.HIT_RATE_LIST)



cdef class RegressaoLogistica:
    cdef object weigths
    
    cpdef train(self, object train_data, object class_d, int epocks, double rate_learning_final):
        self.weigths = np.random.random((1,train_data.shape[1] + 1))
        cdef int epock
        cdef int i
        bias = pd.DataFrame(np.ones(train_data.shape[0]))
        train_data = train_data.join(bias)
        print(train_data)

        for epock in range(epocks):
            gradiente = np.zeros((1,train_data.shape[1]))
            train_data = train_data.loc[np.random.choice(
                    train_data.index, train_data.shape[0], replace = False
            )] # Shuffle train x
            class_d = class_d.loc[train_data.index] # Shuffle train d
            
            for i in range(train_data.shape[0]):
                x = train_data.iloc[i,:]
                u = np.dot(x, self.weigths.T)
                gradiente += (self.sigmoid(u) - class_d[i])*x
            self.weigths += rate_learning_final*gradiente

    
    cpdef sigmoid(self,u):
        return 1 / (1 + np.exp(-u))

    cpdef predict(self, data, limite = 0.5):
        data = np.array(list(data) + [1] )
        u = np.dot(data, self.weigths.T)
        print(u)
        y = u >= limite
        return (y.astype('int'))

#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

sr = 22050
step = 0.5        # times of window_size
window_size = 20       # ms

dirpath_x = 'models/modle1/data/x_input.dat'
dirpath_y = 'models/modle1/data/y_input.dat'
class DataGen:
    def __init__(self, dirpath_x, dirpath_y, batch_size, split=0.8):
        print('>>>>>>>>>> getting data from dirpath')       
        self.__step = 0  # steps per epoch
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self._batch_size = batch_size

        self._dirx = dirpath_x
        self._diry = dirpath_y

        self.__x_inputs = np.array(0)
        self.__y_inputs = np.array(0)
        self.__readmm()
        
        self.__split = math.floor(self.__y_inputs.shape[0]*split)
        self.__x__train = self.__x_inputs[:self.__split]
        self.__y__train = self.__y_inputs[:self.__split]
        self.__x__test = self.__x_inputs[self.__split:]
        self.__y_test = self.__y_inputs[self.__split:]

    def get_test_data(self):
        return self.__x__test, self.__y_test
    def getinfo_train(self):
        return self.__x__train.shape, self.__y__train.shape, sum(self.__y__train)
    def getinfo_test(self):
        return self.__x__test.shape, self.__y_test.shape, sum(self.__y_test)
    def steps(self):
        return self.__step

    def train_gen(self):
        i = 0
        while True:
            if (i + 1) * self._batch_size > self.__x__train.shape[0]:
                # return rest and then switch files
                x, y = self.__x__train[i * self._batch_size:], self.__y__train[i * self._batch_size:]
                i = 0
            else:
                x, y = self.__x__train[i * self._batch_size:(i + 1) * self._batch_size],\
                        self.__y__train[i * self._batch_size:(i + 1) * self._batch_size]
                i += 1
            yield x, y

    def test_gen(self):
        i = 0
        while True:
            if (i + 1) * self._batch_size > self.__x__train.shape[0]:
                # return rest and then switch files
                x, y = self.__x__test[i * self._batch_size:], self.__y_test[i * self._batch_size:]
                i = 0
            else:
                x, y = self.__x__test[i * self._batch_size:(i + 1) * self._batch_size],\
                        self.__y_test[i * self._batch_size:(i + 1) * self._batch_size]
                i += 1
            yield x, y

    def __readmm(self):
        mmx = np.memmap(self._dirx, mode='r', dtype=float)
        self.__x_inputs = mmx.reshape(-1, self.__window_size)
        mmy = np.memmap(self._diry, mode='r')
        self.__y_inputs = mmy
        assert(self.__x_inputs.shape[0]==self.__y_inputs.shape[0])
        del mmx
        del mmy

if __name__ == '__main__':
    # dataGen = DataGen(dirpath, 12, 1)
    print('dodododdo')
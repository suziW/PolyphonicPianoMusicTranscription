#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import glob


dirpath_x = 'models/modle1/data/x*.dat'
dirpath_y = 'models/modle1/data/y*.dat'
class DataGen:
    def __init__(self, dirpath_x, dirpath_y, args):
        print('>>>>>>>>>> getting data from dirpath')       
        self.__step = 0  # steps per epoch
        self._n_bins = args['n_bins']
        self._window_size = args['window_size']
        self._batch_size = args['batch_size']

        self._dirx = []
        self._diry = []
        for file in glob.glob(dirpath_x):
            self._dirx.append(file)
        for file in glob.glob(dirpath_y):
            self._diry.append(file)
        self._fileCnt = len(self._dirx)
        assert(len(self._dirx) == len(self._diry))

        self.__x_inputs, self.__y_inputs = self.__readmm()
        self.__x_inputs = np.concatenate(self.__x_inputs)
        self.__y_inputs = np.concatenate(self.__y_inputs)

    def get_xshape(self):
        return self.__x_inputs.shape
    def get_yshape(self):
        return self.__y_inputs.shape
    def steps(self):
        return self.__step

    def next(self):
        i = 0
        while True:
            if (i + 1) * self._batch_size > self.__x_inputs.shape[0]:
                # return rest and then switch files
                x, y = self.__x_inputs[i * self._batch_size:], self.__y_inputs[i * self._batch_size:]
                i = 0
            else:
                x, y = self.__x_inputs[i * self._batch_size:(i + 1) * self._batch_size],\
                        self.__y_inputs[i * self._batch_size:(i + 1) * self._batch_size]
                i += 1
            yield x, y

    def __readmm(self):
        x_input = []
        y_input = []
        for file in self._dirx:
            mmi = np.memmap(file, mode='r')
            x = mmi.reshape(-1, self._window_size, self._n_bins)
            x_input.append(x)
            del mmi
        for file in self._diry:
            mmi = np.memmap(file, mode='r')
            y = mmi.reshape(-1, 88)
            y_input.append(y)
            del mmi
        return x_input, y_input

if __name__ == '__main__':
    # dataGen = DataGen(dirpath, 12, 1)
    print('dodododdo')
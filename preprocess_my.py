#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import pretty_midi
import librosa
import librosa.display
import math


input_dir = '../mozart/'
output_dir = 'model/'

sr = 22050
step = 0.5        # times of window_size
window_size = 20       # ms
min_midi = 21
max_midi = 108
midinote = 60       # middle C

class Preprocess:
    def __init__(self, input_dir, output_dir):
        self.__msdelta = 1000/sr
        self.__framepms = int(sr//1000)
        self.__window_size = window_size*self.__framepms
        self.__step = math.floor(self.__window_size*step)
        self.__input_dir = input_dir
        self.__output_dir = output_dir

        self.__wavfiles = []
        self.__midfiles = []
        self.__input_num = 0
        self.__get_file()

        self.__y_input = []
        self.__align_list = []
        self.__x_input = []
        self.__midfile2np()
        self.__wavfile2np()
        print(len(self.__x_input), len(self.__y_input))

        self.__save()
    
    def __save(self):
        indices = np.random.permutation(len(self.__x_input))
        self.__x_input = np.array(self.__x_input)
        self.__y_input = np.array(self.__y_input)
        self.__x_input = self.__x_input[indices]
        self.__y_input = self.__y_input[indices]
        mmx = np.memmap(filename=self.__output_dir+'x_input.dat', mode='w+', shape=self.__x_input.shape)
        mmx[:] = self.__x_input[:]
        mmy = np.memmap(filename=self.__output_dir+'y_input.dat', mode='w+', shape=self.__y_input.shape)
        mmy[:] = self.__y_input[:]
        print(mmx.shape, mmy.shape)
        del mmx, mmy


    def __midfile2np(self):
        for file in self.__midfiles:
            midobj = pretty_midi.PrettyMIDI(file)     # loadfile
            # endtime = midobj.get_end_time()
            # print(endtime)
            mid = midobj.get_piano_roll(fs=sr)[midinote]
            mid[mid > 0] = 1
            print('>>>>>>>>>>> mid:', file, mid.shape)
            for i in np.arange(0, len(mid)-self.__window_size+1, self.__step):
                self.__y_input.append(mid[i+int(window_size//2)])
            self.__align_list.append(len(self.__y_input))
            # break
            

    def __wavfile2np(self):
        alignIndex = 0
        for file in self.__wavfiles:
            wav, _ = librosa.load(file, sr)
            print('>>>>>>>>>> wav: ', file, wav.shape)
            for i in np.arange(0, len(wav)-self.__window_size+1, self.__step):
                self.__x_input.append(wav[i:i+self.__window_size])
            self.__x_input = self.__x_input[:self.__align_list[alignIndex]]
            alignIndex += 1
            # break
            
        
    def __get_file(self):
        for wavfile in glob.glob(self.__input_dir+'*.wav'):
            self.__wavfiles.append(wavfile)
        for midfile in glob.glob(self.__input_dir+'*.mid'):
            self.__midfiles.append(midfile)
        self.__wavfiles.sort()
        self.__midfiles.sort()
        self.__input_num = len(self.__wavfiles)
        for i in range(len(self.__wavfiles)):
            assert(os.path.splitext(self.__wavfiles[i])[0] == os.path.splitext(self.__midfiles[i])[0])  

    def get_param(self):
        return {'input_num': self.__input_num, 'window_size': self.__window_size, 
                'step': self.__step, 'frame/ms': self.__framepms, 'x_input': self.__x_input,
                'y_input': self.__y_input}

if __name__=='__main__':
    pre = Preprocess(input_dir, output_dir)
    param = pre.get_param()
     

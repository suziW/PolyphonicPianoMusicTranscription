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


input_dir = 'data/Classical_Piano_piano-midi.de_MIDIRip/mozart/'
output_dir = 'models/modle1/data/'

sr = 22050
step = 0.5        # times of window_size
window_size = 20       # ms
min_midi = 21
max_midi = 108
midinote = 60       # middle C
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple

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

        self.__x_input = []
        self.__y_input = []
        self.__wavfile2np()
        self.__midfile2np()
    
    def __save(self):
        self.__x_input = np.array(self.__x_input)
        self.__y_input = np.array(self.__y_input)
        mmx = np.memmap(filename=self.__output_dir+'x_input.dat', mode='w+', shape=self.__x_input.shape)
        mmx[:] = self.__x_input[:]
        mmy = np.memmap(filename=self.__output_dir+'y_input.dat', mode='w+', shape=self.__y_input.shape)
        mmy[:] = self.__y_input[:]
        del mmx, mmy


    def __midfile2np(self):
        for file in self.__midfiles:
            midobj = pretty_midi.PrettyMIDI(file)     # loadfile
            mid = midobj.get_piano_roll(fs=sr)[midinote] 
            print('>>>>>>>>>>> mid:', mid.shape)
            for i in np.arange(0, len(mid)-self.__window_size+1, self.__step):
                self.__y_input.append(mid[i+int(i+window_size//2)])
            break
            

    def __wavfile2np(self):
        for file in self.__wavfiles:
            wav, _ = librosa.load(file, sr)
            print('>>>>>>>>>> wav: ', len(wav))
            for i in np.arange(0, len(wav)-self.__window_size+1, self.__step):
                self.__x_input.append(wav[i:i+self.__window_size])
                # plt.figure()
                # plt.plot(wav[i:i+self.__window_size])
                # plt.show()
                break
            break
            
        
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
                'step': self.__step}

if __name__=='__main__':
    pre = Preprocess(input_dir, output_dir)
    print(pre.get_param())
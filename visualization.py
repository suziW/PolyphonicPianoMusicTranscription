#!/home/suzi/anaconda3/bin/python3.6
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import math
from collections import defaultdict

bin_multiple = 4
window_size = 7
note_range = 88
n_bins = bin_multiple * note_range
threshhole = 0.5
begin = 0
frames = 320    # default about 8s

mmy_groundtruth = np.memmap('models/new/0.00043/y_groundtruth.dat', mode='r')
y_groundtruth = np.reshape(mmy_groundtruth, (-1, note_range))
mmy_score = np.memmap('models/new/0.00043/y_score.dat', mode='c', dtype=float)
y_score = np.reshape(mmy_score, (-1, note_range))

print('>>>>>>>>>>>> shapes: ', y_groundtruth.shape, y_score.shape)
assert(y_groundtruth.shape == y_score.shape)
# y_score[y_score>threshhole] = 1
# y_score[y_score<threshhole] = 0

# figure = plt.figure()
# plt.imshow(y_groundtruth[begin:begin+frames, :].T)

# figure = plt.figure()
# plt.imshow(y_score[begin:begin+frames, :].T)
# plt.show()


class Eval:
    """ prediction and groundtruth should be shape (frame, one_hot_notes)
        discard: the time(ms) threshhole of notes u wanna discard detected frome prediction
        threshhole: the threshhole u convert prediction possibilites to one_hot code
        sr: sampling rate of .wav file
        hop_length: hop_length of CQT 
        onset_tolerance: time(ms) u tolerate the difference between prediction note onset and 
                            groundtruth note onst
        offset_tolerance: same as above discribe
    """
    def __init__(self, prediction, groundtruth, discard=50, threshhole=0.5,
                sr=22050, hop_length=512, onset_tolerance=100, offset_tolerance=100):
        # self._prediction = prediction
        # self._groundtruth = groundtruth
        self.__note_range = 88
        assert(prediction.shape == groundtruth.shape)
        self.__shape = groundtruth.shape
        self.__mspframe = 1000/sr*hop_length    # defaut is 23ms
        self.__discard = math.floor(discard/self.__mspframe)
        self.__discarded = defaultdict(list)
        self.__onset_tolerance = math.floor(onset_tolerance/self.__mspframe)
        self.__offset_tolerance = math.floor(offset_tolerance/self.__mspframe)
        self.__threshhole = threshhole

        self.__prediction_onehot = self.__prob2onehot(prediction)
        self.__groundtruth_onehot = groundtruth
        self.__prediction_note = self.__onehot2note(self.__prediction_onehot)
        # for note, fragments in self.__prediction_note.items():
        #     print(note, len(fragments))
        self.__adjust()
        # for note, fragments in self.__prediction_note.items():
        #     print(note, len(fragments))
        self.__groundtruth_note = self.__onehot2note(self.__groundtruth_onehot)
        # print(self.__prediction_note.items())
        # print('--------------------------')
        # print(self.__groundtruth_note.items())

        self.__note_metric_info = {'ptotal':0, 'rtotal':0, 'perror':0, 'rerror':0}
        self.__perror_note = defaultdict(list)
        self.__rerror_note = defaultdict(list)
        self.__note_precision = 0
        self.__note_recall = 0
        self.__note_fmeasure = 0
        
    def __isfragmentin(self, note, fragment, obj):
        temp = False
        for i in obj[note]:
            if ((i[0]-self.__onset_tolerance)<=fragment[0]) & ((i[1]+self.__offset_tolerance)>=fragment[1]):
                temp = True
        return temp

    def __adjust(self):
        for note, fragments in self.__prediction_note.items():
            for fragment in fragments:
                if not (fragment[1]-fragment[0]) > self.__discard:
                    self.__discarded[note].append(fragment) 
            for fragment in self.__discarded[note]:
                fragments.remove(fragment)

    def get_param(self):
        return {}

    def __onehot2note(self, one_hot):
        temp = np.zeros(self.__note_range, dtype=int)
        onset = np.zeros(self.__note_range, dtype=int)
        notes = defaultdict(list)
        for frame in range(self.__shape[0]):
            for note in range(self.__shape[1]):
                if (temp[note], one_hot[frame, note]) == (0, 0): # offset over and onset not detected
                    pass                                          # normal state, no action
                elif (temp[note], one_hot[frame, note]) == (0, 1): # onset detected
                    temp[note] = 1                                  # record state, this note pressed
                    onset[note] = frame                              # record the frame when note pressed
                elif (temp[note], one_hot[frame, note]) == (1, 1): # onset over and offset not detected
                    pass                                          # normal state, note pressed
                else:                                            # offset detected
                    temp[note] = 0                                 # reset temp which mark notepressed
                    notes[note].append((onset[note], frame-1))   # append tuple(onset, offset)
        return notes

    def __prob2onehot(self, prediction):
        temp = np.zeros(self.__shape)
        temp[prediction>=self.__threshhole] = 1
        temp[prediction<self.__threshhole] = 0
        return temp

    def plot(self, begin, end, mode='frame'):  # mode: time(s) or frame  
        assert(mode in ['time', 'frame'])
        if mode=='time':
            begin = math.floor(1000*begin/self.__mspframe)
            end = math.floor(1000*end/self.__mspframe)
        plt.figure()
        plt.imshow(self.__groundtruth_onehot.T[:, begin:end])
        plt.title('groundtruth')
        plt.figure()
        plt.imshow(self.__prediction_onehot.T[:, begin:end])
        plt.title('prediction')
        plt.show()


    def frameP(self):       # how many predictions are true 
        cnt = 0
        for frame in range(self.__shape[0]):
            if (self.__prediction_onehot[frame]==self.__groundtruth_onehot[frame]).all():
                cnt += 1
        return cnt/self.__shape[0]
        
    def frameR(self):       # how many groundtruth are predicted right
        return self.frameP()
    def frameF(self):
        return self.frameP()
    def noteP(self):
        for note, fragments in self.__prediction_note.items():
            self.__note_metric_info['ptotal'] += len(fragments)
            for fragment in fragments:
                if not self.__isfragmentin(note, fragment, self.__groundtruth_note):
                    self.__note_metric_info['perror'] +=1
                    self.__perror_note[note].append(fragment)
        self.__note_precision = 1 - self.__note_metric_info['perror']/self.__note_metric_info['ptotal']
        return self.__note_precision, self.__perror_note
        
    def noteR(self):
        for note, fragments in self.__groundtruth_note.items():
            self.__note_metric_info['rtotal'] += len(fragments)
            for fragment in fragments:
                if not self.__isfragmentin(note, fragment, self.__prediction_note):
                    self.__note_metric_info['rerror'] +=1
                    self.__rerror_note[note].append(fragment)
        self.__note_recall = 1 - self.__note_metric_info['rerror']/self.__note_metric_info['rtotal']
        return self.__note_recall, self.__rerror_note

    def noteF(self):
        assert((self.__note_precision, self.__note_recall) != (0, 0))
        self.__note_fmeasure = (2*self.__note_precision*self.__note_recall) / (self.__note_precision+self.__note_recall)
        return self.__note_fmeasure

if __name__=='__main__':
    print('-------', id(y_score))
    evaluation = Eval(y_score, y_groundtruth, offset_tolerance=100, onset_tolerance=100)
    print(evaluation.frameP())
    print(evaluation.noteP())
    print(evaluation.noteR())
    print(evaluation.noteF())
    evaluation.plot(begin, begin+frames)
    
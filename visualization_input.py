#!/home/suzi/anaconda3/bin/python3.6
# -*- coding: utf-8 -*-

import librosa.display
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import math
from skimage import transform


sr = 22050
hop_length = 512
window_size = 7
min_midi = 21
max_midi = 108
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple

class Visualization:
    def __init__(self, wavdir, middir):
        self.__mspframe = 1000/sr*hop_length
        self.wavdir = wavdir
        self.middir = middir
        self.cqt = 0 
        self.x_input = 0
        self.y_input = 0
        self.wav2inputnp()
        self.mid2inputnp()


    def wav2inputnp(self):
        print(">>>>>>>>>> in wav2inputnp")

        # down-sample,mono-channel
        y, _ = librosa.load(self.wavdir, sr)
        self.cqt = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=3)
        # 可视化
        # plt.figure()
        # librosa.display.specshow(S, sr=sr, fmin=librosa.midi_to_hz(min_midi),
                            #   fmax=librosa.midi_to_hz(max_midi), y_axis='linear', x_axis='time', )
        # plt.figure()
        # plt.imshow(abs(S))
        # plt.show()
        S = self.cqt.T  
        # S上下（0轴）填充0.5 window size的 minDB值
        S = np.abs(S)
        minDB = np.min(S)
        S = np.pad(S, ((window_size // 2, window_size // 2), (0, 0)), 'constant', constant_values=minDB)

        # slice by window size 
        # IMPORTANT NOTE:
        # Since we pad the the spectrogram frame,
        # the onset frames are actually `offset` frames.
        # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
        # starting at frame 0 of the padded spectrogram
        windows = []
        for i in range(S.shape[0] - window_size + 1):
            w = S[i:i + window_size, :]
            windows.append(w)
        self.x_input = np.array(windows)

    def mid2inputnp(self):
        midiData = pretty_midi.PrettyMIDI(self.middir)     # loadfile
        times = librosa.frames_to_time(np.arange(self.x_input.shape[0]), sr=sr, hop_length=hop_length)
        # print(midiData.get_onsets())
        self.y_input = midiData.get_piano_roll(fs=sr, times=times)[min_midi:max_midi + 1]
        self.y_input[self.y_input > 0] = 1      # piano roll has got numbers more than 1
    
    def plot(self, begin, end, mode='frame'):  # mode: time(s) or frame  
        assert(mode in ['time', 'frame'])
        if mode=='time':
            begin = math.floor(1000*begin/self.__mspframe)
            end = math.floor(1000*end/self.__mspframe)
        plt.figure()
        plt.imshow(abs(self.cqt[:, begin:end]))
        plt.title('cqt')
        plt.figure()
        y_input_resize = transform.resize(self.y_input, (self.cqt.shape[0], self.y_input.output_shape[1]))
        plt.imshow(y_input_resize[:, begin:end])
        plt.title('groundtrth')
        plt.show()        

if __name__=='__main__':
    wavdir = 'data/Classical_Piano_piano-midi.de_MIDIRip/mozart/mz_545_2.wav'
    middir = 'data/Classical_Piano_piano-midi.de_MIDIRip/mozart/mz_545_2.mid'
    begin = 0
    frames = 320    # about 8s
    end = begin + frames
    visual = Visualization(wavdir, middir)
    visual.plot(begin, end)

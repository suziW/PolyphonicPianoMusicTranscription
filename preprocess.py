#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import librosa.display
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import pretty_midi
from skimage import transform


input_dir = 'data/Classical_Piano_piano-midi.de_MIDIRip/mozart/*.wav'
output_dir = 'data/Classical_Piano_piano-midi.de_MIDIRip/models/modle1/data'
output_name_x = 'x_train.dat'
output_name_y = 'y_train.dat'

sr = 22050
hop_length = 512
window_size = 7
min_midi = 21
max_midi = 108
bin_multiple = 3
bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
n_bins = (max_midi - min_midi + 1) * bin_multiple


def wav2inputnp(audio_fn):
    print(">>>>>>>>>> in wav2inputnp")

    # down-sample,mono-channel
    y, _ = librosa.load(audio_fn, sr)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins, filter_scale=3)
    # 可视化
    # plt.figure()
    # librosa.display.specshow(S, sr=sr, fmin=librosa.midi_to_hz(min_midi),
                        #   fmax=librosa.midi_to_hz(max_midi), y_axis='linear', x_axis='time', )
    # plt.figure()
    # plt.imshow(abs(S))
    # plt.show()
    S = S.T  

    # TODO: LogScaleSpectrogram?
    '''
    if spec_type == 'cqt':
        #down-sample,mono-channel
        y,_ = librosa.load(audio_fn,sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                          bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T
    else:
        #down-sample,mono-channel
        y = madmom.audio.signal.Signal(audio_fn, sample_rate=sr, num_channels=1)
        S = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(y,fmin=librosa.midi_to_hz(min_midi),
                                            hop_size=hop_length, num_bands=bins_per_octave, fft_size=4096)'''

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
    x = np.array(windows)
    return x


def preprocese():
    print('>>>>>>>>>> in preprocese')
    x_list = []
    y_list = []
    frame_list = []
    fileCnt = 0
    for wavFile in glob.iglob(input_dir):
        print('---------------------{}-----------------------'.format(wavFile))
        fileName, _ = os.path.splitext(wavFile)
        midFile = fileName + '.mid'
        if not os.path.exists(midFile):
            print('!!!!!!!!!!!!!!error, midfile <<{}>> no exist:'.format(midFile))
            break

        # get x y from name.file
        x = wav2inputnp(wavFile)

        midiData = pretty_midi.PrettyMIDI(midFile)     # loadfile
        times = librosa.frames_to_time(np.arange(x.shape[0]), sr=sr, hop_length=hop_length)
        # print(midiData.get_onsets())
        y = midiData.get_piano_roll(fs=sr, times=times)[min_midi:max_midi + 1].T
        y[y > 0] = 1      # piano roll has got numbers more than 1
        print('x shape: {}, y shape: {}'.format(x.shape, y.shape))

        # ground truth visualization
        # print('>>>>>>>>>>>>>>>>>>>>>')
        # print(piano_roll.shape)
        # plt.figure()
        # piano_roll_resize = transform.resize(piano_roll.T, (880, piano_roll.shape[0]))
        # plt.imshow(piano_roll_resize[:, :2000])
        # plt.show()

        # check shape and add data to list
        if not x.shape[0]==y.shape[0]:
            print('!!!!!!!!!!!!!!error, midfile <<{}>> no match:'.format(midFile))
            break
        fileCnt += 1
        frame_list.append(x.shape[0])
        x_list.append(x)
        y_list.append(y)

    print('>>>>>>>>>>> file read and transform finished')
    print('{} examples in dataset'.format(fileCnt))
    print('total frames: ', np.sum(frame_list))
    # convert list to array and save to .dat file
    x_input = np.concatenate(x_list)
    y_input = np.concatenate(y_list)
    print('x_input shape: {}, y_input shape: {}'.format(x_input.shape, y_input.shape))
    mmx = np.memmap(filename=os.path.join(output_dir, output_name_x), mode='w+', shape=x_input.shape)
    mmx[:] = x_input[:]
    mmy = np.memmap(filename=os.path.join(output_dir, output_name_y), mode='w+', shape=y_input.shape)
    mmy[:] = y_input[:]
    del mmx
    del mmy
        


        

if __name__ == '__main__':
    preprocese()
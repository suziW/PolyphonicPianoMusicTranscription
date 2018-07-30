#!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import load
import preprocess
import numpy as np
import os

#keras utils
from keras.callbacks import Callback
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model

dirpath_x = '/data'
dirpath_y = 'data'
modelpng_path = '.'
class Train:
    def __init__(self, args, model_dir):
        self._model_dir = model_dir
        self.__batch_size = args['batch_size']
        self.__epochs = args['epochs']
        self._args = args
    def train(self):
        trainGen = load.DataGen(dirpath_x, dirpath_y, self._args)
        valGen = load.DataGen(dirpath_x, dirpath_y, self._args)

        if os.path.isfile(self._model_dir):
            print('loading model')
            model = load_model(self._model_dir)
        else:
            print('training new model from scratch')
            model = resnet_model()

        init_lr = float(args['init_lr'])

        model.compile(loss='binary_crossentropy',
                optimizer=SGD(lr=init_lr,momentum=0.9))
        model.summary()
        plot_model(model, to_file=os.path.join(path,'model.png'))

        checkpoint = ModelCheckpoint(model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(patience=1,monitor='val_loss', verbose=1, mode='min')
        #tensorboard = TensorBoard(log_dir='./logs/baseline/', histogram_freq=250, batch_size=batch_size)
        if args['lr_decay'] == 'linear':
            decay = linear_decay(init_lr,epochs)
        else:
            decay = half_decay(init_lr,5)
        csv_logger = CSVLogger(os.path.join(path,'training.log'))
        #t = Threshold(valData)
        callbacks = [checkpoint,early_stop,decay,csv_logger]

        history = model.fit_generator(trainGen.next(),trainGen.steps(), epochs=epochs,
                verbose=1,validation_data=valGen.next(),validation_steps=valGen.steps(),callbacks=callbacks)

        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('baseline/loss.png')


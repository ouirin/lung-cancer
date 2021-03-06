import settings
import helpers
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from typing import List, Tuple
from keras.optimizers import Adam, SGD , RMSprop , Adagrad , Adadelta , Adam , Adamax , Nadam
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil

import keras

from keras.layers import Add

from keras import backend as K

from keras.layers import multiply

from keras.layers.core import Permute

from keras.layers.core import Reshape

from keras.layers import GRU,Reshape

from keras.layers import TimeDistributed

from keras.layers.wrappers import Bidirectional

from keras.utils.vis_utils  import plot_model

# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
# POS_IMG_DIR = "luna16_train_cubes_pos"
LEARN_RATE = 0.001

USE_DROPOUT = False

if __name__ == "__main__":
    
    if True:
            
        train(model_name="luna16_full_RNN", train_full_set=True, load_weights_path=None)      
        
        if not os.path.exists("models/"):
            os.mkdir("models")
            
        shutil.copy("workdir/model_luna16_full_RNN_best.hd5", "models/model_luna16_full_RNN_best.hd5")
		

def train(model_name, train_full_set, load_weights_path):
    
    batch_size = 16
    
    #?????????????????????????????????????????????class label???size label???????????????
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, full_luna_set=train_full_set)
    
    #???????????????
    train_gen = data_generator(batch_size, train_files, train_set=True)
    
    #print(train_gen.__next__()[1])
    #sys.exit(1)
    
    #???????????????
    holdout_gen = data_generator(batch_size, holdout_files, train_set=False)
   
    #?????????????????????
    learnrate_scheduler = LearningRateScheduler(step_decay)
    
    #??????model
    model = get_net(load_weight_path=load_weights_path)
    
    
    checkpoint = ModelCheckpoint("workdir/model_" + model_name + "_"  + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
   
    checkpoint_fixed_name = ModelCheckpoint("workdir/model_" + model_name + "_"  + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    model.fit_generator(train_gen, len(train_files) / 1, 10, validation_data=holdout_gen, nb_val_samples=len(holdout_files) / 1, callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])
 
    model.save("workdir/model_" + model_name + "_"  + "_end.hd5")
	
	
def step_decay(epoch):
    
    res = 0.001
 
    if epoch > 5:
        res = 0.0001
        
    print("learnrate: ", res, " epoch: ", epoch)
    return res
	
	
def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None) -> Model:
        
    input_shapeT1=(8,32)  
    input1 = Input( shape=input_shapeT1 ) 
    gru1 = GRU(128,activation='tanh', recurrent_activation='sigmoid',return_sequences=True)(input1)
    gru1 = attention_3d_block(gru1)
    gru1 = Flatten()(gru1)
    Encoder1 = Model(input1, gru1)

    input_shapeT2=(8,8,32)
    input2 = Input( shape=input_shapeT2 )
    embed2 = TimeDistributed(Encoder1)(input2)
    gru2 = GRU(256,activation='tanh', recurrent_activation='sigmoid',return_sequences=True)(embed2)
    gru2 = attention_3d_block(gru2)
    gru2 = Flatten()(gru2)
    Encoder2 = Model(input2, gru2)
    

    input_shapeT3=(8,8,8,32) 
    input3 = Input( shape=input_shapeT3 )
    embed3 = TimeDistributed(Encoder2)(input3)
    gru3 = GRU(512,activation='tanh', recurrent_activation='sigmoid',return_sequences=True)(embed3)
    gru3 = attention_3d_block(gru3)
    gru3 = Flatten()(gru3)
   
    
    gru3 = Dense(512, activation='relu')(gru3)
    
    out_class = Dense(1, activation='sigmoid', name='out_class')(gru3)
    model = Model(input=input3, output=out_class)
  
    
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)

    model.compile(optimizer=Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True), loss={"out_class": "binary_crossentropy"}, metrics={"out_class": [binary_accuracy, binary_crossentropy] })
    model.summary(line_length=140)
    
    return model
	

def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(8, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name=???attention_mul???, mode=???mul???)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul
	
	
def data_generator(batch_size, record_list, train_set):
    
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    
    while True:
        
        img_list = []
        class_list = []
        size_list = []
        
        if train_set:
            random.shuffle(record_list)
            
        CROP_SIZE = CUBE_SIZE
        
        #????????????????????????
        for record_idx, record_item in enumerate(record_list):
            
            class_label = record_item[1]
            size_label = record_item[2]              #??????????????????????????????????????????????????????????????????????????????
            
            #??????negative cube
            if class_label == 0:
                
                cube_image = helpers.load_cube_img(record_item[0], 6, 8, 48)
              
                wiggle = 48 - CROP_SIZE - 1
                indent_x = 0
                indent_y = 0
                indent_z = 0
                
                if wiggle > 0:
                    indent_x = random.randint(0, wiggle)
                    indent_y = random.randint(0, wiggle)
                    indent_z = random.randint(0, wiggle)
                
                #?????????crop_size?????????cube
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
                
                #????????????
                if train_set:   
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]

                if CROP_SIZE != CUBE_SIZE:
                    
                    cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                    
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
                
            #??????positive cube
            else:
                cube_image = helpers.load_cube_img(record_item[0], 8, 8, 64)

                if train_set:
                    pass

                current_cube_size = cube_image.shape[0]
                
                indent_x = (current_cube_size - CROP_SIZE) / 2
                indent_y = (current_cube_size - CROP_SIZE) / 2
                indent_z = (current_cube_size - CROP_SIZE) / 2

                indent_x = int(indent_x)
                indent_y = int(indent_y)
                indent_z = int(indent_z)
                
                #?????????crop_size?????????cube
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
                
                if CROP_SIZE != CUBE_SIZE:
                    cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                    
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
                
                #????????????
                if train_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]
                        
                        
            #??????cube???????????????100??????cube?????????
            means.append(cube_image.mean())
            if train_set: 
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            
            
            #RNN???????????????????????? 1024*32 
            img3d = prepare_image_for_net3D_RNN(cube_image)
                    
            #????????????
            img_list.append(img3d)
            class_list.append(class_label)
            size_list.append(size_label)

            batch_idx += 1
            
            if batch_idx >= batch_size:
                
                x = numpy.vstack(img_list)
                y_class = numpy.vstack(class_list)
                y_size = numpy.vstack(size_list)
                yield x, {"out_class": y_class, "out_malignancy": y_size}
                img_list = []
                class_list = []
                size_list = []
                batch_idx = 0

def prepare_image_for_net3D_RNN(img):
    
   
    img = img
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    
    imgN=numpy.zeros(shape=(16,32,32))

    for i in range(16):
    
        imgN[i]=(img[2*i]+img[2*i+1])/2
        
        
    imgN = imgN.reshape(1,8,8,8,32)
    
    return imgN	
		
		

	
def get_train_holdout_files(train_percentage=80, full_luna_set=False):
    
    print("Get train/holdout files.")
    
    
    ####################################################################################################
    ####################################################################################################
    
    #positive cube?????????
    
    #luna16 ????????????1
    pos_samples = glob.glob(settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_lidc/*.png")
    print("Pos samples: ", len(pos_samples))
    
    random.shuffle(pos_samples)
      
    #?????????????????????????????????
    train_pos_count = int((len(pos_samples) * train_percentage) / 100)
    pos_samples_train = pos_samples[:train_pos_count]
    pos_samples_holdout = pos_samples[train_pos_count:]
    
    #??????????????????????????????????????????????????????
    if full_luna_set:
        pos_samples_train += pos_samples_holdout
       

    #################################################################################################################
    #################################################################################################################
    
    #negative cube?????????
    
    neg_samples_edge = glob.glob(settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_edge.png")
    print("Edge samples: ", len(neg_samples_edge))
    
    neg_samples_luna = glob.glob(settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_luna.png")
    print("Luna samples: ", len(neg_samples_luna))
    
    neg_samples = neg_samples_edge + neg_samples_luna
    random.shuffle(neg_samples)
    
    #?????????????????????????????????
    train_neg_count = int((len(neg_samples) * train_percentage) / 100)
    neg_samples_train = neg_samples[:train_neg_count]
    neg_samples_holdout = neg_samples[train_neg_count:]
    
    #??????????????????????????????????????????????????????
    if full_luna_set:
        neg_samples_train += neg_samples_holdout
    
    
    #################################################################################################################
    #################################################################################################################
    
    print("Positive Train Count:",len(pos_samples_train))
    print("Negative Train Count:",len(neg_samples_train))

    #################################################################################################################
    #################################################################################################################
    
    
    #??????????????????
    train_res = []
    holdout_res = []
    sets = [(train_res, pos_samples_train, neg_samples_train), (holdout_res, pos_samples_holdout, neg_samples_holdout)]
    
    #?????????????????????
    for set_item in sets:
        
        pos_idx = 0
        negs_per_pos = NEGS_PER_POS
        
        res = set_item[0]
        pos_samples = set_item[1]
        neg_samples = set_item[2]
        
        for index, neg_sample_path in enumerate(neg_samples):
            
            res.append((neg_sample_path, 0, 0))
            
            if index % negs_per_pos == 0:
                
                pos_sample_path = pos_samples[pos_idx]
                
                file_name = ntpath.basename(pos_sample_path)
                
                parts = file_name.split("_")
                
                if True:
                    
                    class_label = int(parts[-2])
                    size_label = int(parts[-3])
                    
                    assert class_label == 1
                    assert parts[-1] == "pos.png"
                    assert size_label >= 1
                    
                    
                res.append((pos_sample_path, class_label, size_label))
                pos_idx += 1
                pos_idx %= len(pos_samples)
                
    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    
    return train_res, holdout_res

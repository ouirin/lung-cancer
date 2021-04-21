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
            
        train(model_name="luna16_full_CNN", train_full_set=True, load_weights_path=None)      
        
        if not os.path.exists("models/"):
            os.mkdir("models")
            
        shutil.copy("workdir/model_luna16_full_CNN_best.hd5", "models/model_luna16_full_CNN_best.hd5")
		

def train(model_name, train_full_set, load_weights_path):
    
    batch_size = 16
    
    #获得训练和测试集合，以：路径、class label，size label的形式保存
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, full_luna_set=train_full_set)
    
    #训练数据集
    train_gen = data_generator(batch_size, train_files, train_set=True)
    
    #print(train_gen.__next__()[1])
    #sys.exit(1)
    
    #测试数据集
    holdout_gen = data_generator(batch_size, holdout_files, train_set=False)
   
    #动态设置学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    
    #获取model
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
    
    out_sphericiy = Dense(1, activation=None, name='out_sphericiy')(gru3)
    model = Model(input=input3, output=out_sphericiy)
  
    
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)

    model.compile(optimizer=Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True), loss={"out_sphericiy": mean_absolute_error2}, metrics={"out_sphericiy": [mean_absolute_error2] })
    model.summary(line_length=140)
    
    return model

def mean_absolute_error2(y_true, y_pred):
    return K.mean( K.abs((y_pred-y_true)/y_true),axis=-1 )

def attention_3d_block(inputs):
   
    a = Permute((2, 1))(inputs)
    a = Dense(8, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    
    return output_attention_mul
	
def data_generator(batch_size, record_list, train_set):
    
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    
    while True:
        
        img_list = []
        subtlety_list = []
        lobulation_list = []
        internal_structure_list = []
        calcification_list = []
        texture_list = []
        spiculation_list = []
        margin_list = []
        sphericiy_list = []
        malignacy_list = []
        diameter_list = []
        
        if train_set:
            random.shuffle(record_list)
            
        CROP_SIZE = CUBE_SIZE
        
        #逐一遍历所有数据
        for record_idx, record_item in enumerate(record_list):
            
            subtlety_label = record_item[10] 
            lobulation_label = record_item[9] 
            internal_structure_label = record_item[8] 
            calcification_label = record_item[7] 
            texture_label = record_item[6] 
            spiculation_label = record_item[5] 
            margin_label = record_item[4] 
            sphericiy_label = record_item[3] 
            malignacy_label = record_item[2] 
            diameter_label = round(record_item[1],4) 
                         
            #处理cube
            cube_image = helpers.load_cube_img(record_item[0], 8, 8, 64)

            current_cube_size = cube_image.shape[0]

            indent_x = (current_cube_size - CROP_SIZE) / 2
            indent_y = (current_cube_size - CROP_SIZE) / 2
            indent_z = (current_cube_size - CROP_SIZE) / 2

            #数据增强
            wiggle_indent = CROP_SIZE / 4
            wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1

            if train_set:

                indent_x = wiggle_indent + random.randint(0, wiggle)
                indent_y = wiggle_indent + random.randint(0, wiggle)
                indent_z = wiggle_indent + random.randint(0, wiggle)

            indent_x = int(indent_x)
            indent_y = int(indent_y)
            indent_z = int(indent_z)

            #截取到crop_size大小的cube
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]

            if CROP_SIZE != CUBE_SIZE:
                cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

            #数据增强
            if train_set:
                if random.randint(0, 100) > 50:
                    cube_image = numpy.fliplr(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = numpy.flipud(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, :, ::-1]
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, ::-1, :]
            
            #3D卷积的正规化 32*32*32
            img3d = prepare_image_for_net3D(cube_image)
                    
            #添加数据
            img_list.append(img3d)
            
            subtlety_list.append(subtlety_label)
            lobulation_list.append(lobulation_label) 
            internal_structure_list.append(internal_structure_label) 
            calcification_list.append(calcification_label) 
            texture_list.append(texture_label)  
            spiculation_list.append(spiculation_label)  
            margin_list.append(margin_label)  
            sphericiy_list.append(sphericiy_label)  
            malignacy_list.append(malignacy_label)  
            diameter_list.append(diameter_label)  

            batch_idx += 1
            
            if batch_idx >= batch_size:
                
                x = numpy.vstack(img_list)
                y_diamter = numpy.vstack(diameter_list)
                y_malignacy = numpy.vstack(malignacy_list)
                y_sphericiy = numpy.vstack(sphericiy_list)
                y_margin = numpy.vstack(margin_list)
                y_spiculation = numpy.vstack(spiculation_list)
                y_texture = numpy.vstack(texture_list)
                y_calcification = numpy.vstack(calcification_list)
                y_internal_structure = numpy.vstack(internal_structure_list)
                y_lobulation = numpy.vstack(lobulation_list)
                y_subtlety = numpy.vstack(subtlety_list)

                yield x, {"out_diamter": y_diamter, "out_malignancy": y_malignacy, "out_sphericiy": y_sphericiy, "out_margin": y_margin, "out_spiculation": y_spiculation, "out_texture": y_texture, "out_calcification": y_calcification, "out_internal_structure": y_internal_structure, "out_lobulation": y_lobulation, "out_subtlety": y_subtlety }
                img_list = []
                subtlety_list = []
                lobulation_list = []
                internal_structure_list = []
                calcification_list = []
                texture_list = []
                spiculation_list = []
                margin_list = []
                sphericiy_list = []
                malignacy_list = []
                diameter_list = []
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
    
    #LIDC放射学家标注数据合集
    samples = glob.glob(settings.BASE_DIR_SSD + "LIDC_Annotations/cube/*.png")
    print("Samples Quantity: ", len(samples))
    
    random.shuffle(samples)
      
    #分割训练数据和测试数据
    train_count = int((len(samples) * train_percentage) / 100)
    samples_train = samples[:train_count]
    samples_holdout = samples[train_count:]
    
    #如果需要训练所有数据，训练集则为全集
    if full_luna_set:
        samples_train += samples_holdout
       
    #################################################################################################################
    #################################################################################################################
    
    print("Train Count:",len(samples_train))

    #################################################################################################################
    #################################################################################################################
    
    #建立描述集合
    train_res = []
    holdout_res = []
    sets = [(train_res, samples_train), (holdout_res, samples_holdout)]
    
    #对集合进行处理
    for set_item in sets:
  
        res = set_item[0]
        samples = set_item[1]
          
        for index, sample_path in enumerate(samples):    
        
            file_name = ntpath.basename(sample_path)

            parts = file_name.split("_")
        
            if True:
                
                subtlety_label = float(parts[-3])
                lobulation_label = float(parts[-4])
                internal_structure_label = float(parts[-5])
                calcification_label = float(parts[-6])
                texture_label = float(parts[-7])
                spiculation_label = float(parts[-8])
                margin_label = float(parts[-9])
                sphericiy_label = float(parts[-10])
                malignacy_label = float(parts[-11])
                diameter_label = float(parts[-12])

                res.append((sample_path, diameter_label, malignacy_label, sphericiy_label, margin_label, spiculation_label, texture_label, calcification_label, internal_structure_label, lobulation_label, subtlety_label))
          
    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
   
    return train_res, holdout_res
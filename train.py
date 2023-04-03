from tensorflow.keras.callbacks import *
from tensorflow.keras.datasets import mnist
from tkinter import Tk, filedialog
import tensorflow as tf
#importing GenerateData.py
import GenerateData
import numpy as np
#importing model.py
import model
import random
import os
import math


class DataPipeline:

    def __init__(self, folder_path = None):
        self.folder_path = folder_path

    def data_pipeline(self, data_type = None):

        if data_type == 'testing':

            with tf.device('/cpu:0'):
        
                os.chdir(os.path.join(self.folder_path,data_type))
                input_list =  np.load('x_list.npy', allow_pickle = True)
                input_list = list(input_list)
    
                dataset = tf.data.Dataset.from_tensor_slices(input_list)
                #dataset = dataset.shuffle(len(input_list))
                dataset = dataset.map(self.process_test_path, num_parallel_calls=8)
    
                dataset = dataset.batch(16).prefetch(1)
    
                return dataset
        
        else:

            with tf.device('/cpu:0'):
        
                #load input and output filenames
                os.chdir(os.path.join(self.folder_path,data_type))
                input_list =  np.load('x_list.npy', allow_pickle = True)
                input_list = list(input_list)
                output_list = np.load('y_list.npy', allow_pickle = True)
                output_list = list(output_list)
    
                #Create the dataset from slices of the input and output filenames
                dataset = tf.data.Dataset.from_tensor_slices((input_list, output_list))
        
                #shuffle the data with a buffer size equal to the length of the dataset
                #(this ensures good shuffling)
                dataset = dataset.shuffle(len(input_list))
        
                if data_type != 'validation':
                    dataset = dataset.repeat()
                #Parse the images from filename to the pixel values.
                #Use multiple threads to improve the speed of preprocessing
                dataset = dataset.map(self.process_path, num_parallel_calls=8)
    
                #Preprocess images- multiplying output by 255, so that ourput range is 0-255.
                #Use multiple threads to improve the speed of preprocessing
                dataset = dataset.map(self.preprocess, num_parallel_calls=8)
    
                #batch the images and prefetch one batch
                #to make sure that a batch is ready to be served at all time
                dataset = dataset.batch(16).prefetch(1)
    
                return dataset
    
    def process_test_path(self, image_path):

        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    def process_path(self, image_path, mask_path):
    
        #read the content of the file
        img = tf.io.read_file(image_path)
    
        #decode using jpeg format
        img = tf.image.decode_png(img, channels=1)
    
        #convert to float values in [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        return img, mask

    def preprocess(self, image, mask):
    
        #to ensure output range is 0-255
        mask = mask * 255.0

        return image, mask

class Training:

    def __init__(self, fine_tuning_data = None, train_data = None, val_data = None, test_data = None):
        self.fine_tuning_data = fine_tuning_data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train(self):

        #nested fuunction to implement learning rate decay
        def step_decay(epoch, lr):
                initial_lrate = 0.001
                drop = 0.5
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                return lrate
        
        #initializing and compiling model
        model_object = model.DNN(input_shape=(256,256,1))
        dnn = model_object.model_instance()
        dnn.compile(loss='mean_absolute_error', optimizer = tf.keras.optimizers.Adam(clipvalue=1))

        #create folder in current working directory to store training weights
        os.makedirs('weights', exist_ok=True)
        #logs train and validation losses
        csv_logger = CSVLogger(os.path.join('fine-tuning-losses.log'))   
        lrate = LearningRateScheduler(step_decay, verbose=1)

        #callbacks list will be passed to fit function
        callbacks_list = [lrate, csv_logger]

        #training to fine-tune the weights
        dnn.fit(self.fine_tuning_data, steps_per_epoch = 375,
                    epochs=20, callbacks=callbacks_list, initial_epoch=0)
        
        #save weights
        dnn.save_weights(filepath = os.path.join('weights','fine_tuned_weights'), save_format='tf')

        csv_logger_2 = CSVLogger(os.path.join('main_losses.log'))

        callbacks_list = [lrate, csv_logger_2]

        #train on main data set having fine-tuned the network
        dnn.fit(self.train_data, steps_per_epoch=5625,
                    epochs=20, validation_data = self.val_data, callbacks=callbacks_list, initial_epoch=0)
        
        #save weights
        dnn.save_weights(filepath = os.path.join('weights','main_weights'), save_format='tf')

        #predict on test data
        predictions = dnn.predict(self.test_data, verbose = 1)

        #save predictions
        np.save('predictions.npy', predictions, allow_pickle = True)

#driver code to execute script on running
if __name__ == '__main__':

    #store current working directory
    cwd = os.getcwd()

    #loading and splitting mnist images
    (x_train, _),(x_test, _ ) = mnist.load_data()
    total_data = random.sample(list(x_train),11000)

    #shuffling 11k images
    random.shuffle(total_data)
    
    #6000 images (1000 * 6 combinations of kernel and noise = 6000 ) for fine-tuning
    #90,000 images (10,000 * 9 combinations of kernel and noise = 90,000) for training
    fine_tuning_set = total_data[0:1000]
    main_train_set = total_data[1000:]

    #validation and testing set
    test_list=random.sample(list(x_test),1000)
    random.shuffle(test_list)
    main_val_set,main_test_set=np.array_split(test_list,2)

    #pointing root to Tk() to use it as Tk() in program.
    root = Tk() 
    #Hides small tkinter window.
    root.withdraw() 

    #to make selection window appear above all open windows
    root.attributes('-topmost', True) 

    #Returns opened path as str
    fine_tuning_dir = filedialog.askdirectory(title='SELECT FINE-TUNING ROOT FOLDER') 
    main_dir = filedialog.askdirectory(title='SELECT MAIN DATASET ROOT FOLDER')

    #make fine-tuning data set
    fine_tuning_data_object = GenerateData.GenerateData(dir = fine_tuning_dir, x_train= fine_tuning_set)
    fine_tuning_data_object.execute(type = 'fine-tuning set')

    #make main data set
    main_data_object = GenerateData.GenerateData(dir= main_dir, x_train = main_train_set, x_val= main_val_set, 
                                                                                x_test= main_test_set)
    main_data_object.execute(type = 'main dataset')

    #create relevant tf.data.Dataset pipelines

    tuning_pipeline_object = DataPipeline(folder_path = fine_tuning_dir)
    fine_tuning_pipeline = tuning_pipeline_object.data_pipeline(data_type = 'train')

    main_pipeline_object = DataPipeline(folder_path = main_dir)
    train_pipeline = main_pipeline_object.data_pipeline(data_type = 'train')
    val_pipeline = main_pipeline_object.data_pipeline(data_type = 'validation')
    test_pipeline = main_pipeline_object.data_pipeline(data_type = 'testing')


    #training
    training_object = Training(fine_tuning_data = fine_tuning_pipeline, train_data = train_pipeline,
                                            val_data = val_pipeline, test_data = test_pipeline)
    
    #change current working directory to cwd in order to make sure weights and losses are logged here
    os.chdir(cwd)
    training_object.train()

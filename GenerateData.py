import numpy as np
import cv2
import os

#class to generate data
class GenerateData:

    def __init__(self, dir = None, x_train = None, x_val = None, x_test = None):
        self.dir = dir
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        

    def createFoldersandImages(self, data_type = None, kernels =  None, noise_levels = None, X= None):

        #making requisite directories
        path = os.path.join(self.dir,data_type)
        input_path = os.path.join(path, 'input')
        output_path = os.path.join(path,'output')
        os.makedirs(path, exist_ok = True)
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path,exist_ok=True)
        os.chdir(path)

        #creating and saving images in aforementioned directories
        self.writeImages(X= X, main_path = path, output_path = output_path, input_path = input_path, 
                                kernels = kernels, noise_levels = noise_levels)


    def writeImages(self, X =  None, main_path = None, output_path = None, input_path = None, kernels = None, noise_levels = None):
        
        x = []
        y = []
        c=0

        #for every image
        for img in X:
            #for every noise level
            for noise_level in noise_levels:
                #in every kernel
                for kernel in kernels:
                    #converting uint8 to float32
                    img = img.astype('float32')
                
                    #resizing to 256x256
                    img_256x256 = cv2.resize(img,(256,256))
                
                    #Convolution
                    image_in = cv2.filter2D(src = img_256x256, kernel = kernel, ddepth = -1)
                
                    #Adding noise to the resulting 256x256 image
                    image_in = self.noisy('gauss', image = image_in, level =  noise_level)
                
                    #normalizing image such that input values range between 0-255.0
                    maxI = np.max(image_in)
                
                    minI = np.min(image_in)
                
                    mat = minI*np.ones(image_in.shape)
                
                    new_maxI = maxI - minI
                
                    image_in = (image_in - mat)*255/new_maxI
                
                    #saving image as png; 0-255 range chosen to avoid storing blank images
                    os.chdir(input_path)
                    cv2.imwrite(str(c)+'.png', image_in) 
                    os.chdir(output_path)
                    cv2.imwrite(str(c)+'.png', img_256x256)
                
                    #saving image filenames in a list - will be useful when creating tf.data.Dataset pipeline
                    x.append(os.path.join(input_path,str(c)+'.png'))
                    y.append(os.path.join(output_path,str(c)+'.png'))
                
                    c = c + 1
                
        os.chdir(main_path)
        np.save('x_list.npy', x, allow_pickle = True)
        np.save('y_list.npy', y, allow_pickle = True)

    #function to add noise
    def noisy(self, noise_typ = None, image = None, level = None):

        if noise_typ == 'gauss' :

            row,col = image.shape
        
            mean = 0
            sigma = level*np.max(image)
            gauss = np.random.normal(mean,sigma,(row,col))
            gauss = gauss.reshape(row,col)
        
            noisy_image = image + gauss
        
            #clipping negative values to zero
            noisy_image=noisy_image.clip(min=0)
        
            return noisy_image

    def execute(self, type = None):


        #create fine-tuning data
        if type == 'fine-tuning set':

            kernels = [np.random.rayleigh(scale=1.0, size = (128, 128)),
                        np.random.rayleigh(scale=1.0, size = (128,128))]
            
            noise_levels = [0.125, 0.225, 0.325]

            self.createFoldersandImages(data_type = 'train', kernels = kernels, 
                                            noise_levels= noise_levels, X = self.x_train)
        
        #create main training, validation and testing data
        else:
            
            train_kernels = [np.random.rayleigh(scale=1.0, size = (256,256)), 
                                np.random.rayleigh(scale=1.0, size = (256,256)), 
                                    np.random.rayleigh(scale=1.0, size = (256,256))]

            val_kernels = [np.random.rayleigh(scale=1.0, size = (256,256)), 
                            np.random.rayleigh(scale=1.0, size = (256,256))]

            test_kernels = [np.random.rayleigh(scale=1.0, size = (256,256)), 
                                np.random.rayleigh(scale=1.0, size = (256,256))]

            train_noise_levels = [0.1, 0.2, 0.3]
            val_noise_levels = [0.15, 0.25]
            test_noise_levels = [0.12, 0.22]



            self.createFoldersandImages(data_type = 'train', kernels = train_kernels, 
                                                noise_levels = train_noise_levels, X = self.x_train)

            self.createFoldersandImages(data_type = 'validation', kernels = val_kernels, 
                                                noise_levels = val_noise_levels, X = self.x_val)

            self.createFoldersandImages(data_type = 'testing', kernels = test_kernels, 
                                                noise_levels = test_noise_levels, X = self.x_test)

#driver code to test this script
'''if __name__ == "__main__":

    (x_train, _),(x_test, _ ) = mnist.load_data()
    total_data = random.sample(list(x_train),11000)
    random.shuffle(total_data)
    
    fine_tuning_set = total_data[0:1000]
    main_train_set = total_data[1000:]

    test_list=random.sample(list(x_test),1000)
    random.shuffle(test_list)
    main_val_set,main_test_set=np.array_split(test_list,2)


    obj1 = GenerateData(x_train= fine_tuning_set)
    obj1.execute(type = 'fine-tuning set')

    obj2= GenerateData(x_train = main_train_set, x_val= main_val_set, x_test= main_test_set)
    obj2.execute(type = 'main dataset')
'''
    






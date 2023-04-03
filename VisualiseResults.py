from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

predictions = np.load('predictions.npy', allow_pickle = True)

root = Tk() 
root.withdraw() 

root.attributes('-topmost', True)

testing_dir = filedialog.askdirectory(title='SELECT TESTING IMAGE DIRECTORY')

input_path = os.path.join(testing_dir,'input')

output_path = os.path.join(testing_dir,'output')

#generate random index everytime script is run - it is returned as a list of length = 1
random_idx = random.sample(range(0,2000),1)

dnn_input = cv2.imread(os.path.join(testing_dir,os.path.join(input_path,str(random_idx[0])+'.png')),0)

exp_output = cv2.imread(os.path.join(testing_dir,os.path.join(output_path,str(random_idx[0])+'.png')),0)

fig = plt.figure(figsize=(3, 4))

fig.add_subplot(1, 3, 1)
plt.imshow(dnn_input.astype('float32'))
plt.title('DNN I/P')

fig.add_subplot(1, 3, 2)
plt.imshow(exp_output.astype('float32'))
plt.title('EXPECTED O/P')

fig.add_subplot(1, 3, 3)
plt.imshow(np.squeeze(predictions[random_idx[0]]).astype('float32'))
plt.title('DNN O/P')


plt.show()

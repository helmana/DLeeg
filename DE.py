
# coding: utf-8

# In[2]:


import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne import io
from mne import viz
#from mne.datasets import testing
from mne import Epochs, io, pick_types
from mne.event import define_target_events
from mne.time_frequency import psd_welch
print(__doc__)


# In[3]:


subject_number=2  



# load dataset in array
list_raw_fnames = [[0]*2]*subject_number
for x in range(subject_number):
    list_raw_fnames[x] = mne.datasets.eegbci.load_data(x+1,[1,2])

list_rawdata = np.zeros((subject_number,2), dtype='object')

for i in range(subject_number):
    for j in range(2):
        list_rawdata[i][j] = mne.io.read_raw_edf(list_raw_fnames[i][j], preload=True)
print( list_rawdata[:][:])


# In[118]:


import math
task_number=2
task_time = 60
sampel_number_per_sec =  160 # sampel rate
total_sampel_number =  sampel_number_per_sec *task_time # 60*160
sample_shift = 4 #step len
subject_img_number = math.floor((total_sampel_number - sampel_number_per_sec) / sample_shift) +1
subject_img_number


# In[247]:


img = np.zeros((subject_img_number*subject_number, 1, 64, 160), dtype = float)
label =[]

#x_test = np.zeros((1, 1, 64, 160), dtype = float)
#y_test =[]
#images

def normalize_channel_data(ch , min_t, max_t):

    ch = ((ch - min_t) / (max_t - min_t ) ) *1000
    #print(ch)
    return ch


# In[248]:



for s in range(subject_number):
    rawdataChannels_t, times_t=rawdataChannels, times =list_rawdata[s][0][:,:]
    ch_max =[]
    ch_min =[]
    for j in range(len(rawdataChannels_t)):
        ch_max = np.append(ch_max, max(rawdataChannels_t[j]))
        ch_min = np.append(ch_min, min(rawdataChannels_t[j]))
   
    min_t = min(ch_min) 
    max_t = max(ch_max) 
    print(min_t, max_t)
    for i in range(subject_img_number):
        rawdataChannels, times=rawdataChannels, times =list_rawdata[s][0][:64,0+i*sample_shift:sampel_number_per_sec+i*sample_shift]
        
        rawdataChannels = normalize_channel_data(rawdataChannels, min_t, max_t)
        
        img[s*subject_img_number + i][0] = rawdataChannels
        label = np.append(label, (s))
img


# In[249]:


import keras
from keras.utils import to_categorical

label = to_categorical(label, subject_number)


# In[124]:


img.shape


# In[250]:


# test & trian
train_img = img[0:4000]
train_label =label[0:4000]

x_test = img[4000:]
y_test =label[4000:]

# valid & train
x_train =train_img[:3000]
y_train =train_label[:3000]

x_valid =train_img[3000:]
y_valid =train_label[3000:]


# In[252]:


from keras import layers
from keras import models
from keras import regularizers


model = models.Sequential()
model.add(layers.Conv2D(64, (5,5), activation = 'relu', input_shape = (1,64,160), data_format= "channels_first" ))
print(model.output.shape)
model.add(layers.MaxPooling2D((2,2)))
print(model.output.shape)
model.add(layers.Conv2D(128, (5,5), activation = 'relu'))
print(model.output.shape)
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, (5,5), activation = 'relu'))
print(model.output.shape)
model.add(layers.MaxPooling2D((2,2)))
print(model.output.shape)
model.add(layers.Flatten())
print(model.output.shape)
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dropout(0.5))
print(model.output.shape)
model.add(layers.Dense(subject_number, activation = 'softmax'))
print(model.output.shape)


# In[253]:


from keras import optimizers

model.compile(loss= 'categorical_crossentropy',
              optimizer= optimizers.RMSprop(lr= 1e-4),
              metrics = ['acc'])


# In[255]:


history = model.fit(
        x_train,
        y_train,
        epochs = 5,
        batch_size = 10,
        validation_data = (x_valid, y_valid)
)


# In[256]:


model.save('EEG_dataset')


# In[257]:


import matplotlib.pyplot as plt

history_dict = history.history 

loss_values = history_dict ['loss'] 

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss') 

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss') 

plt.xlabel('Epochs') 

plt.ylabel('Loss') 

plt.legend()

plt.show()


# In[259]:


history_dict = history.history 

acc_values = history_dict ['acc'] 

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'bo', label='Training acc') 

plt.plot(epochs, val_acc_values, 'b', label='Validation acc') 

plt.title('Training and validation acc') 

plt.xlabel('Epochs') 

plt.ylabel('acc') 

plt.legend()

plt.show()


# model.evaluate(x_test, y.test)

# In[260]:


model.evaluate(x_test, y_test)


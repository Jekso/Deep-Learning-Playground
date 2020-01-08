
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import collections
import myo
import threading
import time


# In[2]:


class Listener(myo.DeviceListener):
    def __init__(self, queue_size=8):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append(event.emg)
            
    def data_size(self):
        return len(self.emg_data_queue)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)


# In[3]:


def myo_sign_language_predict(myo_emg_input):    
    scaler = pickle.load(open('model_checkpoints/scaler.sav', 'rb'))
    myo_emg_input = np.array([myo_emg_input[0:64]])
    # print(myo_emg_input)
    myo_emg_input = scaler.transform(myo_emg_input)
    model = load_model('model_checkpoints/my_model.h5')
    print('Predicted Number is : ' + str(model.predict_classes(myo_emg_input)[0]))


# In[ ]:


myo.init()
hub = myo.Hub()
listener = Listener(512)

try:
    threading.Thread(target=lambda: hub.run_forever(listener.on_event)).start()
    print('Start Connecting...')
    while True:
        if(listener.data_size() >= 64):
            myo_emg_input = np.array([x for x in listener.get_emg_data()]).flatten()
            myo_sign_language_predict(myo_emg_input)
finally:
    hub.stop()  # Will also stop the thread


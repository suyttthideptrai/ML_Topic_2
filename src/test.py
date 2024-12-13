import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

from utils import load_wav_16k_mono
from IPython import display

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


file_name = 'bark.wav'
file_url = 'https://storage.googleapis.com/audioset/yamalyzer/audio/bark.wav'

testing_wav_file_name = tf.keras.utils.get_file(file_name,
                                                file_url,
                                                cache_dir='./',
                                                cache_subdir='test_data')

print(testing_wav_file_name)

testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

_ = plt.plot(testing_wav_data)

# Play the audio file.
display.Audio(testing_wav_data, rate=16000)
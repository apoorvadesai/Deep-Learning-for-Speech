import numpy as np
from scipy.io import wavfile
import os  

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras import optimizers
from keras.utils import to_categorical

from keras.initializers import glorot_normal
import math
import matplotlib.pyplot as plt
import glob

number_of_input_files=5

files_wav = os.listdir('/Users/apoorvadesai/ee599/wav/')
onlyJac_fullname = [x for x in files_wav if 'AJJacobs' in x]
onlyJac_basename=[]
for i in onlyJac_fullname:
    basename, _=os.path.splitext(i)
    onlyJac_basename.append(basename)

files_wav = os.listdir('/Users/apoorvadesai/ee599/wav/')
onlyAlgore_fullname = [x for x in files_wav if 'AlGore' in x]
onlyAlgore_basename=[]
for i in onlyAlgore_fullname:
    basename, _=os.path.splitext(i)
    onlyAlgore_basename.append(basename)
    
phone_list = np.loadtxt("phonelist.txt", delimiter='\n', unpack=False, dtype = 'str')

all_alignments = np.loadtxt("all_alignments.txt", delimiter=',', unpack=False, dtype = 'str')
newJac = [x for x in all_alignments if 'AJJacobs' in x]
d = {}
new_dict = {}
for row in newJac:
    key = row.split()[0]
    value = row.split()[1:]
    d[key] = value
    new_dict[key] = [i.partition('_')[0] for i in d[key]]

newAlgore = [x for x in all_alignments if 'AlGore' in x]
d_test = {}
new_dict_test = {}
for row in newAlgore:
    key = row.split()[0]
    value = row.split()[1:]
    d_test[key] = value
    new_dict_test[key] = [i.partition('_')[0] for i in d_test[key]]


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
        
def normalise_audio(audio_signal):
    max_val=np.max(abs(audio_signal))
    print(max_val)
    audio_signal=np.divide(audio_signal,max_val)
    return audio_signal

    
def create_windowed_dataset(audio_signal, window_size, step_size):
    x=[]
    for i in range(0,(len(audio_signal)-window_size+1),step_size):
        x.append(audio_signal[i:i+window_size])
    return x


def window_ham_rfft(audio_windowed_seq):
    hamming_window=np.hamming(400)
    hamminged_audio=[frames*hamming_window for frames in audio_windowed_seq]
    rfft= np.fft.rfft(np.array(hamminged_audio))  #check if this should be absolute or no
    return rfft

def freq2mel(freq):
    num=float(1+float(freq/700.0))
    x=2595*(math.log10(num))
    return x

def mel2freq(mel):
    num = math.pow(10,float(mel/2595))
    freq=700*(num-1)
    return freq

def create_melseq(mel_bank_size,start_mel,end_mel):
    bin_size=(end_mel-start_mel)/(mel_bank_size+1)
    mel_sequence=[]
    for i in range(mel_bank_size+2):
        mel_sequence.append(i*bin_size)
    return mel_sequence

def irfft_overlap_add(audio_signal,length,window_size):
    irfft_signal=np.fft.irfft(audio_signal)
    overlap_added_signal=np.zeros(length)
    for i in range(len(irfft_signal)):
        for j in range(window_size):
            overlap_added_signal[(i*128)+j]+=irfft_signal[i,j]
    return overlap_added_signal
        
def abs_norm(signal):
    signal_1=np.absolute(signal)
    abs_norm_signal = normalise_audio(signal_1)
    return abs_norm_signal
    
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return ((2*p*r)/(p+r))

    
window_size=400
sampling_freq=16000
mel_bank_size=40

mel_sequence=np.array(create_melseq(mel_bank_size,freq2mel(0),freq2mel(8000)))
freq_sequence=[(mel2freq(mels)) for mels in mel_sequence]

mel_sequence=np.array(mel_sequence)
freq_sequence=np.array(freq_sequence)

bins=np.floor((window_size+1)*freq_sequence/sampling_freq)

mel_filter_matrix=np.zeros((mel_bank_size, int(window_size/2 +1)))


for m in range(1,len(mel_sequence)-1):
    for k in range(int(bins[m-1]),int(bins[m])):
        mel_filter_matrix[m-1,k]=(k-bins[m-1])/(bins[m]-bins[m-1])
    for k in range(int(bins[m]),int(bins[m+1])):
        mel_filter_matrix[m-1,k]=(bins[m+1]-k)/(bins[m+1]-bins[m])
        
print(mel_filter_matrix)

count=number_of_input_files
overall_train_x=[]
overall_train_y=[]
for c,i in enumerate (onlyJac_fullname):
#     print(i)
#     print(onlyJac_basename[c])
    if c<count:
        wavefile=wavfile.read(os.path.join('wav/',i))
        windowed_training_set=create_windowed_dataset(wavefile[1], 400, 160)
        np_array_audio = np.array(windowed_training_set)
        np_array_train_y=np.zeros(np_array_audio.shape[0])        
        for i,value in enumerate (new_dict[onlyJac_basename[c]]):
            np_array_train_y[i]=phone_list.tolist().index(value)
        y_train=to_categorical(np_array_train_y, 46)
        overall_train_y.append(y_train)
        np_array_audio_rfft = window_ham_rfft(np_array_audio)
        np_array_train_x=np.dot(np.abs(np_array_audio_rfft),mel_filter_matrix.T)
        overall_train_x.append(np_array_train_x)
        

overall_x_train=np.concatenate(overall_train_x)
overall_y_train=np.concatenate(overall_train_y)
print(overall_x_train.shape)
print(overall_y_train.shape)    


input_layer=Input(shape=(40,))
# encoded_1 = Dense(40, activation='tanh')(input_audio)

layer_1 = Dense(500, activation='tanh', kernel_initializer=glorot_normal(seed=0))(input_layer)
layer_2 = BatchNormalization(axis=1)(layer_1)

layer_3 = Dense(1000, activation='tanh', kernel_initializer=glorot_normal(seed=0))(layer_2)
layer_4 = BatchNormalization(axis=1)(layer_3)

layer_5 = Dense(3000, activation='tanh', kernel_initializer=glorot_normal(seed=0))(layer_4)
layer_6=BatchNormalization(axis=1)(layer_5)

layer_7 = Dense(1000, activation='tanh', kernel_initializer=glorot_normal(seed=0))(layer_6)
layer_8=BatchNormalization(axis=1)(layer_7)

layer_9 = Dense(500, activation='tanh', kernel_initializer=glorot_normal(seed=0))(layer_8)
layer_10=BatchNormalization(axis=1)(layer_9)

output = Dense(46, activation='softmax')(layer_10)
adam = optimizers.Adam(lr=0.0001)

autoencoder = Model(input_layer, output)
autoencoder.compile(optimizer=adam, loss='mse', metrics=["accuracy", precision, recall, fmeasure])

autoencoder.summary()

history=autoencoder.fit(overall_x_train,overall_y_train,epochs=30, validation_split=0.3, shuffle=True)


wavefile=wavfile.read(os.path.join('wav/','AJJacobs_2007P-0050911-0051777.wav'))
windowed_training_set=create_windowed_dataset(wavefile[1], 400, 160)
np_array_audio = np.array(windowed_training_set)
np_array_test_y=np.zeros(np_array_audio.shape[0])        
for i,value in enumerate (new_dict['AJJacobs_2007P-0050911-0051777']):
    np_array_test_y[i]=phone_list.tolist().index(value)
y_test=to_categorical(np_array_test_y, 46)
np_array_audio_rfft = window_ham_rfft(np_array_audio)
array_test=np.dot(np.abs(np_array_audio_rfft),mel_filter_matrix.T)

pred_op=autoencoder.predict(array_test)
metrics=autoencoder.evaluate(array_test,y_test, verbose=1)
print("predicting on AJJacobs")
print("Autoencoder evaluation metrics: ",autoencoder.metrics_names)
print(metrics)

wavefile=wavfile.read(os.path.join('wav/','AlGore_2006-0086158-0087175.wav'))
windowed_training_set=create_windowed_dataset(wavefile[1], 400, 160)
np_array_audio = np.array(windowed_training_set)
np_array_test_y=np.zeros(np_array_audio.shape[0])        
for i,value in enumerate (new_dict_test['AlGore_2006-0086158-0087175']):
    np_array_test_y[i]=phone_list.tolist().index(value)
y_test=to_categorical(np_array_test_y, 46)
np_array_audio_rfft = window_ham_rfft(np_array_audio)
array_test=np.dot(np.abs(np_array_audio_rfft),mel_filter_matrix.T)

pred_op=autoencoder.predict(array_test)
metrics=autoencoder.evaluate(array_test,y_test, verbose=1)
print("predicting on AlGore")
print("Autoencoder evaluation metrics: ",autoencoder.metrics_names)
print(metrics)
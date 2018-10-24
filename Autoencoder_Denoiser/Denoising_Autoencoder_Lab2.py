import youtube_dl
import numpy as np
from ffmpy import FFmpeg
import glob
import os  
import sox
import wave
from itertools import islice
from scipy.io import wavfile

from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers

import math
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


#import and audio manipulation


window_size=512 #window size n
clip_start=5 #clip audio at start seconds
clip_end=35 #clip audio at end seconds
# clean_amp=1 #amplitude of clean speech 
# noisy_amp=1  #amplitude of noise
sampling_freq=16000 #sampling rate
lr=0.001 #learning rate
no_epochs=30 #number of epochs

#insert of audio signals to be used
list_url=['https://www.youtube.com/watch?v=NzliTBkMthg','https://www.youtube.com/watch?v=phiMuhhIjAM']


dir = '.'
ydl_opts = {}

files = glob.glob(os.path.join(dir, '*.mp4'))
files_wav = glob.glob(os.path.join(dir, '*.wav'))

try:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for urls in range(len(list_url)):
            ydl.download([list_url[urls]])
except:
    print("already downloaded")
        

files = glob.glob(os.path.join(dir, '*.mp4'))
files_wav = glob.glob(os.path.join(dir, '*.wav'))

for f in range(len(files)):
    name, ext=os.path.splitext(os.path.basename(files[f]))
    if(os.path.exists(os.path.join(dir, name +'.wav'))):
        print("Wave file for this input file already exists")
    else:
        ff=FFmpeg(inputs={name +'.mp4':None},
                 outputs={name + '.wav':['-vn', '-acodec', 'pcm_s16le', '-ar', '16000' , '-ac', '1']})
        ff.run()

files_wav = glob.glob(os.path.join(dir, '*.wav'))
print(files_wav)

audio_clean=wave.open(files_wav[0],'rb')
audio_noise=wave.open(files_wav[1],'rb')

#clip audio to equal lengths and mix

if not os.path.exists('trimmed_audio'):
    os.makedirs('trimmed_audio')

sox_trans=sox.Transformer()
sox_trans.trim(clip_start,clip_end)
sox_trans.rate(sampling_freq)
# sox_trans.vol(clean_amp, gain_type='amplitude')
sox_trans.build(files_wav[0],'trimmed_audio/trimmed_clean_audio.wav')

# sox_trans.vol(noisy_amp, gain_type='amplitude')
sox_trans.build(files_wav[1],'trimmed_audio/trimmed_noisy_audio.wav')

sox_trans2=sox.Transformer()
sox_trans2.trim((clip_start + (clip_end-clip_start) +5)  ,clip_end + 15)
sox_trans2.rate(sampling_freq)
# sox_trans.vol(clean_amp, gain_type='amplitude')
sox_trans2.build(files_wav[0],'trimmed_audio/trimmed_testclean_audio.wav')

# sox_trans.vol(noisy_amp, gain_type='amplitude')
sox_trans2.build(files_wav[1],'trimmed_audio/trimmed_testNoise_audio.wav')

# sox_comb=sox.Combiner()
# sox_comb.convert(samplerate=sampling_freq)
# sox_comb.build(['trimmed_audio/trimmed_noisy_audio.wav','trimmed_audio/trimmed_noisy_audio.wav'], 'trimmed_audio/noise.wav','mix')
# sox_comb.build(['trimmed_audio/trimmed_clean_audio.wav','trimmed_audio/trimmed_clean_audio.wav'], 'trimmed_audio/speech.wav','mix')


#functions

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
    hamming_window=np.hamming(512)
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

    

#audio, normalise, create windows, apply hamming and rfft

audio_in_clean=wavfile.read('trimmed_audio/trimmed_clean_audio.wav')
audio_in_noise=wavfile.read('trimmed_audio/trimmed_noisy_audio.wav')
audio_np_clean=np.array(audio_in_clean[1],dtype=float)
audio_np_noise=np.array(audio_in_noise[1],dtype=float)

audio_np_mix=np.add(audio_np_clean,(0.75*audio_np_noise))


audio_windows=np.array(create_windowed_dataset(audio_np_mix, 512, 128))
audio_xtrain_rfft=window_ham_rfft(audio_windows)
# print(audio_xtrain_rfft)
# print(audio_xtrain_rfft.shape)
# print(np.count_nonzero(np.isnan(audio_xtrain_rfft)))

audio_np_clean=np.array(audio_in_clean[1],dtype=float)
audio_windows=np.array(create_windowed_dataset(audio_np_clean, 512, 128))
audio_ytrain_rfft=np.array(window_ham_rfft(audio_windows))
# print(audio_ytrain_rfft)
# print(audio_ytrain_rfft.shape)
# print(np.count_nonzero(np.isnan(audio_ytrain_rfft)))

audio_in_clean=wavfile.read('trimmed_audio/trimmed_testclean_audio.wav')
audio_in_noise=wavfile.read('trimmed_audio/trimmed_testNoise_audio.wav')
audio_np_clean=np.array(audio_in_clean[1],dtype=float)
audio_np_noise=np.array(audio_in_noise[1],dtype=float)

audio_np_mix=np.add(audio_np_clean,(0.55*audio_np_noise))

audio_windows=np.array(create_windowed_dataset(audio_np_mix, 512, 128))
audio_xtest_rfft=np.array(window_ham_rfft(audio_windows))
# print(audio_xtest_rfft)
# print(audio_xtest_rfft.shape)
# print(np.count_nonzero(np.isnan(audio_xtest_rfft)))

audio_np_clean=np.array(audio_in_clean[1],dtype=float)
audio_windows=np.array(create_windowed_dataset(audio_np_clean, 512, 128))
audio_ytest_rfft=np.array(window_ham_rfft(audio_windows))
# print(audio_ytest_rfft)
# print(audio_ytest_rfft.shape)
# print(np.count_nonzero(np.isnan(audio_ytest_rfft)))


#generate mel frequency bank

start_mel=freq2mel(0)
end_mel=freq2mel(8000)
mel_bank_size=40

mel_sequence=create_melseq(mel_bank_size,start_mel,end_mel)

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
# print(mel_filter_matrix)
audio_xtrain_mel=np.dot(np.abs(audio_xtrain_rfft),mel_filter_matrix.T)
audio_ytrain_mel=np.dot(np.abs(audio_ytrain_rfft),mel_filter_matrix.T)
audio_ytest_mel=np.dot(np.abs(audio_ytest_rfft),mel_filter_matrix.T)
audio_xtest_mel=np.dot(np.abs(audio_xtest_rfft),mel_filter_matrix.T)

audio_xtrain_mel1 = np.array(normalise_audio(audio_xtrain_mel))
audio_ytrain_mel1 =  np.array(normalise_audio(audio_ytrain_mel))
audio_xtest_mel1 =  np.array(normalise_audio(audio_xtest_mel))
audio_ytest_mel1 =  np.array(normalise_audio(audio_ytest_mel))

audio_fs_by_fx=np.divide(audio_ytrain_mel1,audio_xtrain_mel1)
audio_fs_by_fx_test=np.divide(audio_ytest_mel1,audio_xtest_mel1)

# audio_fs_by_fx1 = np.array(normalise_audio(audio_fs_by_fx))

input_audio=Input(shape=(40,))
# encoded_1 = Dense(40, activation='tanh')(input_audio)
encoded_2 = Dense(100, activation='tanh')(input_audio)
encoded_3 = Dense(300, activation='tanh')(encoded_2)

decoded_1 = Dense(100, activation='tanh')(encoded_3)
# decoded_2 = Dense(40, activation='tanh')(decoded_1)
output = Dense(40, activation='sigmoid')(decoded_1)
adam = optimizers.Adam(lr=lr)
autoencoder = Model(input_audio, output)
autoencoder.compile(optimizer=adam, loss='mse', metrics=["accuracy", precision, recall, fmeasure])

autoencoder.summary()

history=autoencoder.fit(audio_xtrain_mel1,audio_fs_by_fx,epochs=no_epochs, validation_split=0.2, shuffle=True)

plt.figure(1)
plt.subplot(2,2,1)
plt.plot(history.history['val_acc'])
plt.title("Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(2,2,2)
plt.plot(history.history['val_loss'])
plt.title("Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(2,2,3)
plt.plot(history.history['acc'])
plt.title("Training Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(2,2,4)
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.figure(2)
plt.subplot(2,2,1)
plt.plot(history.history['precision'])
plt.title("Precision")
plt.xlabel('epochs')
# plt.ylabel()

plt.subplot(2,2,2)
plt.plot(history.history['recall'])
plt.title("Recall")
plt.xlabel('epochs')
# plt.ylabel()

plt.subplot(2,2,3)
plt.plot(history.history['fmeasure'])
plt.title("Fmeasure")
plt.xlabel('epochs')
# plt.ylabel()
plt.show()

pred_op=autoencoder.predict(audio_xtest_mel1)

metrics=autoencoder.evaluate(audio_xtest_mel,audio_fs_by_fx_test, verbose=1)

print("Autoencoder evaluation metrics: ",autoencoder.metrics_names)
print(metrics)

mel_output=np.dot(pred_op,mel_filter_matrix)

dot_prod_mat=mel_output*audio_xtest_rfft
ifft_signal=irfft_overlap_add(dot_prod_mat,len(audio_np_mix),window_size)

ifft_signal_wr=ifft_signal/max(ifft_signal)*32767
wavfile.write("output_denoised.wav", sampling_freq, ifft_signal_wr.astype('int16'))

audio_np_mix_op=audio_np_mix/max(audio_np_mix)*32767
wavfile.write("input_test.wav", sampling_freq, audio_np_mix_op.astype('int16'))


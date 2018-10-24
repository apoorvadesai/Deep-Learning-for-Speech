import youtube_dl
import numpy as np
from ffmpy import FFmpeg
import glob
import os
import sox
import wave
from itertools import islice
from scipy.io import wavfile
import math
import matplotlib.pyplot as plt


window_size=500 #window size n
clip_start=5 #clip audio at start seconds
clip_end=25 #clip audio at end seconds
clean_amp=0.80 #amplitude of clean speech 
noisy_amp=0.20  #amplitude of noise
sampling_freq=8000 #sampling rate
lr=0.01 #learning rate


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
                 outputs={name + '.wav':['-vn', '-acodec', 'pcm_s16le', '-ar', '44100' , '-ac', '1']})
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
sox_trans.vol(clean_amp, gain_type='amplitude')
sox_trans.build(files_wav[0],'trimmed_audio/trimmed_clean_audio.wav')

sox_trans.vol(noisy_amp, gain_type='amplitude')
sox_trans.build(files_wav[1],'trimmed_audio/trimmed_noisy_audio.wav')

sox_comb=sox.Combiner()
sox_comb.convert(samplerate=sampling_freq)
sox_comb.build(['trimmed_audio/trimmed_clean_audio.wav','trimmed_audio/trimmed_noisy_audio.wav'], 'trimmed_audio/combined_output.wav','mix')

#audio preprocessing display
audio_comb=wavfile.read('trimmed_audio/combined_output.wav')
audio_clean=wavfile.read('trimmed_audio/trimmed_clean_audio.wav')

audio_np=np.array(audio_comb[1],dtype=float)
max_val=max(abs(audio_np))


for i in range(len(audio_np)):
    audio_np[i]/=max_val

def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

plt.plot(audio_np)
plt.ylabel('Normalised Amplitude')
plt.title('Input Audio')
plt.show()

#LMS without non linearity
windowed_sequence=window(audio_np,n=window_size)
w=np.random.random(window_size)
y_predict=0
y_actual=0
error=0
error_decay=[]
for sample in windowed_sequence:
    x_train=sample[:window_size-1]
    x_train=np.insert(x_train,0,1)
    y_actual=sample[window_size-1]
    y_predict=np.dot(w.T,x_train)
    error = y_predict-y_actual
    error_decay.append(math.pow(error,2))
    w=w-(lr*error*x_train)
    
print("Outputs for LMS without Non-Linearity")
print("y_pred:",y_predict)
print("y_act:",y_actual)
print("error:", error)
# print("weights:",w)

plt.plot(error_decay)
plt.ylabel('error')
plt.title('LMS Error without Non Linearity')
plt.show()


#LMS with non-linearity
windowed_sequence=window(audio_np,n=window_size)
w=np.random.random(window_size)
y_predict=0
y_actual=0
error=0
error_decay=[]
for sample in windowed_sequence:
    x_train=sample[:window_size-1]
    x_train=np.insert(x_train,0,1)
    y_actual=sample[window_size-1]
    y_predict=np.dot(w.T,x_train)
#     print(y_actual)
    non_lin_y_pred=math.tanh(y_predict)
    y_predict=non_lin_y_pred
    error = y_predict-y_actual
    error_decay.append(math.pow(error,2))
    w=w-(lr*error*x_train*(1-math.pow(non_lin_y_pred,2)))

print("Outputs for LMS with Tanh Non-Linearity")
print("y_pred:",y_predict)
print("y_act:",y_actual)
print("error:", error)

plt.plot(error_decay)
plt.title('LMS Error with Tanh Non-Linearity')
plt.ylabel('error')
plt.show()

windowed_sequence=window(audio_np,n=window_size)


lr=0.01

w=np.random.random(window_size)
y_predict=0
y_actual=0
error=0

error_decay=[]

for sample in windowed_sequence:
    x_train=sample[:window_size-1]
    x_train=np.insert(x_train,0,1)
    y_actual=sample[window_size-1]
    y_predict=np.dot(w.T,x_train)
#     print(y_actual)
    non_lin_y_pred=np.reciprocal((1+math.exp(-y_predict)))
    y_predict=non_lin_y_pred
#     print(y_predict)
    error = y_predict-y_actual
    error_decay.append(math.pow(error,2))
    w=w-(lr*error*x_train*(non_lin_y_pred*(1-non_lin_y_pred)))

print("Outputs for LMS with Sigmoid Non-Linearity")   
print("y_pred:",y_predict)
print("y_act:",y_actual)
print("error:", error)
# print("weights:",w)
plt.plot(error_decay)
plt.ylabel('error')
plt.title('LMS Error with Sigmoid Non-linearity')
plt.show()


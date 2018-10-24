To run this code you will need to have SoX and its python wrapper pysox, ffmpeg and its python wrapper ffmpy, youtube-dl and its python wrapper youtube_dl.

Run the .py file to execute the program. It will output plots and values for the input audio signal, outputs for LMS without Non-Linearity and LMS with Non-Linearity of Tanh and Sigmoid. The code will also create a subfolder of the processed audio signals. This will include a clipped, amplitude modified and resampled version of both the clean and noise signals. It also saves a third signal which is a combined version of the two.

Values that you can change are from line 14 to line 25 of the .py file. These include the window size n for the samples, duration of the audio signals to clip, amplitudes of the audio signals and learning rate for the neural net. To use audio other than the default ones, input the url at list_url on line 24.

import os, sys
from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt

import speech_recognition as sr

import sounddevice as sd
import pyaudio
import wave



class AudioData(sr.AudioData):
    def __init__(self, frame_data, sample_rate, sample_width, sample_channels):
        super().__init__(frame_data, sample_rate, sample_width)
        self.sample_channels = sample_channels

        if sample_width == 4:
            self.sample_format_np = np.int32
        elif sample_width == 2:
            self.sample_format_np = np.int16
        elif sample_width == 1:
            self.sample_format_np = np.int8
        else:
            raise ValueError(' sample_width must be: 4, 2 or 1')
        
        self.sample_norm = np.iinfo(self.sample_format_np).max

        self._signal = None
        return None

    def get_signal(self):
        if self._signal is None:
            frame_bytes = self.frame_data
            w = np.frombuffer(self.frame_data, dtype=self.sample_format_np) / self.sample_norm
            self._signal =  w.reshape(-1, self.sample_channels)
            
        return self._signal

    def plot(self, voice_threshold=None):
        signal = self.get_signal()

        t = np.arange(signal.shape[0]) / self.sample_rate
        for i_c, w in enumerate(signal.T):
            plt.plot(t, w, label='Channel {}'.format(i_c))

        if voice_threshold is not None:
            plt.plot([0, signal.shape[0]/self.sample_rate], [ voice_threshold,  voice_threshold], 'r--', label='Voice Threshold')
            plt.plot([0, signal.shape[0]/self.sample_rate], [-voice_threshold, -voice_threshold], 'r--')

        plt.xlabel('Time (s)')
        plt.ylabel('Mic signal (-)')
        
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        return None

    def to_mono(self):
        signal = self.get_signal()
        signal_mono = signal.mean(axis=-1)
        
        frame_data_mono = (signal_mono * self.sample_norm).astype(self.sample_format_np)
        audio_data = AudioData(frame_data_mono, self.sample_rate, self.sample_width, 1)

        return audio_data

    def get_aiff_data(self, convert_rate=None, convert_width=None):
        if self.sample_channels !=1:
            raise NotImplementedError('get_aiff_data only works for mono samples')
        else:
            return super().get_aiff_data()

    def get_flac_data(self, convert_rate=None, convert_width=None):
        if self.sample_channels !=1:
            raise NotImplementedError('get_flac_data only works for mono samples')
        else:
            return super().get_flac_data()

    def get_wav_data(self, convert_rate=None, convert_width=None):
        if self.sample_channels != 1:
            raise NotImplementedError('get_wav_data only works for mono samples. (you can use save_wav method)')
        else:
            return super().get_wav_data(convert_rate, convert_width)


    def get_segment(self, start_ms=None, end_ms=None):
        if self.sample_channels !=1:
            raise NotImplementedError('get_segment only works for mono samples.')
        else:
            return super().get_segment()


    def save_wav(self, filename='borrame.wav'):
        
        # Save the recorded data as a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.sample_channels)
            wf.setsampwidth(self.sample_width )
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.frame_data)
        
        return None

    def play(self):
        signal = self.get_signal()
        sd.play(
            signal.mean(axis=-1),
            samplerate=self.sample_rate)
        
        return None

    
            
class VoiceHandler():
    def __init__(
        self,
        device_idx=None,
        sample_rate=22050,
        sample_channels=2,
        sample_width=2,
        chunk_size=1024,
        voice_threshold=0.03,
        start_secs=0.10,
        end_secs=0.80,
        start_buffer_secs=0.50,
        max_rec_secs=10,
        voice_language='es-AR',
        verbose=True):

        self.device_idx = device_idx
        self.voice_language = voice_language

        
        if sample_rate in [22050, 44100]:
            self.sample_rate = sample_rate
        else:
            raise ValueError('sample_rate, please use: 44100')
        
        if sample_channels in [1, 2]:
            self.sample_channels = sample_channels
        else:
            raise ValueError('sample_channels, please use: 1 or 2')

        if chunk_size in [128, 256, 512, 1024]:
            self.chunk_size = chunk_size
        else:
            raise ValueError('chunk_size, please use: 128, 256, 512 or 1024')


        self.sample_width = sample_width
        if sample_width == 2:
            self.sample_format_pa = pyaudio.paInt16
            self.sample_format_np = np.int16
            
        elif sample_width == 4:
            self.sample_format_pa = pyaudio.paInt32
            self.sample_format_np = np.int32
            
        else:
            raise ValueError(' sample_width, please use: 2 or 4')

        assert start_buffer_secs >= start_secs, 'ERROR, start_buffer_secs < start_frames.'
        self.sample_norm = np.iinfo(self.sample_format_np).max

        self.start_frames        = max( int(round(       start_secs / self.chunk_size * self.sample_rate)), 1 )
        self.end_frames          = max( int(round(         end_secs / self.chunk_size * self.sample_rate)), 1 )
        self.start_buffer_frames = max( int(round(start_buffer_secs / self.chunk_size * self.sample_rate)), 1 )

        self.R = None
        self.verbose = verbose
        self.pa = pyaudio.PyAudio()

        if voice_threshold is None:
            self.autoset_voice_threshold()
        else:
            self.voice_threshold = voice_threshold
            
        return None

        
    def config_device_idx(self, only_input_dev=True, do_print=True):
        dev_v = []

        if do_print:
            s = '{:5s} | {:64s} | {:6s} | {:7s}'.format('Index', 'Name', 'Inputs', 'Outputs')
            print(s)
            print('-'*len(s) )

        pos_devs = [None]
        for device_idx in range(self.pa.get_device_count()):
            dev_d = self.pa.get_device_info_by_index(device_idx)
            if only_input_dev and dev_d['maxInputChannels'] == 0:
                continue
            
            dev_v.append(dev_v)
            if do_print:
                print('{index:5d} | {name:64s} | {maxInputChannels:6d} | {maxOutputChannels:7d}'.format(**dev_d))
                pos_devs.append(dev_d['index'])
                
        if do_print and only_input_dev:
            print(' (*) Only shown input devices.')


        print(' - Device index:', self.device_idx)
        device_idx = -1
        while not device_idx in pos_devs:
            try:
                r = input(' Enter device index (leave blank for system default device) >>> ')
                if r == '':
                    device_idx = None
                else:
                    device_idx =int(r)
            except:
                continue

        if device_idx != self.device_idx:
            self.device_idx = device_idx
            print(' - Device index configured to:', self.device_idx)
            
        return dev_v


    def terminate_pa_instance(self):
        
        # Terminate the PortAudio interface
        self.pa.terminate()

        if self.verbose:
            print(' PyAudio Termiated!')
            
        return 

        
    def __del__(self):
        self.terminate_pa_instance()
        return None

    
    def rec_fix_len(self, seconds=3):
        if self.verbose:
            print('Start Recording ...')

        stream = self.pa.open(
            input_device_index=self.device_idx,
            format=self.sample_format_pa,
            channels=self.sample_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True)
        

        frames = []  # Initialize array to store frames
        try:
            for i in range(int(self.sample_rate / self.chunk_size * seconds)):
                data = stream.read(self.chunk_size)
                frames.append(data)
##                print( self.detect_voice(data) )

            stream.stop_stream()
            
        except Exception as e:
            print(' WARNING:', e, file=sys.stderr)
        
        stream.close()

        if self.verbose:
            print('Finished recording !')


        frame_data = b''.join(frames)
        ad = AudioData(frame_data, self.sample_rate, self.sample_width, self.sample_channels)
        return ad

    def rec_speech(self):
        start_buffer = deque(maxlen=self.start_buffer_frames)
        start_flags  = deque([False], maxlen=self.start_frames)
        end_flags    = deque([False], maxlen=self.end_frames)
        
        on_rec = False
        
        stream = self.pa.open(
            input_device_index=self.device_idx,
            format=self.sample_format_pa,
            channels=self.sample_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True)

        if self.verbose:
            print('Please talk ...')

        frames = []  # Initialize array to store frames
        try:
            while True:
                data = stream.read(self.chunk_size)
                detected_voice = self.detect_voice(data)
                
                
                if on_rec:
                    frames.append(data)
                    end_flags.append(not detected_voice)

##                    print(len(frames), 'end_flags:', end_flags)
                    
                    if all(end_flags):
                        break

                else:
                    start_buffer.append(data)
                    start_flags.append( detected_voice )

                    if all(start_flags):
                        on_rec = True
                        frames = list(start_buffer)
                    

            stream.stop_stream()
            
        except Exception as e:
            print(' WARNING:', e, file=sys.stderr)
        
        stream.close()

        if self.verbose:
            print('Finished recording !')

        frame_data = b''.join(frames)

        ad = AudioData(frame_data, self.sample_rate, self.sample_width, self.sample_channels)
        return ad


    
    def _frame2wave(self, frame_bytes):            
        w = np.frombuffer(frame_bytes, dtype=self.sample_format_np) / self.sample_norm
        w_v =  w.reshape(-1, self.sample_channels).T
        return w_v
    
    def detect_voice(self, frame_bytes, chennel=0):
        w_v = self._frame2wave(frame_bytes)
        has_voice = (np.abs(w_v[chennel]) > self.voice_threshold).mean() > 0.1
        return has_voice

    
    def plot_audio_data(self, audio_data):
        audio_data.plot(self.voice_threshold)
        return None


    def STT(self, audio_data):
        if self.R is None:
            self.R = sr.Recognizer()

        if audio_data.sample_channels !=1:
           audio_data = audio_data.to_mono() 
            
        text = self.R.recognize_google(audio_data, language=self.voice_language)
        return text

    def rec_speech2text(self, return_audio_data=False):
        audio_data = self.rec_speech()

        try:
            text = self.STT(audio_data)
            
        except Exception as e:
            print(' - WARNING: Voice Recognition ERROR.', file=sys.stderr)
            text = ''

        if return_audio_data:
            return text, audio_data
        else:
            return text


    def autoset_voice_threshold(self, audio_len=5, th_detector='mean', do_plot=True, verbose=True):
        if verbose:
            input('Please press ENTER and do not speake for {} secs ... '.format(audio_len))
            
        ad = self.rec_fix_len(audio_len)
        w = ad.get_signal()

        if th_detector=='mean':
            voice_threshold = np.abs(w).mean() + 10*w.std()
        elif th_detector=='max':
            voice_threshold = 2*np.abs(w).max()
        else:
            raise ValueError('th_detector must be mean or max.')

        self.voice_threshold = voice_threshold

        if verbose:
            print(' - voice_threshold set to:', self.voice_threshold)

        if do_plot:
            self.plot_audio_data(ad)
            
        return None
    
    
if __name__ == '__main__':
    sys.path.append('../Acapela_TTS_API')
    sys.path.append('../Mitsuku')
    from Acapela_TTS_API import Acapela_API
    from mitsuku import PandoraBot

    use_acapela_mk = False

    if use_acapela_mk:
        mk = PandoraBot(bot_name="David", is_male=True, bot_lang='es', verbose=False)
        acapela = Acapela_API(verbose=False)
        acapela.login(credentials_path='../Acapela_TTS_API/acapela_credentials.json')
    
    vr = VoiceHandler(voice_threshold=None)
    vr.config_device_idx()
    
    while True:
        text, ad = vr.rec_speech2text(return_audio_data=True)
        if text != '':
            print(' >>> ', text)

            if use_acapela_mk:
                resp_text = mk.query(text)
                print(' <<< ', resp_text, '\n')
                acapela.speak_and_download_wav(resp_text, convert_to_ogg=False)
                acapela.play()
            else:
                ad.play()
        
    vr.plot_audio_data(ad)
    

### REC
##chunk = 1024  # Record in chunks of 1024 samples
##sample_format = pyaudio.paInt16  # 16 bits per sample
##channels = 2
##fs = 22050  # Record at 44100 samples per second
##seconds = 3
##filename = "output.wav"
##
##p = pyaudio.PyAudio()  # Create an interface to PortAudio
##
##print('Recording')
##
##stream = p.open(format=sample_format,
##                channels=channels,
##                rate=fs,
##                frames_per_buffer=chunk,
##                input=True)
##
##frames = []  # Initialize array to store frames
##
### Store data in chunks for 3 seconds
##for i in range(0, int(fs / chunk * seconds)):
##    data = stream.read(chunk)
##    frames.append(data)
##
### Stop and close the stream 
##stream.stop_stream()
##stream.close()
### Terminate the PortAudio interface
##p.terminate()
##
##print('Finished recording')
##
### Save the recorded data as a WAV file
##wf = wave.open(filename, 'wb')
##wf.setnchannels(channels)
##wf.setsampwidth(p.get_sample_size(sample_format))
##wf.setframerate(fs)
##wf.writeframes(b''.join(frames))
##wf.close()
##
##
##
##
##
##
### PLAY
##import pyaudio
##import wave
##
##filename = 'myfile.wav'
##
### Set chunk size of 1024 samples per data frame
##chunk = 1024  
##
### Open the sound file 
##wf = wave.open(filename, 'rb')
##
### Create an interface to PortAudio
##p = pyaudio.PyAudio()
##
### Open a .Stream object to write the WAV file to
### 'output = True' indicates that the sound will be played rather than recorded
##stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
##                channels = wf.getnchannels(),
##                rate = wf.getframerate(),
##                output = True)
##
### Read data in chunks
##data = wf.readframes(chunk)
##
### Play the sound by writing the audio data to the stream
##while data != '':
##    stream.write(data)
##    data = wf.readframes(chunk)
##
### Close and terminate the stream
##stream.close()
##p.terminate()


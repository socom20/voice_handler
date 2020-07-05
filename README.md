# Voice Handler

- This implementation is intended to record voices.
- Using a threshold, the application listens to an audio device and returns only the signal of an input voice.
- Recording stops when the person stops talking.

## Installation

Just install dependencies:
```sh
$ pip3 install speech_recognition
$ pip3 install sounddevice
$ pip3 install pyaudio
$ pip3 install wave
```
## Usage Example

```python3
# Create an instance
vr = VoiceHandler(voice_threshold=None)

# Config de input device
vr.config_device_idx()

# Config the threshold level to detect a voice. A signal plot will be displayed after the configuration.
vr.autoset_voice_threshold()

# Just detect your voice
text, ad = vr.rec_speech2text(return_audio_data=True)

# If you need to see the audio signal recorded
ad.plot()

# If you need to lesten it, just:
ad.play()
```

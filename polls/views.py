import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd
from django.http import JsonResponse

# Load the YAMNet model 
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Helper function to resample audio to 16 kHz
def resample_to_16k(audio, original_sr):
    if original_sr != 16000:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
    return audio

def index(request):
    freq = 44100  
    duration = 3  # Recording duration in seconds

    print("Starting continuous audio processing...")

    while True:
        # Record audio
        print("Recording audio...")
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype='float32')
        sd.wait()
        print("Recording complete.")

        # Convert recording to the required format
        recording = recording.flatten() 
        
        recording_16k = resample_to_16k(recording, freq)
        wav_tensor = tf.convert_to_tensor(recording_16k, dtype=tf.float32)

        # Process audio with YAMNet
        print("Processing audio with YAMNet...")
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])

        scores, embeddings, spectrogram = yamnet_model(wav_tensor)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.math.argmax(class_scores)
        inferred_class = class_names[top_class]

        print(f'The main sound is: {inferred_class}')

        # Check if the inferred class is 'dog'
        if inferred_class == "Dog":
            result = {'inferred_class': inferred_class}
            return JsonResponse(result)  # Return the result and stop the loop
        else:
            print(f"Not a dog sound. Continuing...")


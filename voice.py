import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pyaudio
import wave
import requests
import threading

# Global Değişkenler
record_thread = None
stop_recording = False
audio_emotions = {} 

audio_process_completed = threading.Event() 


# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Load the pre-trained model and tokenizer for text emotion detection
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model_text = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Set your Hugging Face token here
HF_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def detect_emotions_text(text):
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    output = model_text(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = torch.softmax(torch.from_numpy(scores), dim=0)
    scores = scores.tolist()

    labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
    emotion_probabilities = {label: score for label, score in zip(labels, scores)}
    return emotion_probabilities

def transcribe_audio(audio_file):
    try:
        response = requests.post("https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h",
                                 headers=headers,
                                 data=audio_file.read(),
                                 timeout=10)
        response.raise_for_status()
        return response.json()["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def record_and_analyze():
    global record_thread, stop_recording
    global audio_emotions, audio_process_completed
    
    stop_recording = False 

    # Record audio from microphone
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    WAVE_OUTPUT_FILE = "output.wav"

    p = pyaudio.PyAudio()

    

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    try:
        while not stop_recording:
            data = stream.read(CHUNK)
            frames.append(data)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Transcribing audio...")
    audio_file = open(WAVE_OUTPUT_FILE, "rb")
    transcription = transcribe_audio(audio_file)
    audio_file.close()

    print("Detecting emotions from text...")
    audio_emotions = detect_emotions_text(transcription)  

    print("Audio Emotions Received:", audio_emotions)

    audio_process_completed.set()

    stop_recording = True
    audio_process_completed.clear()


def stop_recording_thread():
    global stop_recording
    stop_recording = True

def display_emotions(emotions):
    print("\nEmotion Probabilities:")
    for emotion, probability in emotions.items():
        print(f"{emotion}: {probability * 100:.2f}%")

if __name__ == "__main__":
    record_and_analyze()

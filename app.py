from flask import Flask, render_template, Response, jsonify

import threading
import cv2
from voice import record_and_analyze, stop_recording_thread
from voice import audio_process_completed
from face import detect_emotions
from PIL import Image

app = Flask(__name__, static_folder="./static")

is_camera_active = False
audio_thread = None  
face_emotions = None

 

# Camera stream generator function
def gen_frames():
    global is_camera_active
    global face_emotions
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if is_camera_active:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face, class_probabilities = detect_emotions(image)
                face_emotions = class_probabilities

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                face_emotions = None  # Kamera durdurulduğunda sıfırlanmalı
                img = cv2.imread('camera.png')  # Load placeholder image
                ret, buffer = cv2.imencode('.png', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global is_camera_active 
    is_camera_active = True
    return "Camera started"

@app.route('/stop_camera')
def stop_camera():
    global is_camera_active 
    is_camera_active = False
    return "Camera stopped"

@app.route('/start_audio')
def start_audio():
    global audio_thread
    if audio_thread is None or not audio_thread.is_alive():
        audio_thread = threading.Thread(target=record_and_analyze)
        audio_thread.start()
        return "Audio analysis started"
    else:
        return "Audio analysis already running"

@app.route('/stop_audio')
def stop_audio():
    global audio_thread
    if audio_thread and audio_thread.is_alive():
        stop_recording_thread()
        audio_thread.join()
        audio_emotions = {}  
        return "Audio analysis stopped"
    else:
        return "No audio analysis running"

@app.route('/audio_results')
def audio_results():
    from voice import audio_emotions  
    return jsonify(audio_emotions)

@app.route('/audio_process_status')
def audio_process_status():
    return jsonify({'is_completed': audio_process_completed.is_set()})

@app.route('/video_results')
def video_results():
    global face_emotions
    return jsonify(face_emotions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
else:
    print("This script is not meant to be imported as a module.")

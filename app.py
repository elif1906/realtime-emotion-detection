from flask import Flask, render_template, Response
from flask import jsonify
import threading
import cv2
import face  
from voice import record_and_analyze
from face import face_and_analyze
from voice import stop_recording_thread
app = Flask(__name__)

is_camera_active = False
audio_thread = None  
text_emotions = None
face_emotions = None

# Camera stream generator function
def gen_frames():
    global is_camera_active
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if is_camera_active:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                img = cv2.imread('camera.png')  # Load placeholder image
                ret, buffer = cv2.imencode('.png', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start camera endpoint
@app.route('/start_camera')
def start_camera():
    global is_camera_active 
    is_camera_active = True
    return "Camera started"

# Stop camera endpoint
@app.route('/stop_camera')
def stop_camera():
    global is_camera_active 
    is_camera_active = False
    return "Camera stopped"

# Start audio analysis endpoint
@app.route('/start_audio')
def start_audio():
    global audio_thread
    if audio_thread is None or not audio_thread.is_alive():
        audio_thread = threading.Thread(target=record_and_analyze)
        audio_thread.start()
        return "Audio analysis started"
    else:
        return "Audio analysis already running"

# Stop audio analysis endpoint
@app.route('/stop_audio')
def stop_audio():
    global audio_thread
    if audio_thread and audio_thread.is_alive():
        stop_recording_thread()
        audio_thread.join()
        return "Audio analysis stopped"
    else:
        return "No audio analysis running"
    
@app.route('/audio_results')
def audio_results():
    global text_emotions
    return jsonify(text_emotions)

@app.route('/video_results')
def video_results():
    global text_emotions
    return jsonify(text_emotions)    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
else:
    print("This script is not meant to be imported as a module.")
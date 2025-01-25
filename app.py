from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify , Response
import os
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
OUTPUT_FOLDER = 'output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

model = YOLO('yolov8s.pt')  # Load YOLOv8 model

# Tracking processing status and progress
processing_status = {}

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle file upload
@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        
        # If no file is selected
        if file.filename == '':
            return redirect(request.url)
        
        # If the file is allowed, save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Start the video processing in a separate thread
            threading.Thread(target=process_video, args=(filepath, filename)).start()
            # Track processing status
            processing_status[filename] = {'status': 'processing', 'progress': 0}
            return redirect(url_for('processing_page', filename=filename))
    return render_template('upload.html')

# Video processing function (YOLO detection & live stream)
def process_video(filepath, filename):
    # Open video and get frame properties
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    # Get the original video frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Generate output filename based on uploaded filename
    output_filename = f"processed_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Define codec (use 'mp4v' for MP4 container) and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))  # Use original resolution

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)
        annotated_frame = results[0].plot()

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # Update progress
        processed_frames += 1
        progress = (processed_frames / total_frames) * 100
        processing_status[filename]['progress'] = progress

    cap.release()
    out.release()  # Ensure the video file is properly saved

    # Update processing status to 'complete' and store output filename
    processing_status[filename]['status'] = 'complete'
    processing_status[filename]['output_filename'] = output_filename

# Route to show the live processing page
@app.route('/processing/<filename>')
def processing_page(filename):
    # Check the processing status
    status_info = processing_status.get(filename, {'status': 'processing', 'progress': 0})
    return render_template('processing.html', filename=filename, status=status_info['status'], progress=status_info['progress'])

# Route to fetch progress in JSON format (for AJAX)
@app.route('/progress/<filename>')
def get_progress(filename):
    status_info = processing_status.get(filename, {'status': 'unknown', 'progress': 0})
    return jsonify(status=status_info['status'], progress=status_info['progress'])

# Route to stream the processed video (live detection)
def generate_video_stream(filepath):
    cap = cv2.VideoCapture(filepath)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform YOLO detection on the frame
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG and send it to the frontend
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_video_stream(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Background task to delete the file after download
def delete_file_after_download(file_path):
    os.remove(file_path)
    print(f"Processed file {file_path} deleted after download.")

# Route to download the processed video
@app.route('/download/<filename>')
def download_video(filename):
    # Construct the processed filename based on the original filename
    output_filename = f"processed_{filename}"  # Use the same name format as during processing
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    if os.path.exists(output_path):
        # Serve the file for download
        response = send_file(output_path, as_attachment=True)
        
        # Start a background thread to delete the file after download
        threading.Thread(target=delete_file_after_download, args=(output_path,)).start()
        
        return response
    
    return "File not found or processing not completed yet.", 404

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run(debug=True)

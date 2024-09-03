# app.py

from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                # No faces detected: Add warning text to the frame
                warning_text = "Warning: No face detected!"
                cv2.putText(
                    frame,
                    warning_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            else:
                # Draw rectangles around detected faces
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte array
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)

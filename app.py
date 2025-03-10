from flask import Flask, request, jsonify, send_from_directory
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import os
import cv2
from processEmotion import doIt
from email.mime.base import MIMEBase
from email import encoders




app = Flask(__name__)


# Function to send email
def send_email(email):
    sender_email = "phoenix00100gaming@gmail.com"
    app_password = "zxpt jncd pnse mfbm"  # Use an App Password, NOT your Gmail password
    pdf_path = "./EmotionRecogniation/pdfs/output.pdf"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = email
    message["Subject"] = "Video Emotion Recognition Result"

    body = "Emotion Summary:\n\nYour video emotion analysis result is attached."
    message.attach(MIMEText(body, "plain"))

    try:
        with open(pdf_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(pdf_path)}",
        )
        message.attach(part)
    except Exception as e:
        print(f"Failed to attach file: {e}")
        return

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)  # Authenticate using App Password
        text = message.as_string()
        server.sendmail(sender_email, email, text)
        server.quit()
        print(f"Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# # Background task to process video and send email
def background_task(video_path,email):
    cap = cv2.VideoCapture(video_path)
    doIt()
    send_email(email)
    cap.release()  # Release file before deleting
    cv2.destroyAllWindows()
    os.remove(video_path) 
    os.remove("./EmotionRecogniation/pdfs/output.pdf")


@app.route('/')
def index():
    send_email()
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_video():
    # Check if video file is provided
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400

    # Get the email from the form data
    email = request.form.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Save the video file
    video_file = request.files['video']
    video_path = os.path.join("./EmotionRecogniation", "sample.mp4")
    video_file.save(video_path)

    # # Start a background thread to process the video
    thread = threading.Thread(target=background_task, args=(video_path,email))
    thread.start()

    return jsonify({"message": "Processing started. You can close the tab, the server will continue working.Mail will be send to with result."})

@app.route('/check_email', methods=['GET'])
def check_email():
    # This endpoint can be used to check if the email was sent
    return jsonify({"message": "\nEmail has been sent."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned PORT or default to 10000
    app.run(host='0.0.0.0', port=port, debug=True)
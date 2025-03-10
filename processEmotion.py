import cv2
import os
import pandas as pd
import numpy as np
import torch
import joblib
import librosa
import whisper
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from deepface import DeepFace
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit

def one():
  video_path = "./EmotionRecogniation/sample.mp4"

  output_dir = "./EmotionRecogniation/Frames"

  def extract_frames(video_path, output_dir, interval=10):
      cap = cv2.VideoCapture(video_path)
      fps = cap.get(cv2.CAP_PROP_FPS)
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      duration = total_frames / fps

      frame_count = 0
      for i in range(0, int(duration), interval):
          cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
          ret, frame = cap.read()
          if ret:
              frame_path = os.path.join(output_dir, f"frame_{i//interval}.jpg")
              cv2.imwrite(frame_path, frame)
              frame_count += 1
      cap.release()
      print(f"Extracted {frame_count} frames at {interval}-second intervals.")

  extract_frames(video_path, output_dir, interval=10)



def two():
  frame_dir = "./EmotionRecogniation/Frames"

  emotion_output_dir = "./EmotionRecogniation/FrameEmotionResult"

  def analyze_emotions(frame_dir, output_dir):
      results = []

      for frame_name in os.listdir(frame_dir):
          frame_path = os.path.join(frame_dir, frame_name)

          try:
              analysis = DeepFace.analyze(
                  img_path=frame_path,
                  actions=['emotion'],
                  detector_backend='opencv',
                  enforce_detection=False
              )


              emotion = analysis[0]['dominant_emotion']
              confidence = analysis[0]['emotion'][emotion]
              results.append({
                  "frame": frame_name,
                  "emotion": emotion,
                  "confidence": confidence / 100.0000
              })

              print(f"Processed {frame_name}: {emotion} ({confidence:.2f}%)")

          except Exception as e:
              print(f"Failed to process {frame_name}: {str(e)}")


      df = pd.DataFrame(results)
      df.to_csv(os.path.join(output_dir, "emotion_results.csv"), index=False)
      return df


  results_df = analyze_emotions(frame_dir, emotion_output_dir)



def three():

  video_path = "./EmotionRecogniation/sample.mp4"
  audio_output_dir = "./EmotionRecogniation/audio_segments"
  os.makedirs(audio_output_dir, exist_ok=True)


  video = VideoFileClip(video_path)
  audio = video.audio
  audio.write_audiofile(os.path.join(audio_output_dir, "full_audio.wav"))


  full_audio = AudioSegment.from_wav(os.path.join(audio_output_dir, "full_audio.wav"))
  segment_length = 10 * 1000

  for i, start_time in enumerate(range(0, len(full_audio), segment_length)):
      end_time = start_time + segment_length
      chunk = full_audio[start_time:end_time]
      chunk.export(os.path.join(audio_output_dir, f"segment_{i}.wav"), format="wav")



  model = whisper.load_model("base")

  transcriptions = []
  for file in sorted(os.listdir(audio_output_dir)):
      if file.startswith("segment_"):
          segment_path = os.path.join(audio_output_dir, file)
          result = model.transcribe(segment_path, fp16=False)
          transcript = result["text"].strip()

          segment_id = int(file.split("_")[1].split(".")[0])
          start_time = segment_id * 10
          end_time = start_time + 10

          transcriptions.append({
              "segment": file,
              "start_time": start_time,
              "end_time": end_time,
              "text": transcript
          })


  df = pd.DataFrame(transcriptions)
  df.to_csv("./EmotionRecogniation/transcriptions.csv", index=False)
  print(df.head())

def four():
  transcriptions_path = "./EmotionRecogniation/transcriptions.csv"
  transcriptions_df = pd.read_csv(transcriptions_path)


  classifier = pipeline(
      "text-classification",
      model="j-hartmann/emotion-english-distilroberta-base",
      return_all_scores=False,
      device=0 if torch.cuda.is_available() else -1
  )


  EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


  def analyze_text_emotions(df):
      results = []

      for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Text"):
          text = row["text"]
          if not text.strip():
              results.append({
                  "segment": row["segment"],
                  "start_time": row["start_time"],
                  "text_emotion": "neutral",
                  "confidence": 1.0
              })
              continue


          result = classifier(text, truncation=True, max_length=512)
          emotion = result[0]['label'].lower()
          confidence = result[0]['score']

          if emotion == "joy":
              emotion = "happy"
          elif emotion == "sadness":
              emotion = "sad"

          results.append({
              "segment": row["segment"],
              "start_time": row["start_time"],
              "text_emotion": emotion,
              "confidence": confidence
          })

      return pd.DataFrame(results)

  text_emotion_df = analyze_text_emotions(transcriptions_df)

  output_path = "./EmotionRecogniation/text_emotion_results.csv"
  text_emotion_df.to_csv(output_path, index=False)
  print("Text emotion results saved!")


def five():

  model_name = "superb/wav2vec2-base-superb-er"
  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
  model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

  label_map = {
      "angry": "angry",
      "disgust": "disgusted",
      "fear": "fearful",
      "happy": "happy",
      "neutral": "neutral",
      "sad": "sad",
      "surprise": "surprise"
  }

  def analyze_audio_emotions(audio_segment_dir):
      results = []
      audio_files = sorted([f for f in os.listdir(audio_segment_dir) if f.startswith("segment_")])

      for file in tqdm(audio_files, desc="Processing Audio Segments"):
          segment_path = os.path.join(audio_segment_dir, file)

          try:

              audio, sr = librosa.load(segment_path, sr=16000)

              inputs = feature_extractor(
                  audio,
                  sampling_rate=16000,
                  return_tensors="pt",
                  padding=True
              )


              with torch.no_grad():
                  outputs = model(**inputs)


              predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
              emotion_id = predictions.argmax().item()
              emotion = model.config.id2label[emotion_id].lower()
              confidence = predictions[0][emotion_id].item()

              emotion = label_map.get(emotion, "neutral")

              results.append({
                  "segment": file,
                  "start_time": int(file.split("_")[1].split(".")[0]) * 10,
                  "audio_emotion": emotion,
                  "confidence": confidence
              })

          except Exception as e:
              print(f"Error processing {file}: {str(e)}")
              results.append({
                  "segment": file,
                  "start_time": int(file.split("_")[1].split(".")[0]) * 10,
                  "audio_emotion": "neutral",
                  "confidence": 0.0
              })

      return pd.DataFrame(results)


  audio_emotion_df = analyze_audio_emotions("./EmotionRecogniation/audio_segments")


  output_path = "./EmotionRecogniation/audio_emotion_results.csv"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
  audio_emotion_df.to_csv(output_path, index=False)
  print(f"\n✅ Results saved to: {output_path}")




def six():

  # --------------------------
  # 1. Configuration
  # --------------------------
  # PATHS = {
  #     'face_data': "./EmotionRecogniation/FrameEmotionResult/emotion_results.csv",
  #     'audio_data': "./EmotionRecogniation/audio_emotion_results.csv",
  #     'text_data': "./EmotionRecogniation/text_emotion_results.csv",
  #     'output': "./EmotionRecogniation/final_emotion_results.csv"
  # }

  # --------------------------
  # 1. Configuration
  # --------------------------
  PATHS = {
      'face_data': "./EmotionRecogniation/FrameEmotionResult/emotion_results.csv",
      'audio_data': "./EmotionRecogniation/audio_emotion_results.csv",
      'text_data': "./EmotionRecogniation/text_emotion_results.csv",
      'output': "./EmotionRecogniation/final_emotion_results.csv",
      'transcriptions': "./EmotionRecogniation/transcriptions.csv",
      'frames_folder': "./EmotionRecogniation/Frames",
      'plot_path': "./EmotionRecogniation/timeline_plot.png",
      'pdf_folder': "./EmotionRecogniation/pdfs",
      'pdf_filename': "output.pdf"
  }

  # --------------------------
  # 2. Data Processing Functions
  # --------------------------
  def get_final_emotion(row):
      """Determine final emotion based on majority of maximum confidence levels"""
      emotions = ["face", "text", "audio"]
      confidences = [row[f"confidence_{m}"] for m in emotions]
      max_conf = max(confidences)
      max_indices = [i for i, conf in enumerate(confidences) if conf == max_conf]
      if len(max_indices) > 1:
          # If there's a tie, choose the first one (or any other logic you prefer)
          return row[f"{emotions[max_indices[0]]}_emotion"]
      else:
          return row[f"{emotions[max_indices[0]]}_emotion"]

  def load_and_merge_data(face_path, audio_path, text_path):
      """Load and merge data from face, audio, and text emotion files"""
      # Load raw data
      face_data = pd.read_csv(face_path)
      audio_data = pd.read_csv(audio_path)
      text_data = pd.read_csv(text_path)

      # Process face data
      face_data['start_time'] = face_data['frame'].str.extract(r'frame_(\d+)').astype(int) * 10
      face_data = face_data.rename(columns={
          'emotion': 'face_emotion',
          'confidence': 'confidence_face'
      })

      # Process audio/text data
      audio_data = audio_data.rename(columns={
          'audio_emotion': 'audio_emotion',
          'confidence': 'confidence_audio'
      })
      text_data = text_data.rename(columns={
          'text_emotion': 'text_emotion',
          'confidence': 'confidence_text'
      })

      # Debugging: Print start_time values
      print("Face Data Start Time:", face_data["start_time"].head())
      print("Text Data Start Time:", text_data["start_time"].head())

      # Merge datasets without strict timestamp alignment
      merged_df = face_data.merge(
          text_data[['start_time', 'text_emotion', 'confidence_text']],
          on='start_time',
          how='outer'
      ).merge(
          audio_data[['start_time', 'audio_emotion', 'confidence_audio']],
          on='start_time',
          how='outer'
      )

      # Fill missing values
      return merged_df.fillna({
          'face_emotion': 'neutral',
          'text_emotion': 'neutral',
          'audio_emotion': 'neutral',
          'confidence_face': 0.5,
          'confidence_text': 0.5,
          'confidence_audio': 0.5
      })

  # --------------------------
  # 3. Timeline Plot Function
  # --------------------------
  def create_timeline_plot(merged_df, plot_path):
      """Create and save emotion timeline visualization"""
      plt.figure(figsize=(10, 6))
      sns.set_theme(style="whitegrid")

      unique_emotions = merged_df["final_emotion"].unique()
      palette = sns.color_palette("husl", len(unique_emotions))

      for i, emotion in enumerate(unique_emotions):
          subset = merged_df[merged_df["final_emotion"] == emotion]
          plt.scatter(
              subset["start_time"],
              [emotion] * len(subset),
              color=palette[i],
              label=emotion,
              s=100,
              alpha=0.7,
              edgecolors="black"
          )

      plt.plot(merged_df["start_time"], merged_df["final_emotion"],
              linestyle='dotted', alpha=0.5, color="gray")

      plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
      plt.xlabel("Time (seconds)")
      plt.ylabel("Emotion")
      plt.title("Final Emotion Timeline", fontsize=14, fontweight="bold")
      plt.xticks(rotation=45)
      plt.grid(axis="x", linestyle="--", alpha=0.7)
      plt.tight_layout()
      # Save plot instead of showing
      plt.savefig(plot_path, bbox_inches='tight')
      plt.show()
      return plot_path

  # --------------------------
  # 4. PDF Creation Function
  # --------------------------
  def create_pdf(output_folder, filename, data, image_folder, plot_image_path=None):
      """Create PDF with timeline plot as first page"""
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)

      pdf_path = os.path.join(output_folder, filename)
      c = canvas.Canvas(pdf_path, pagesize=letter)
      width, height = letter

      # Add timeline plot as first page
      if plot_image_path and os.path.exists(plot_image_path):
          # Center the plot on the page
          c.drawImage(plot_image_path, 50, height-650, width=500, height=500)
          c.showPage()

      # Add individual frame pages
      for item in data:
          image_name, text1, text2 = item
          image_path = os.path.join(image_folder, image_name)

          # Add frame image
          if os.path.exists(image_path):
              c.drawImage(image_path, 50, height-250, width=200, height=200)

          # Add text content
          c.setFont("Helvetica", 12)
          wrapped_text = simpleSplit(text1, "Helvetica", 12, width-300)
          y_pos = height-270

          for line in wrapped_text:
              c.drawString(300, y_pos, line)
              y_pos -= 20

          c.drawString(300, y_pos-20, f"Emotion: {text2}")
          c.showPage()

      c.save()
      print(f"PDF saved at: {pdf_path}")

  # --------------------------
  # 5. Execution Flow
  # --------------------------
  # Step 1: Load and merge data
  merged_data = load_and_merge_data(
      PATHS['face_data'],
      PATHS['audio_data'],
      PATHS['text_data']
  )

  # Step 2: Determine final emotion
  merged_data['final_emotion'] = merged_data.apply(get_final_emotion, axis=1)

  # Step 3: Save final results
  os.makedirs(os.path.dirname(PATHS['output']), exist_ok=True)  # Ensure directory exists
  merged_data.to_csv(PATHS['output'], index=False)
  print(f"\n✅ Results saved to: {PATHS['output']}")

  # Step 4: Create timeline plot
  plot_path = PATHS['plot_path']
  create_timeline_plot(merged_data, plot_path)

  # Step 5: Prepare data for PDF creation
  text = pd.read_csv(PATHS['transcriptions'])["text"]
  final = merged_data[["frame", "final_emotion"]]
  data = list(zip(final["frame"], text, final["final_emotion"]))

  # Step 6: Create PDF
  create_pdf(
      PATHS['pdf_folder'],
      PATHS['pdf_filename'],
      data,
      PATHS['frames_folder'],
      plot_image_path=plot_path
  )

def doIt():
    one()
    two()
    three()
    four()
    five()
    six()


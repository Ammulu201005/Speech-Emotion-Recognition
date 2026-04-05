# Speech-Emotion-Recognition

🎧 Speech Emotion Recognition (SER) System
📌 Overview

This project is a Speech Emotion Recognition (SER) system that analyzes audio input (from file, microphone, or URL) and predicts the emotional state of the speaker.

It uses Machine Learning / Deep Learning models to classify emotions like:

😊 Happy
😌Calm
😢 Sad
😡 Angry
😐 Neutral
😱 Fear
🤮Disgust
😍 Surprise

The system is built with an interactive Streamlit web interface for easy usage.

🎯 Objective

The main goal of this project is to:

Detect human emotions from speech signals
Provide real-time or uploaded audio analysis
Visualize emotional variations across audio
🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
Libraries Used:
NumPy
Librosa (audio processing)
TensorFlow 
Faster whisper
yt-dlp (for audio extraction from URLs)

⚙️ Features
🎤 Live Audio Recording
📁 Upload Audio Files
🔗 Analyze Audio from URL (YouTube, etc.)
📊 Emotion Visualization (chunk-based analysis)
⚡ Fast and Stable Analyzer Mode
🔍 Segment-wise Emotion Detection (chunks)

📂 Project Structure
speech-emotion-recognition/
│
├── app/
│   ├── app.py                # Main Streamlit app
│   ├── model/               # Trained model files            
│   └── utils/              # Helper functions
│
├── data/                    # Dataset (if included)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── notebooks/               # Training notebooks (optional)


🧠 How It Works
Audio is collected from:
Upload / Mic / URL
Audio is processed,
Feature extraction (MFCC, etc.)
Audio is split into chunks
Each chunk is passed to the trained model
Emotions are predicted
Results are displayed with visualization


🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv

Activate it:

Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Fix Common Dependency (IMPORTANT ⚠️)

If you face this error:

ModuleNotFoundError: No module named 'yt_dlp'

Run:

pip install yt-dlp
▶️ Execution Steps (VERY IMPORTANT)

Follow these steps exactly to run the project:

Step 1: Go to project folder
cd speech-emotion-recognition
Step 2: Run the Streamlit App
streamlit run app/app.py
Step 3: Open in Browser

After running, you will see something like:

Local URL: http://localhost:8501

Open it in your browser.

Step 4: Use the Application

You can now:

Upload an audio file 🎵
Record live audio 🎤
Paste a YouTube/audio URL 🔗
Step 5: View Results
Emotion prediction will be shown
Graphs will display emotion variations
Chunk-based analysis will be visible

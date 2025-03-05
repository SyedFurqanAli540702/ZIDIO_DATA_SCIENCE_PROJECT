import cv2
import numpy as np
import librosa
import sounddevice as sd
from transformers import pipeline
from fer import FER  # FER is a library for facial emotion detection
import joblib  # For loading pre-trained speech emotion model

# Load pre-trained Hugging Face sentiment analysis pipeline for text
text_analyzer = pipeline("sentiment-analysis")

# Load pre-trained OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the FER detector
face_emotion_detector = FER()

# Function to analyze emotion from text
def analyze_text_emotion(text):
    """
    Analyze the emotion from the given text using Hugging Face pipeline.
    """
    result = text_analyzer(text)
    return result[0]['label'], result[0]['score']

# Function to analyze facial expression using FER
def analyze_facial_expression(frame):
    """
    Analyze facial expressions using FER.
    """
    # Detect emotions from the frame
    emotion, score = face_emotion_detector.top_emotion(frame)
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return emotion

# Function to analyze emotion from speech
def analyze_speech_emotion(duration=3, sampling_rate=16000):
    """
    Analyze speech emotion from microphone input.
    """
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete!")

    # Extract features for emotion detection (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sampling_rate, n_mfcc=40)
    # Use a trained model to predict emotion (this should be a trained model file, e.g., "speech_emotion_model.pkl")
    # For this example, we're simplifying and using a random emotion, replace with a model for better predictions
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
    emotion = np.random.choice(emotions)  # Replace with real model prediction
    return emotion

# Main function to call all emotion detection features
def main():
    # Example usage for text emotion detection
    text = input("Enter text for emotion analysis: ")
    text_emotion, confidence = analyze_text_emotion(text)
    print(f"Text Emotion: {text_emotion} with confidence {confidence:.2f}")
    
    # Example usage for facial emotion detection using webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        emotion = analyze_facial_expression(frame)
        cv2.imshow('Emotion Detection - Press Q to Quit', frame)
        print(f"Facial Emotion: {emotion}")

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Example usage for speech emotion detection
    speech_emotion = analyze_speech_emotion()
    print(f"Speech Emotion: {speech_emotion}")

# Run the main function
if __name__ == "__main__":
    main()

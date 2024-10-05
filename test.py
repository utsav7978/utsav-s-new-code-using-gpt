import openai

# Set up your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to generate a response based on the detected emotion
def get_emotional_response(emotion, health_data=None):
    prompt = f"""
    A person is feeling {emotion}. Provide a comforting message for them.
    """
    
    if health_data:
        prompt += f" Additionally, their heart rate is {health_data['heart_rate']} bpm and their oxygen level is {health_data['oxygen_level']}%. Give some health advice."

    response = openai.Completion.create(
        engine="gpt-4",  # You can use 'gpt-4' or any other available model
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

# Example use case
emotion = "Sad"
health_data = {
    'heart_rate': 72,
    'oxygen_level': 98
}

response = get_emotional_response(emotion, health_data)
print(response)


import cv2
from keras.models import load_model
import numpy as np

# Load pre-trained emotion detection model
emotion_model = load_model('emotion_detector_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize OpenCV for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)
    
    detected_emotion = None
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            prediction = emotion_model.predict(roi)
            detected_emotion = emotion_labels[np.argmax(prediction)]
            
            # Draw rectangle around face and add emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return frame, detected_emotion

# Capture emotion through webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_with_emotion, emotion = detect_emotion(frame)
    
    # If an emotion is detected, generate a response using OpenAI API
    if emotion:
        health_data = {'heart_rate': 75, 'oxygen_level': 98}
        response = get_emotional_response(emotion, health_data)
        print(f"Detected Emotion: {emotion}")
        print(f"AI Response: {response}")
    
    cv2.imshow('Emotion Detector', frame_with_emotion)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



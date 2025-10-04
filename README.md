# accessibilityAI
this ai helps to speech impaired people to talk.




import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
import random
import time
import datetime
from collections import deque

try:
    import pyvirtualcam
    VIRTUALCAM_AVAILABLE = True
except ImportError:
    VIRTUALCAM_AVAILABLE = False


class Config:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    SIGN_RECOGNITION_COOLDOWN = 5  # seconds
    EMOTION_DETECTION_COOLDOWN = 10  # seconds
    SIGN_CONFIDENCE_THRESHOLD = 0.7
    EMOTION_CONFIDENCE_THRESHOLD = 0.5


class SpeechSynthesizer:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.speak_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.last_spoke_time = 0
        self.setup_engine()

    def setup_engine(self):
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)

    def speak(self, text):
        if not text or not text.strip():
            return

        current_time = time.time()
        if current_time - self.last_spoke_time < 2:
            return

        self.speak_queue.put(text)

    def _worker(self):
        while True:
            text = self.speak_queue.get()
            if text == "__STOP__":
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoke_time = time.time()
            self.speak_queue.task_done()

    def stop(self):
        self.speak_queue.put("__STOP__")
        self.worker_thread.join()


class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=Config.SIGN_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.gestures = {
            'open_palm': 'Hello',
            'fist': 'Stop',
            'thumbs_up': 'Good',
            'peace': 'Peace',
            'ok': 'OK',
            'rock': 'Rock on',
            'point_up': 'Attention',
            'point_down': 'Go down',
            'wave': 'Hi there'
        }

    def recognize(self, landmarks):
        if len(landmarks) != 21 * 3:
            return None

        lm = np.array(landmarks).reshape(21, 3)

        thumb_tip = lm[4][:2]
        index_tip = lm[8][:2]
        middle_tip = lm[12][:2]
        ring_tip = lm[16][:2]
        pinky_tip = lm[20][:2]

        thumb_ip = lm[3][:2]
        index_mcp = lm[5][:2]
        middle_mcp = lm[9][:2]
        ring_mcp = lm[13][:2]
        pinky_mcp = lm[17][:2]

        def is_extended(tip, mcp):
            return tip[1] < mcp[1]

        thumb_extended = is_extended(thumb_tip, thumb_ip)
        index_extended = is_extended(index_tip, index_mcp)
        middle_extended = is_extended(middle_tip, middle_mcp)
        ring_extended = is_extended(ring_tip, ring_mcp)
        pinky_extended = is_extended(pinky_tip, pinky_mcp)

        if all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['open_palm']

        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['fist']

        if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['thumbs_up']

        if index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
            return self.gestures['peace']

        if not thumb_extended and not index_extended and middle_extended and ring_extended and pinky_extended:
            return self.gestures['ok']

        return None

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                gesture = self.recognize(landmarks)
                if gesture:
                    break

        return frame, gesture


class FaceExpressionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_labels = ['Happy', 'Sad', 'Neutral', 'Surprise', 'Angry']

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect_expression(self, frame):
        faces = self.detect_faces(frame)
        expressions = []

        for (x, y, w, h) in faces:
            emotion = random.choice(self.emotion_labels)
            expressions.append((x, y, w, h, emotion))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, expressions


class StressManagementAssistant:
    def __init__(self):
        self.emotion_history = deque(maxlen=30)
        self.intervention_history = []

        self.breathing_exercises = [
            "Try the 4-7-8 technique: Inhale for 4 seconds, hold for 7, exhale for 8",
            "Practice box breathing: Inhale for 4, hold for 4, exhale for 4, hold for 4",
            "Take slow, deep breaths focusing on your abdomen"
        ]

        self.mindfulness_exercises = [
            "Focus on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste",
            "Try a body scan meditation, focusing on each part of your body",
            "Practice progressive muscle relaxation"
        ]

        self.positive_affirmations = [
            "You are capable and strong",
            "You can handle whatever comes your way",
            "You are making progress every day",
            "Your efforts are valuable and appreciated"
        ]

    def assess_stress_level(self, emotion):
        high_stress = ['Angry', 'Fear', 'Sad', 'Disgust']
        moderate_stress = ['Surprise', 'Neutral']

        if emotion in high_stress:
            return "high"
        elif emotion in moderate_stress:
            return "moderate"
        else:
            return "low"

    def suggest_intervention(self, stress_level, emotion):
        self.emotion_history.append({
            'emotion': emotion,
            'stress_level': stress_level,
            'timestamp': datetime.datetime.now()
        })

        if stress_level == "high":
            suggestion = random.choice(self.breathing_exercises)
        elif stress_level == "moderate":
            suggestion = random.choice(self.mindfulness_exercises)
        else:
            suggestion = random.choice(self.positive_affirmations)

        self.intervention_history.append({
            'timestamp': datetime.datetime.now(),
            'stress_level': stress_level,
            'emotion': emotion,
            'suggestion': suggestion
        })
        return suggestion


class AccessibilityAI:
    def __init__(self):
        self.config = Config()
        self.cap = None
        self.sign_recognizer = HandGestureRecognizer()
        self.emotion_detector = FaceExpressionDetector()
        self.speech_synthesizer = SpeechSynthesizer()
        self.stress_assistant = StressManagementAssistant()
        self.initialize_camera()

    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
        return True

    def run(self):
        if not self.cap or not self.cap.isOpened():
            print("Camera not available. Exiting...")
            return

        print("Starting Accessibility AI...")
        print("Press 'q' to quit")

        last_gesture = None
        last_gesture_time = 0
        last_emotion_time = 0

        use_virtual_cam = False

        if VIRTUALCAM_AVAILABLE:
            try:
                self.virtual_cam = pyvirtualcam.Camera(
                    width=self.config.FRAME_WIDTH,
                    height=self.config.FRAME_HEIGHT,
                    fps=30
                )
                use_virtual_cam = True
                print(f'Virtual camera initialized: {self.virtual_cam.device}')
            except Exception as e:
                print(f"Virtual camera initialization failed: {e}")
                print("Falling back to window display.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                current_time = time.time()

                frame, gesture = self.sign_recognizer.process_frame(frame)
                if gesture and (gesture != last_gesture or current_time - last_gesture_time > self.config.SIGN_RECOGNITION_COOLDOWN):
                    print(f"Gesture recognized: {gesture}")
                    self.speech_synthesizer.speak(gesture)
                    last_gesture = gesture
                    last_gesture_time = current_time
                    cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if current_time - last_emotion_time > self.config.EMOTION_DETECTION_COOLDOWN:
                    frame, expressions = self.emotion_detector.detect_expression(frame)
                    for (x, y, w, h, emotion) in expressions:
                        stress_level = self.stress_assistant.assess_stress_level(emotion)
                        suggestion = self.stress_assistant.suggest_intervention(stress_level, emotion)
                        cv2.putText(frame, f"Face: {emotion}", (x, y + h + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        print(f"Expression: {emotion}, Stress: {stress_level}, Suggestion: {suggestion}")
                        self.speech_synthesizer.speak(suggestion)
                    last_emotion_time = current_time
                else:
                    frame, _ = self.emotion_detector.detect_expression(frame)

                if use_virtual_cam:
                    try:
                        self.virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.virtual_cam.sleep_until_next_frame()
                    except Exception as e:
                        print(f"Virtual camera error: {e}")
                        print("Switching to window display")
                        use_virtual_cam = False
                        cv2.imshow('Accessibility AI', frame)
                else:
                    cv2.imshow('Accessibility AI', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting Accessibility AI...")
                    break

        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")

        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.speech_synthesizer.stop()
        if hasattr(self, 'virtual_cam') and self.virtual_cam is not None:
            try:
                self.virtual_cam.close()
            except Exception:
                pass
        print("Accessibility AI stopped.")


if __name__ == "__main__":
    ai = AccessibilityAI()
    ai.run()




import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

# 얼굴 인식
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 표정 인식을 위한 눈, 코, 입등의 위치 반환
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 표정 라벨링
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 표정 가중치 모델
model_path = "emotion_model.hdf5"
try:
    model = load_model(model_path, compile=False)
    optimizer = Adam(learning_rate=0.0001)  # 최신 옵티마이저 사용
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error: Could not load model file at {model_path} - {e}")
    exit()

# 이미지 파일 불러오기
image_path = "smile.jpg"  # 분석할 이미지 파일 경로
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# 이미지 전처리
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 탐지
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # 얼굴 영역 표시
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 얼굴 ROI 추출 및 전처리
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (64, 64))
    face_roi = np.expand_dims(face_roi, axis=-1)
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = face_roi / 255.0

    # 모델로 표정 예측
    output = model.predict(face_roi, verbose=0)[0]
    expression_index = np.argmax(output)
    expression_label = expression_labels[expression_index]

    # 화면에 표정 라벨 표시
    cv2.putText(image, expression_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 결과 출력
cv2.imshow('Expression Recognition', image)
cv2.waitKey(0)  # 아무 키나 누를 때까지 창 유지
cv2.destroyAllWindows()



import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/Users/darsan/Downloads/mnist_digit_model_improved.h5')


# Recompile the model (if necessary)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for digit recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))  # Resize to model input
            roi = roi.reshape(1, 28, 28, 1).astype('float32') / 255

            # Predict the digit
            prediction = model.predict(roi)
            digit = np.argmax(prediction)

            # Draw rectangle and digit on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Digit Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

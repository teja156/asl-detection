import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the video capture and Mediapipe hands model
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Variables to store prediction and confidence for original and flipped images
    original_confidence = flipped_confidence = 0
    original_prediction = flipped_prediction = None

    # Process the original frame for drawing landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Drawing on the original frame for display
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Process both original and flipped images for predictions
    for flip in [False, True]:
        # Flip the image if required
        processed_frame = cv2.flip(frame_rgb, 1) if flip else frame_rgb

        # Hand landmarks processing (for prediction only, not drawing)
        results = hands.process(processed_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Prepare data for prediction
                data_aux = []
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                if len(x_) > 21:
                    continue

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Make prediction and calculate confidence
                probabilities = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probabilities) * 100

                # Store prediction and confidence
                if flip:
                    flipped_confidence, flipped_prediction = confidence, model.predict([np.asarray(data_aux)])[0]
                else:
                    original_confidence, original_prediction = confidence, model.predict([np.asarray(data_aux)])[0]

    # Compare confidences and choose the prediction to display
    if original_confidence > flipped_confidence:
        final_prediction = original_prediction
        final_confidence = original_confidence
    else:
        final_prediction = flipped_prediction
        final_confidence = flipped_confidence

    # Display the chosen prediction and its confidence
    text_to_display = f"{final_prediction} ({final_confidence:.2f}%)"
    cv2.putText(frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize MediaPipe object detection (using selfie segmentation as base)
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Load the video
cap = cv2.VideoCapture('video.mp4')

# Vehicle counter
vehicle_count = 0

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Car') as objectron:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural view and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = objectron.process(rgb_frame)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

                # Draw label
                bbox = detected_object.landmarks_2d.landmark
                x = int(bbox[0].x * frame.shape[1])
                y = int(bbox[0].y * frame.shape[0])

                cv2.putText(frame, "Vehicle", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                vehicle_count += 1

        # Show vehicle count
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Vehicle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

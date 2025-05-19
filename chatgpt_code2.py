import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Webcam
cap = cv2.VideoCapture(0)

# Background (you can replace this with an image)
bg_color = (255, 0, 255)  # Purple background

# Custom threshold for segmentation
threshold = 0.95  # You can adjust this between 0.0 and 1.0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = segmentor.process(img_rgb)

    # Get segmentation mask
    mask = results.segmentation_mask

    # Create condition mask based on custom threshold
    condition = mask > threshold

    # Create background image
    bg_image = np.zeros(img.shape, dtype=np.uint8)
    bg_image[:] = bg_color

    # Apply the mask
    output_image = np.where(condition[..., np.newaxis], img, bg_image)

    cv2.imshow("Background Removed", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

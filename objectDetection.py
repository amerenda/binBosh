import cv2
import tensorflow as tf

def detect_and_draw_boxes(image, detector):
    # Convert image to array and add a batch dimension
    input_image = tf.convert_to_tensor([image])
    
    # Run detection
    detections = detector(input_image)
    
    # Extract detection boxes and draw them
    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    
    height, width, _ = image.shape
    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5: # Confidence score threshold
            box = detection_boxes[i]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 0), 2) # Color is black
    
    return image

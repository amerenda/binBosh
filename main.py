import cv2
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import time
# For downloading the image.
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())


matplotlib.use('TkAgg')
high_res = cv2.imread('assets/30000×17078.jpg')
med_res = cv2.imread('assets/2560x1457.jpg')
low_res = cv2.imread('assets/1280x729.jpg')

test_res = (256, 256)
pc_res = (1080, 1920)
phone_res = (1440, 3120)

black_border_threshold = 2


def remove_horizontal_lines(image, line_color, line_thickness=20, threshold=2):
    """
    Removes horizontal lines from an image.
    
    Parameters:
        image (numpy.ndarray): The input image.
        line_color (tuple): BGR values of the line to be removed.
        line_thickness (int): Expected thickness of the line to be removed.
        threshold (int): Threshold value for color comparison.
    
    Returns:
        numpy.ndarray: The processed image.
    """

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to keep only the lines
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Define a horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, 1))
    
    # Apply morphological opening to remove horizontal lines
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find contours of the lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours
    for contour in contours:
        # Get bounding box of the line
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check whether the found contour is of a horizontal line
        if w > 1.5*h:
            # Replace line area with the background color
            cv2.rectangle(image, (x, y), (x+w, y+h), line_color, -1)
            
    return image

def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def detect_objects():
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
    detector = hub.load(module_handle).signatures['default']
    print(start_time = time.time())

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img
     

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  display_image(image_with_boxes)
     
def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def get_content_coordinates(image, threshold=255):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    max_area = 0
    best_box = (0, 0, image.shape[1], image.shape[0]) # default to whole image if no contours found
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            best_box = (x, y, x + w, y + h)
    
    return best_box

def draw_grid_adjusted(image, slices_coordinates):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    color_idx = 0
    
    for coordinates in slices_coordinates:
        x1, y1, x2, y2 = coordinates
        
        # Get color
        color = colors[color_idx]
        color_idx = (color_idx + 1) % len(colors)  # Move to the next color
        
        # Draw horizontal and vertical lines using OpenCV
        cv2.line(image, (x1, y1), (x2, y1), color, 1)  # Horizontal top line
        cv2.line(image, (x1, y2), (x2, y2), color, 1)  # Horizontal bottom line
        cv2.line(image, (x1, y1), (x1, y2), color, 1)  # Vertical left line
        cv2.line(image, (x2, y1), (x2, y2), color, 1)  # Vertical right line
    
    cv2.imshow("Grid Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def slice_image_fixed_size(image, resolution):
    """
    Slices an image into pieces of specified resolution with possible overlap.
    
    :param image: Numpy array of the image.
    :param resolution: Tuple (width, height) specifying the desired resolution of slices.
    :return: List of slices with specified resolution.
    """
    img_height, img_width, _ = image.shape
    slice_width, slice_height = resolution  # Modified line
    
    slices = []
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            # Ensure the slicing does not exceed image boundaries
            y_end = min(y + slice_height, img_height)
            x_end = min(x + slice_width, img_width)
            
            # Check if the slice does not meet the desired resolution and adjust
            if y_end - y < slice_height:
                y_start = img_height - slice_height  # Start slice at the boundary to maintain resolution
            else:
                y_start = y
                
            if x_end - x < slice_width:
                x_start = img_width - slice_width  # Start slice at the boundary to maintain resolution
            else:
                x_start = x
                
            # Crop the image to extract the desired slice
            slice_ = image[y_start:y_end, x_start:x_end]
            slices.append(slice_)
            
            x += slice_width  # Adjust the step size horizontally
        y += slice_height  # Adjust the step size vertically
    
    return slices


def draw_slice_overlay(image, resolution, overlay_alpha=0.3, show_index=True, show_coordinates=False):
    """
    Overlays the original image with semi-transparent colored rectangles,
    representing each slice. Different colors are used to highlight overlaps.

    :param image: Original image.
    :param resolution: Tuple (width, height) specifying the desired resolution of slices.
    :param overlay_alpha: Transparency level of the overlay. 0 is fully transparent, 1 is opaque.
    :param show_index: Boolean, if True, display the slice index on the overlay.
    :param show_coordinates: Boolean, if True, display the coordinates on the overlay.
    :return: Image with overlay of slices.
    """
    overlay = image.copy()  # Create a copy to draw overlay slices on
    output = image.copy()  # Create another copy to blend with the overlay
    
    img_height, img_width, _ = image.shape
    slice_width, slice_height = resolution
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Example colors
    
    y = 0
    slice_index = 0
    
    while y < img_height:
        x = 0
        while x < img_width:
            # Determine the height and width of the slice to overlay
            overlay_height = min(slice_height, img_height - y)
            overlay_width = min(slice_width, img_width - x)
            
            # Choose a color for the current slice (cycling through the defined colors)
            color = colors[(x // slice_width + y // slice_height) % len(colors)]
            
            # Draw a semi-transparent rectangle on the overlay
            cv2.rectangle(overlay, (x, y), (x + overlay_width, y + overlay_height), color, -1)
            
            # Add text annotations for slice index and/or coordinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            font_color = (255, 255, 255)  # White color for text

            text_x = x + 10  # Small margin from the top left corner of the slice
            text_y = y + 20 
            
            if show_index:
                cv2.putText(overlay, f"Idx: {slice_index}", (text_x, text_y), 
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            
            if show_coordinates:
                coord_text = f"({x}, {y})"
                cv2.putText(overlay, coord_text, (text_x, text_y + 20), 
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            
            x += slice_width  # Move to the next column
            slice_index += 1  # Increment the slice index
            
        y += slice_height  # Move to the next row
    
    # Blend the overlay with the original image using the specified alpha
    cv2.addWeighted(overlay, overlay_alpha, output, 1 - overlay_alpha, 0, output)
    
    return output


def slices_sequentially(slices, border_threshold=2, remove_lines=False, line_color=(0, 0, 0), line_thickness=5, resolution=(256, 256), action="save"):
    """
    Displays slices of an image sequentially.
    
    Parameters:
        slices (list): List of image slices.
        ...
        resolution (tuple): Desired resolution of slices.
    """
    for i, slice_ in enumerate(slices):
        content_coordinates = get_content_coordinates(slice_, threshold=border_threshold)
        cropped_slice = slice_[content_coordinates[1]:content_coordinates[3], 
                               content_coordinates[0]:content_coordinates[2]]
        
        # Ensure the slice is resized to the desired resolution
        resized_slice = cv2.resize(cropped_slice, resolution)
        
        if remove_lines:
            resized_slice = remove_horizontal_lines(
                resized_slice, line_color, line_thickness=line_thickness
            )
        
        if action.lower() == "show":        
            cv2.imshow(f"Slice {i+1}", resized_slice)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif action.lower() == "save":
            if not os.path.exists("output"):
                os.makedirs("output")
            res_string = "{}x{}".format(resolution[0], resolution[1])
            cv2.imwrite(f"output/{res_string}_{i+1}.png", resized_slice)


def slice(image, action):
    slices = slice_image_fixed_size(image, resolution)

    # To display slices with horizontal lines removed, set remove_lines to True
    slices_sequentially(slices, border_threshold=black_border_threshold, remove_lines=True, line_color=(0, 0, 0), line_thickness=5, action=action)


def slice_and_process(image, action, resolution, overlay_only=False):
    # Slice image into defined resolution with overlaps
    if overlay_only:
        overlay_img = draw_slice_overlay(image, resolution)
        if action.lower() == "show":        
            cv2.imshow("Overlay", overlay_img)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()
        elif action.lower() == "save":        
            cv2.imwrite("output/overlay.jpg", overlay_img)
    else:
        slices = slice_image_fixed_size(image, resolution)
    

        # Optionally display each slice with removed horizontal lines
        slices_sequentially(slices, 
                            border_threshold=black_border_threshold,
                            remove_lines=True, line_color=(0, 0, 0), 
                            line_thickness=5, action=action,
                            resolution=resolution)

        overlay_img = draw_slice_overlay(image, resolution)
        if action.lower() == "show":        
            cv2.imshow("Overlay", overlay_img)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()
        elif action.lower() == "save":        
            cv2.imwrite("output/overlay.jpg", overlay_img)

    

# Invoke the function to slice and process
slice_and_process(high_res, action="save", resolution=phone_res, overlay_only=true)

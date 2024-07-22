import cv2
import pytesseract
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Use morphological operations to remove noise and enhance the image
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
   #display the processed image
    cv2.imshow('Processed Image', morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return morph

def ocr_image(image):
    # Use Tesseract to do OCR on the processed image
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    text = ocr_image(processed_image)
    return text

# Example usage
image_path = 'test_image.png'
extracted_text = extract_text_from_image(image_path)
print("Extracted Text:\n", extracted_text)

import cv2

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.medianBlur(gray_image, 3)
    _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    total_pixels = binary_image.shape[0] * binary_image.shape[1]
    white_pixels = cv2.countNonZero(binary_image)
    black_pixels = total_pixels - white_pixels
    
    if black_pixels > white_pixels:
        inverted_image = cv2.bitwise_not(binary_image)
    else:
        inverted_image = binary_image
    # inverted_image = cv2.resize(inverted_image, (300, 300))
    return inverted_image

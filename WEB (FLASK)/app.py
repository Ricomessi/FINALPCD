from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def calculate_edge_density(edges):
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    density = edge_pixels / total_pixels
    return density

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded_image}'

def process_image(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_base64 = encode_image_to_base64(gray)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_base64 = encode_image_to_base64(blurred)

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = blurred.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(gray.shape)
    segmented_image_base64 = encode_image_to_base64(segmented_image)

    # Binary Image
    _, binary_img = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img_base64 = encode_image_to_base64(binary_img)

    # Opening (morphological operation)
    kernel = np.ones((3, 3), np.uint8)
    opening_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    opening_img_base64 = encode_image_to_base64(opening_img)

    # Filling (morphological operation)
    h, w = opening_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled_img = opening_img.copy()

    inverted_img = cv2.bitwise_not(filled_img)
    cv2.floodFill(inverted_img, mask, (0, 0), 255)
    inverted_img = cv2.bitwise_not(inverted_img)
    filled_out_img = opening_img | inverted_img
    filled_out_img_base64 = encode_image_to_base64(filled_out_img)

    # Canny edge detection
    edges = cv2.Canny(filled_out_img, 30, 100)
    edges_base64 = encode_image_to_base64(edges)

    # Calculate edge density
    edge_density = calculate_edge_density(edges)
    
    return {
        'gray': gray_base64,
        'blurred': blurred_base64,
        'segmented': segmented_image_base64,
        'binary': binary_img_base64,
        'opening': opening_img_base64,
        'filled': filled_out_img_base64,
        'edges': edges_base64,
        'edge_density': edge_density
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    results = process_image(image)
    
    # Watermark detection
    threshold = 0.02  # This should be adjusted based on your analysis
    watermark_detected = results['edge_density'] > threshold
    detection_result = "Tanda Air Terdeteksi" if watermark_detected else "Tanda Air Tidak Terdeteksi"
    
    # Return JSON response with processed image, detection result, and edge density
    return jsonify({
        'processed_images': results,
        'detection_result': detection_result,
        'edge_density': results['edge_density']
    })

if __name__ == '__main__':
    app.run(debug=True)

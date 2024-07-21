import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('uifile.ui', self)
        self.Image = None
        self.button_inputCitra.clicked.connect(self.inputCitra)
        self.button_prosesCitra.clicked.connect(self.prosesCitra)
        self.edge_densities = []

    def inputCitra(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if fileName:
            self.Image = cv2.imread(fileName)
            self.displayImage(self.Image, 2)

    def prosesCitra(self):
        if self.Image is None:
            return

        try:
            # Grayscale
            gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            self.displayImage(gray, 3)

            # Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            self.displayImage(blurred, 4)

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
            self.displayImage(segmented_image, 5)

            # Binary Image
            _, binary_img = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.displayImage(binary_img, 6)

            # Opening (morphological operation)
            kernel = np.ones((3, 3), np.uint8)
            opening_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
            self.displayImage(opening_img, 7)

            # Filling (morphological operation)
            h, w = opening_img.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            filled_img = opening_img.copy()

            inverted_img = cv2.bitwise_not(filled_img)
            cv2.floodFill(inverted_img, mask, (0, 0), 255)
            inverted_img = cv2.bitwise_not(inverted_img)
            filled_out_img = opening_img | inverted_img
            self.displayImage(filled_out_img, 10)

            # Canny edge detection
            edges = cv2.Canny(filled_out_img, 30, 100)  # Adjusting the threshold values for Canny
            self.displayImage(edges, 11)

            # Calculate edge density
            edge_density = self.calculate_edge_density(edges)
            self.edge_densities.append(edge_density)
            print(f"Edge Density: {edge_density}")

            # Detect watermark by edge evaluation
            threshold = 0.03  # This should be adjusted based on your analysis
            if edge_density > threshold:
                self.label_detection.setText("Tanda Air Terdeteksi")
            else:
                self.label_detection.setText("Tanda Air Tidak Terdeteksi")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during image processing: {str(e)}")

    def calculate_edge_density(self, edges):
        # Calculate the density of edges in the image
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        density = edge_pixels / total_pixels
        return density

    def displayImage(self, img, window=1):
        if img is not None:
            qformat = QImage.Format_Indexed8

            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img_qt = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
            img_qt = img_qt.rgbSwapped()

            if window == 1:
                self.label.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 2:
                self.label_2.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 3:
                self.label_3.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 4:
                self.label_4.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 5:
                self.label_5.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 6:
                self.label_6.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 7:
                self.label_7.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 8:
                self.label_8.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 9:
                self.label_9.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 10:
                self.label_10.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 11:
                self.label_11.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 12:
                self.label_12.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 13:
                self.label_13.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 14:
                self.label_14.setPixmap(QPixmap.fromImage(img_qt))
            elif window == 15:
                self.label_detection.setPixmap(QPixmap.fromImage(img_qt))

    def closeEvent(self, event):
        # Plot edge densities
        plt.hist(self.edge_densities, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Edge Density Distribution')
        plt.xlabel('Edge Density')
        plt.ylabel('Frequency')
        plt.show()
        event.accept()


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('UAS Project')
window.show()
sys.exit(app.exec_())

import cv2
import numpy as np
from sklearn.cluster import KMeans

def dominant_color_extraction(image_path, k=3):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply k-means clustering to find the most dominant clusters
    kmeans = KMeans(n_clusters=k, n_init=10)

    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the number of pixels assigned to each cluster
    hist, _ = np.histogram(kmeans.labels_, bins=np.arange(0, k + 1))

    # Find the most dominant cluster
    dominant_color = cluster_centers[hist.argmax()].astype(int)

    return tuple(dominant_color)

image_path = 'test_image1.jpg'
color = dominant_color_extraction(image_path)
print(f"Dominant Color: {color}")

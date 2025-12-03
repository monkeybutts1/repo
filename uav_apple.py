import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image_by_color(image_path, k=4):
    """
    Loads an image, segments it into 'k' colors using K-Means clustering,
    displays each color segment individually, and prints the pixel coordinates
    for the center of each colored object.

    Args:
        image_path (str): The full path to the input image file.
        k (int): The number of distinct colors to segment the image into.
    """
    # 1. Load the Image
    # cv2.imread loads the image in BGR (Blue, Green, Red) format by default.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at '{image_path}'")
        return

    # Convert image from BGR to RGB color space for accurate display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Prepare Data for K-Means Clustering
    # Reshape the image to be a list of pixels (N_pixels x 3 channels)
    pixel_values = image_rgb.reshape((-1, 3))
    # Convert the data type to float32 for the k-means algorithm
    pixel_values = np.float32(pixel_values)

    # 3. Apply K-Means Clustering
    # Define the stopping criteria for the algorithm
    # (after 100 iterations or if epsilon accuracy of 0.85 is reached)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    
    # Perform K-Means clustering
    # It returns the compactness, labels for each pixel, and the cluster centers (colors)
    retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the cluster centers (which are float) back to 8-bit unsigned integers
    centers = np.uint8(centers)
    # Reshape the labels array back to the original image dimensions
    labels = labels.reshape(image_rgb.shape[:2])

    # 4. Create Segmented Images and Find Centers
    segmented_images = []
    print(f"--- Center Coordinates of {k} Color Objects ---")
    
    for i in range(k):
        # Create a new blank image with the same dimensions as the original
        segmented_image = np.zeros_like(image_rgb)
        
        # Get the color of the current cluster
        cluster_color = centers[i]
        
        # In the blank image, apply the cluster color to pixels belonging to the current cluster
        segmented_image[labels == i] = cluster_color
        segmented_images.append(segmented_image)

        # --- Bonus Goal: Find the center of each object ---
        # Create a binary mask (black and white) for the current cluster
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == i] = 255  # Set pixels of the current cluster to white

        # Calculate the "moments" of the binary mask. Moments help describe the shape.
        M = cv2.moments(mask)

        # Calculate the centroid (center x, y) from the moments.
        # We check M["m00"] to avoid division by zero if the object has no area.
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Color Cluster {i+1} (RGB: {cluster_color}) -> Center: ({cX}, {cY}) pixels")
        else:
            print(f"Color Cluster {i+1} (RGB: {cluster_color}) -> No object found (area is zero)")

    # 5. Display the Results
    # Create a plot to display the original image and all the segmented images
    num_images = k + 1
    plt.figure(figsize=(15, 5))

    plt.subplot(1, num_images, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    for i, seg_img in enumerate(segmented_images):
        plt.subplot(1, num_images, i + 2)
        plt.imshow(seg_img)
        plt.title(f'Color {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # You must change this path to point to an image file on your computer.
    # For this example, download an image of a stop sign and save it as 'stop_sign.jpg'
    # in the same directory as this script.
    # Example Image: 
    
    image_file_path = 'obi.jpg'  
    
    # Set the number of colors you want to split the image into.
    # For a stop sign, 3 or 4 is usually good (red, white, background).
    number_of_colors = 2  

    try:
        segment_image_by_color(image_file_path, number_of_colors)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{image_file_path}' was not found.")
        print("Please make sure the image file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
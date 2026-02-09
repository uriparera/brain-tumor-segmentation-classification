import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

# Classification Model Parameters
CLASSIFICATION_MODEL_PATH = r"classification_model.keras"

# Segmentation Model Parameters
SEGMENTATION_MODEL_PATH = r"segmentation_model.pth"

# Image information
IMAGE_FOLDER_PATH = r"evaluate"
IMAGE_SIZE = 256

# Load Classification Model
def load_classification_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Classification model loaded.")
    return model

# Classification Prediction
def predict_tumor_type(model, img_path, image_size):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((image_size, image_size))
    x = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    return tumor_classes[class_idx], img_resized

# Load Segmentation Model
def load_segmentation_model(checkpoint_path, device):
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Segmentation model loaded.")
    else:
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")
    model.to(device)
    return model

# Segmentation Prediction
def segment_image(model, img, device):
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    if prediction['masks'].shape[0] > 0:
        pred_mask = prediction['masks'][0, 0].cpu().numpy() > 0.5
        return pred_mask
    return None

# Visualize Results
def visualize_results(img, tumor_type, pred_mask=None):
    
    # Show the segmented mask if available
    if pred_mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 1, 1)
        overlay = np.array(img).copy()
        overlay_np = np.array(overlay)
        overlay_mask = np.zeros_like(overlay_np, dtype=np.uint8)
        overlay_mask[pred_mask] = [255, 0, 0]  # Red mask (change to specify color)
        # Display the original image
        plt.imshow(overlay_np)
        # Overlay the mask with opacity
        plt.imshow(overlay_mask, alpha = 0.3)  # Adjust alpha for transparency (0.0 to 1.0)
        plt.title(f"Final diagnostic: ({tumor_type})")
        plt.axis("off")
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 1, 1)
        plt.axis("off")
        overlay = np.array(img).copy()
        overlay_np = np.array(overlay)
        plt.imshow(overlay_np)
        plt.title(f"Final diagnostic: ({tumor_type})")

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    # Load Classification Model
    classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

    # Load Segmentation Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH, device)

    # Iterate through images in the classification folder
    for filename in os.listdir(IMAGE_FOLDER_PATH):
        file_path = os.path.join(IMAGE_FOLDER_PATH, filename)

        if file_path.endswith((".jpg", ".jpeg", ".png")):
            # Classification
            tumor_type, img_resized = predict_tumor_type(classification_model, file_path, IMAGE_SIZE)
            
            if tumor_type != "No Tumor":  # Proceed to segmentation if tumor detected
                print(f"Detected {tumor_type} in {filename}. Segmenting tumor region...")
                pred_mask = segment_image(segmentation_model, img_resized, device)
                visualize_results(img_resized, tumor_type, pred_mask)
            else:
                print(f"No tumor detected in {filename}. Skipping segmentation.")

if __name__ == "__main__":
    main()


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image, ImageDraw
import json
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import time
import psutil

# Custom dataset class for training
class BrainTumorDataset(Dataset):
    def __init__(self, images, masks, transforms=None):
        """
        Args:
            images: List of image arrays.
            masks: List of corresponding mask arrays.
            transforms: Torchvision transforms to apply.
        """
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        
        # Convert mask into target format expected by Mask R-CNN
        obj_ids = np.unique(mask)[1:]  
        masks = mask == obj_ids[:, None, None]

        # Bounding boxes around each object
        num_objs = len(obj_ids)
        boxes = []
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            xmin = np.min(pos[1])
            ymin = np.min(pos[0])
            xmax = np.max(pos[1])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels (foreground = 1)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Convert masks to expected format
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        img = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)  # Convert to tensor

        return img, target

# Function to load annotations and process them in batches
def load_annotations_parallel(json_path, image_dir, output_img_dir, output_mask_dir, batch_size=32):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Things to distribute among threads
    tasks = [
        {
            'image_info': next(img for img in annotations['images'] if img['id'] == ann['image_id']),
            'segmentation': ann['segmentation'][0]  # Assuming there’s one segmentation per annotation
        }
        for ann in annotations['annotations']
    ]
    
    images, masks = [], []
    
    # Process the images and masks in batches using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            results = executor.map(
                lambda ann: process_single_annotation(ann, image_dir, output_img_dir, output_mask_dir), batch_tasks
            )
            
            for img_array, mask_array in results:
                images.append(img_array)
                masks.append(mask_array)
    
    return np.array(images).astype(np.float32), np.array(masks).astype(np.float32)

# Function to calculate IoU
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to process annotations and create masks without resizing
def process_single_annotation(annotation, image_dir, output_img_dir, output_mask_dir):
    """
    Processes a single annotation to create resized images and masks.

    Args:
        annotation: Dictionary containing image and annotation data.
        image_dir: Path to the directory containing original images.
        output_img_dir: Path to save the processed images.
        output_mask_dir: Path to save the processed masks.

    Returns:
        img_array: Resized image array (normalized to [0, 1]).
        mask_array: Resized mask array (binary).
    """
    # Extract image information and segmentation
    image_info = annotation['image_info']
    segmentation = annotation['segmentation']
    image_path = os.path.join(image_dir, image_info['file_name'])

    # Load the original image and convert to RGB
    img = Image.open(image_path).convert("RGB")

    # Create a binary mask for the segmentation
    mask = Image.new('L', (image_info['width'], image_info['height']), 0)
    draw = ImageDraw.Draw(mask)

    # Draw the polygon onto the mask
    scaled_polygon = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
    draw.polygon(scaled_polygon, outline=1, fill=1)

    # Resize the image and mask to 320x320
    img = img.resize((320, 320), Image.BICUBIC)
    mask = mask.resize((320, 320), Image.NEAREST)

    # Convert the image and mask to arrays
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    mask_array = np.array(mask).astype(np.float32)  # Convert to binary

    # Save the resized image and mask
    img.save(os.path.join(output_img_dir, image_info['file_name']))
    mask.save(os.path.join(output_mask_dir, image_info['file_name'].replace('.jpg', '.png')))

    return img_array, mask_array

def collate_fn(batch):
    """
    Custom collate function to properly batch images and targets.

    Args:
        batch: List of tuples (image, target) from the dataset.

    Returns:
        Tuple: Batched images and a list of corresponding targets.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)

import matplotlib.pyplot as plt

def plot_resized_images_and_masks(images, masks, titles=None):
    """
    Plots the first two resized images and their corresponding true masks.

    Args:
        images: List of resized images (numpy arrays).
        masks: List of corresponding resized masks (numpy arrays).
        titles: List of titles for the images and masks (optional).
    """
    num_to_plot = 2  # Number of images to plot
    fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, 5 * num_to_plot))

    for i in range(num_to_plot):
        img = images[i]
        mask = masks[i]

        # Plot the image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(titles[i] if titles else f"Image {i+1}")

        # Plot the mask
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"True Mask {i+1}")

    plt.tight_layout()
    plt.show()

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001, 
                              optimizer_class=torch.optim.Adam, batch_size=4, metrics_output_path=None, 
                              model_save_path=None):

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    training_metrics = {"epochs": []}
    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0

        # Monitor resource usage
        cpu_percentages = []
        gpu_utilizations = []

        # Add progress bar for the training loop
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for imgs, targets in progress_bar:
                # Record CPU usage
                cpu_percentages.append(psutil.cpu_percent(interval=None))

                # Record GPU usage if available
                if torch.cuda.is_available():
                    gpu_utilizations.append(torch.cuda.utilization())

                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()
                losses.backward()
                optimizer.step()

                # Update the progress bar with the current loss
                progress_bar.set_postfix(loss=losses.item())

        # Calculate metrics for this epoch
        epoch_time = time.time() - start_time
        avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        avg_gpu_usage = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0

        # Validation IoU computation
        model.eval()
        val_iou_list = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(imgs)
                for pred, target in zip(preds, targets):
                    pred_masks = pred["masks"].cpu().numpy() > 0.5
                    true_masks = target["masks"].cpu().numpy()

                    for pred_mask, true_mask in zip(pred_masks, true_masks):
                        iou = compute_iou(pred_mask[0], true_mask)
                        val_iou_list.append(iou)

        avg_val_iou = np.mean(val_iou_list) if val_iou_list else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - "
              f"Loss: {epoch_loss:.4f}, Avg IoU: {avg_val_iou:.4f}, "
              f"Avg CPU: {avg_cpu_usage:.2f}%, Avg GPU: {avg_gpu_usage:.2f}%")

        # Save metrics for this epoch
        training_metrics["epochs"].append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "avg_iou": avg_val_iou,
            "time": epoch_time,
            "avg_cpu": avg_cpu_usage,
            "avg_gpu": avg_gpu_usage
        })

        # Save checkpoint
        if model_save_path:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(checkpoint, model_save_path)
            print(f"Checkpoint saved to {model_save_path}")
            model.train()

        # Save metrics to JSON file
        if metrics_output_path:
            with open(metrics_output_path, "w") as f:
                json.dump(training_metrics, f, indent=4)
            print(f"Training metrics saved to {metrics_output_path}")

    return model

# Main function
def main():
    train_json_path = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\archive_seg_reduced\train\_filtered_annotations.coco.json"
    train_image_dir = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\archive_seg_reduced\train"
    output_img_dir  = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\results2\images"
    output_mask_dir = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\results2\masks"
    metrics_output_path = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\results2\training_metrics_v5.json"
    model_save_path = r"C:\Users\Uri\OneDrive\Documents\Uni\CPIA\project\models\model_maskrcnn_v5.pth"

    # Load data (use your existing data loading functions)
    print("Loading data...")
    images, masks = load_annotations_parallel(train_json_path, train_image_dir, output_img_dir, output_mask_dir)

    # Plot some images as examples to see the processed data
    # plot_resized_images_and_masks(images[:2], masks[:2], titles=["Resized Image 1", "Resized Image 2"])

    # Split data into training and validation sets
    val_split = int(len(images) * 0.2)
    train_images = images[:-val_split]
    train_masks = masks[:-val_split]
    val_images = images[-val_split:]
    val_masks = masks[-val_split:]

    # Create datasets and data loaders
    train_dataset = BrainTumorDataset(train_images, train_masks)
    val_dataset = BrainTumorDataset(val_images, val_masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load pre-trained Mask R-CNN model
    print("Loading pre-trained Mask R-CNN model...")
    model = maskrcnn_resnet50_fpn_v2(weights='COCO_V1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    print("Training the model...")
    model = train_model(
        model, train_loader, val_loader, device,
        num_epochs=4, learning_rate=0.001,
        optimizer_class=torch.optim.Adadelta, batch_size=4,
        metrics_output_path=metrics_output_path, model_save_path=model_save_path
    )

    print("Training complete.")

if __name__ == "__main__":
    main()

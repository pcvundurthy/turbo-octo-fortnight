import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

import os
import cv2
import json
import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.optim import AdamW
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm


def compute_iou(pred, target):
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return (intersection ) / (union )

def compute_dice(pred, target):
    intersection = (pred * target).float().sum((1, 2))
    return (2. * intersection ) / (pred.float().sum((1, 2)) + target.float().sum((1, 2)) )


def get_img_id(image_name, coco_data):
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            return img['id']
    return None

def get_ann_points(image_name, coco_data):
    img_id = get_img_id(image_name, coco_data)
    if img_id == None:
        return None
    points = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            segment = ann['segmentation'][0]
            pts = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
            points.append(pts)
    return points

def create_binary_masks(image_folder, coco_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    dataset = []
    unannotated = 0
    for fname in image_files:
        image_path = os.path.join(image_folder, fname)
        points = get_ann_points(fname, coco_data)
        if points == None:
            unannotated+=1
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for obj_pts in points:
            poly = np.array(obj_pts, np.int32)
            cv2.fillPoly(mask, [poly], 1)

        mask_filename = fname.replace('.jpg', '_mask.png')
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask * 255)
        dataset.append((image_path, mask_path, mask))
    print(f"unannotated: {unannotated}")
    return dataset



class PipeWallDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) // 255

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image.float() / 255.0, mask.long()


data_path1 = "dataset_segment_blades/Blades_amit"
mask_output_dir1 = "dataset_segment_blades/masks_Blades"
coco_data1 = json.load(open("dataset_segment_blades/Blades_amit/annotations_coco_segmentation_amit.json"))
dataset1 = create_binary_masks(data_path1, coco_data1, mask_output_dir1)


data_path2="dataset_segment_blades/Blades_pavan"
coco_data2 = json.load(open("dataset_segment_blades/Blades_pavan/Blades_dataset_adjusted_ids.json"))
dataset2 = create_binary_masks(data_path2, coco_data2, mask_output_dir1)



len([file for file in os.listdir('masks_new') if file.lower().endswith('.png')])


# In[29]:


dataset = dataset1 + dataset2
print(len(dataset))

# In[31]:

all_images, all_masks = zip(*[(img[0], img[1]) for img in dataset])
train_images, val_images, train_masks, val_masks = train_test_split(all_images, all_masks, test_size=0.05, random_state=42)


# In[33]:


print(f"train size: {len(train_images)}")
print(f"val size: {len(val_images)}")


# In[35]:


transform = A.Compose([A.Resize(512, 512), ToTensorV2()])


# In[37]:


train_dataset = PipeWallDataset(train_images, train_masks, transform)
val_dataset = PipeWallDataset(val_images, val_masks, transform)


# In[39]:


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# In[41]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b3")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)


# In[43]:


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images, labels=masks)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    total_loss, iou_scores, dice_scores = 0, [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits

            logits_resized = torch.nn.functional.interpolate(logits, size=(masks.size(1), masks.size(2)), mode='bilinear', align_corners=False)
            logits_resized = logits_resized.argmax(dim=1)

            iou_scores.extend(compute_iou(logits_resized, masks).cpu().numpy())
            dice_scores.extend(compute_dice(logits_resized, masks).cpu().numpy())

    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    total_loss = 1 - avg_dice

    return total_loss, avg_iou, avg_dice


# In[45]:


train_losses, val_losses, val_iou, val_dice = [], [], [], []


# In[ ]:


best_val_iou = -float('inf')
best_val_dice = -float('inf')

for epoch in range(100):
    print(f"Epoch {epoch + 1}/100")

    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_iou_score, val_dice_score = evaluate_model(model, val_loader, device)

    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Validation IOU: {val_iou_score:.4f}, Validation Dice: {val_dice_score:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_iou.append(val_iou_score)
    val_dice.append(val_dice_score)

    if val_iou_score > best_val_iou or val_dice_score > best_val_dice:
        best_val_iou = val_iou_score
        torch.save(model.state_dict(), "best_model.pth")


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Across Epochs')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(range(len(val_iou)), val_iou, 'b-', label='Validation IOU')
plt.plot(range(len(val_dice)), val_dice, 'r-', label='Validation dice score')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation IOU and dice scores Across Epochs')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


# test_folder = '/kaggle/input/data-imgs'
# test_mask_output_dir = "/kaggle/working/test_masks"
# coco_data = json.load(open("/kaggle/input/data-imgs/labels_my-project-name_2025-03-14-06-04-50.json"))
# test_dataset = create_binary_masks(test_folder, coco_data, test_mask_output_dir)

# test_images, test_masks = zip(*[(img[0], img[1]) for img in test_dataset])

# test_dataset = PipeWallDataset(test_images, test_masks, transform)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# In[ ]:


def overlay_predictions(model, loader, device, output_dir="/kaggle/working/predictions"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    unique_colors = [(255, 0, 0)]
    total_iou = 0
    total_dice = 0
    total_images = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Generating Predictions")):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1)
            # change mode to nearest
            preds_resized = torch.nn.functional.interpolate(preds.unsqueeze(1).float(), size=(masks.shape[1], masks.shape[2]), mode='nearest', align_corners=False).squeeze(1).long()

            # Compute IOU and Dice after resizing
            batch_iou = compute_iou(preds_resized, masks)
            batch_dice = compute_dice(preds_resized, masks)

            total_iou += batch_iou.sum().item()  # Sum up IOU scores across the batch
            total_dice += batch_dice.sum().item()  # Sum up Dice scores across the batch
            total_images += len(images)

            preds = preds_resized.cpu().numpy()

            preds_resized = [cv2.resize(pred.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) for pred in preds]

            for i, (image, pred) in enumerate(zip(images.cpu().numpy(), preds_resized)):
                image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                overlay = image.copy()

                class_idx = 1
                mask = (pred == class_idx).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(overlay, contours, unique_colors[0])

                alpha = 0.5
                final_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0.0)

                output_path = os.path.join(output_dir, f"pred_{batch_idx}_{i}.png")
                cv2.imwrite(output_path, final_image)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                plt.title("Segmented Output")
                plt.axis("off")

                plt.show()

    avg_iou = total_iou / total_images
    avg_dice = total_dice / total_images

    print(f"Average IOU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")

    return avg_iou, avg_dice


# In[ ]:


iou_avg, dice_avg = overlay_predictions(model, val_loader, device)


# In[ ]:


iou_avg


# In[ ]:


dice_avg


# In[ ]:





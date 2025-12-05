import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import os
from pathlib import Path
import kagglehub
import random
import matplotlib.pyplot as plt
import time


def download_coco_dataset():
    try:
        return kagglehub.dataset_download("nikhil7280/coco-image-caption")
    except Exception:
        return None


def merge_dataset(coco_path, max_images=None):
    json_files = [
        os.path.join(coco_path, "annotations_trainval2014", "annotations", "captions_train2014.json"),
        os.path.join(coco_path, "annotations_trainval2017", "annotations", "captions_val2017.json"),
        os.path.join(coco_path, "annotations", "captions_train2014.json"),
        os.path.join(coco_path, "annotations", "captions_val2017.json"),
        os.path.join(coco_path, "captions_train2014.json"),
    ]
    
    files = [f for f in json_files if os.path.exists(f)]
    if not files:
        raise ValueError
    
    image_captions = {}
    for file in files:
        with open(file, 'r') as f:
            coco_data = json.load(f)
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        for ann in coco_data['annotations']:
            if ann['image_id'] in id_to_filename:
                filename = id_to_filename[ann['image_id']]
                image_captions.setdefault(filename, []).append(ann['caption'])
    
    if max_images and len(image_captions) > max_images:
        image_captions = dict(list(image_captions.items())[:max_images])
    
    return image_captions


class COCODataset(Dataset):
    def __init__(self, coco_path, image_captions, processor):
        self.processor = processor
        self.image_captions = image_captions
        coco_path = Path(coco_path)
        
        possible_dirs = []
        for d in ['train2017', 'val2017', 'train2014', 'val2014', 'images']:
            flat_dir = coco_path / d
            nested_dir = coco_path / d / d
            if nested_dir.exists():
                possible_dirs.append(nested_dir)
            elif flat_dir.exists():
                possible_dirs.append(flat_dir)
        
        self.image_dirs = possible_dirs
        if not self.image_dirs:
            raise ValueError(f"No image directories found in {coco_path}")

        filename_lookup = {}
        for img_dir in self.image_dirs:
            for img_file in img_dir.glob('*.jpg'):
                full_name = img_file.name
                short_name = next((full_name.replace(p, '') for p in ['COCO_train2014_', 'COCO_val2014_', 'COCO_train2017_', 'COCO_val2017_'] if p in full_name), full_name)
                filename_lookup[full_name] = filename_lookup[short_name] = full_name

        self.valid_images = []
        self.image_paths = {}
        for filename in image_captions.keys():
            for img_dir in self.image_dirs:
                for name in [filename, filename_lookup.get(filename)]:
                    if name and (img_dir / name).exists():
                        self.valid_images.append(filename)
                        self.image_paths[filename] = img_dir / name
                        break
                if filename in self.image_paths:
                    break
                
        if not self.valid_images:
            raise ValueError(f"No matching images found. Checked {len(image_captions)} filenames in {len(self.image_dirs)} directories")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        filename = self.valid_images[idx]
        image = Image.open(self.image_paths[filename]).convert('RGB')
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        text_inputs = self.processor.tokenizer(self.image_captions[filename][0], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        labels = text_inputs["input_ids"].clone()
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def train_on_coco(
    coco_path=None,
    output_dir="./trained_model",
    num_images=2000,
    num_epochs=10,
    batch_size=4,
    learning_rate=3e-5,
    use_gpu=True
):
    device = "mps" if (use_gpu and torch.backends.mps.is_available()) else "cpu"
    
    if coco_path is None:
        coco_path = download_coco_dataset()
        if not coco_path:
            raise ValueError
    
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    
    image_captions = merge_dataset(coco_path, max_images=num_images)
    train_dataset = COCODataset(coco_path, image_captions, processor)
    
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    
    class ShuffledSequentialSampler:
        def __init__(self, indices):
            self.indices = indices
        
        def __iter__(self):
            return iter(self.indices)
        
        def __len__(self):
            return len(self.indices)
    
    sampler = ShuffledSequentialSampler(indices)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    epoch_losses = []
    batch_losses = []
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            num_batches += 1
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(range(1, num_epochs + 1), epoch_losses, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Average Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss per Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    batch_indices = range(1, len(batch_losses) + 1)
    axes[0, 1].plot(batch_indices, batch_losses, linewidth=1, alpha=0.6, color='#A23B72')
    axes[0, 1].set_xlabel('Batch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Batch-Level Loss', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(range(1, num_epochs + 1), epoch_times, marker='s', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=11)
    axes[1, 0].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(batch_losses, bins=50, edgecolor='black', alpha=0.7, color='#C73E1D')
    axes[1, 1].set_xlabel('Loss', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Loss Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to {plot_path}")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to {plot_path}")
    plt.close()
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    return model, processor


if __name__ == '__main__':
    train_on_coco()
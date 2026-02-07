import torch
import pickle
from PIL import Image
import albumentations as A
import os, sys, glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["name"] = [i[1] for i in inputs]

    return batch

class SegmentationDataset(Dataset):
    def __init__(self, all_images, transform):
        self.dataset = all_images
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #name = img_path.split('/')[-1].split('.')[0]
        name = img_path.split('/')[-1]
        image = Image.open(img_path).convert('RGB')
        original_image = np.array(image)

        transformed = self.transform(image=original_image)
        image = torch.tensor(transformed['image'])

        # convert to C, H, W
        image = image.permute(2, 0, 1)

        return image, name

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config).to(device)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        patch_embeddings = outputs.last_hidden_state[:, 0, :]  # Exclude the CLS token

        return patch_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

transform = A.Compose([
    A.Resize(width=448, height=448),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
print("hello")
image_path_dir = sys.argv[1]
out_csv = image_path_dir+'_dinov2.csv'
all_images = glob.glob(image_path_dir+'/*')
model = Dinov2ForSemanticSegmentation.from_pretrained(r"/mnt/g/bigmodel/pretrain_large").cuda()

if len(all_images) != 0:
    
    val_dataset = SegmentationDataset(all_images, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    results = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            outputs = model(pixel_values.to(device))
            for name, feature in zip(batch["name"], outputs):
                flattened_feature = feature.cpu().numpy().flatten()
                feature_dict = {"ID": name}
                for i, value in enumerate(flattened_feature):
                    feature_dict[f"big_model_feature{i+1}"] = value
                results.append(feature_dict)

    df = pd.DataFrame(results)
    #csv_name = out.split('/')[-1]
    df.to_csv(out_csv, index=False)

    print(f'save feature to {out_csv}')
else:
    print(f'not found pngs in {image_path}')

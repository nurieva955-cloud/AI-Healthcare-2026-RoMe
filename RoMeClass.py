import os
import cv2
import torch
import pandas as pd
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


TEAM_NAME = "RoMe" 
TEST_DIR = '/content/drive/MyDrive/classification/test' # you can change this path
MODEL_PATH = f'./{TEAM_NAME}ClassModel.pth' 
OUTPUT_FILE = f'{TEAM_NAME} test_ground_truth.xlsx'

MODEL_NAME = 'swinv2_base_window12_192'
IMG_SIZE = 192
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

class HackathonTestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image_id = os.path.splitext(img_name)[0]
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        variants = [image, cv2.flip(image, 1), cv2.flip(image, 0), cv2.flip(image, -1)]
        tensors = [self.transform(image=v)['image'] for v in variants]
        return torch.stack(tensors), image_id

def main():
    print(f"Запуск инференса команды {TEAM_NAME}...")
    dataset = HackathonTestDataset(TEST_DIR, transform=base_transform)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch_tensors, image_ids in test_loader:
            B, TTA, C, H, W = batch_tensors.size()
            batch_tensors = batch_tensors.view(B * TTA, C, H, W).to(DEVICE)
            
            with torch.cuda.amp.autocast():
                outputs = model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
                
            probs = probs.view(B, TTA, -1).mean(dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            for i in range(B):
                results.append({
                    'Image_ID': str(image_ids[i]),
                    'Label': int(class_names[preds[i]])
                })
                
    df_results = pd.DataFrame(results)[['Image_ID', 'Label']]
    df_results.to_excel(OUTPUT_FILE, index=False)
    print(f"Готово! Файл сохранен: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

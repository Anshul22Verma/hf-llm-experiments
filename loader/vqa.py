import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class ArtworkTaggingDataset(Dataset): 
    def __init__(self, dataset_csv: str, split: str,
                 train_ratio: float=0.8): 
        self.dataset_csv = dataset_csv
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"{self.dataset_csv} does not exists")
        
        self.df = pd.read_csv(self.dataset_csv)
        if split == "train":
            self.df, _ = train_test_split(self.df, test_size=1-train_ratio)            
        else:
            _, self.df = train_test_split(self.df, test_size=1-train_ratio)            
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        example = self.df.iloc[idx]
        question = "ocr_text: " + example['ocr'][:1000] + " <DocVQA>" + " Extract attributes from the artwork image and OCR in a JSON"  # example['question']
        first_answer = example['answers'][0]
        image = Image.open(example['image']).convert("RGB")
        return question, first_answer, image
    

def get_artwork_tagging_loaders(processor, dataset_csv: str, batch_size: int=1,
                                num_workers: int=0, device = torch.device("cuda")):
    def collate_fn(batch): 
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        return inputs, answers 

    train_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="train")
    val_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="validation") 
    batch_size = batch_size
    num_workers = num_workers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=num_workers, 
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_loader

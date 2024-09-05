import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import PaliGemmaProcessor

model_id = "google/paligemma-3b-pt-224"
device = "cuda"


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
        # print(example['image'])
        question = "Extract all attributes like brand, variety, weight, language, address of the product from " + \
            "artwork image in a valid key-value pair JSON. This information is mostly present in text format in the " + \
            "artwork with visal cues for what they are."
        attr = eval(example['answers'])
        ans = attr  # {
        #     "Brand": attr["Brand"],
        #     "Variety": attr["Variety"],
        #     "Weight": attr["Weight"]
        # }
        answer = str(ans)
        image = Image.open(example['image']).convert("RGB")  # {"bytes": None, "path": example['image']}
        image = image.resize((1024, 1024))
        # return question, answer, image
        return question, answer, image


def get_artwork_tagging_datasets(dataset_csv):
    train_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="train")
    val_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="validation") 
    return train_dataset, val_dataset


def collate_fn(examples):
    print(examples)
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    texts = ["answer " + example["questions"] for example in examples]
    labels= [example["answer"] for example in examples]
    images = [example["image"] for example in examples]  # .convert("RGB")
    tokens = processor(text=texts, images=images, suffix=labels,
                return_tensors="pt", padding="longest",
                tokenize_newline_separately=False)
    
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens  # labels, tokens has answers as well


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


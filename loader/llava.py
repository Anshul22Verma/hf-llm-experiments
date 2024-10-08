import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoProcessor

from torch.utils.data import Dataset


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
        attr = eval(example['answers'])
        messages = []
        
        for k in attr.keys():
            if len(str(attr[k])) > 0 and attr[k]:
                if len(messages) == 0:
                    messages.append({
                            "content": [{"index": None, "text": f"What is {k} for this product?\n", "type": "text"},
                                        {"index": None, "text": None, "type": "image"}],
                            "role": "user"})
                else:
                    messages.append({
                            "content": [{"index": None, "text": f"What is {k} for this product?\n", "type": "text"}],
                            "role": "user"})
                messages.append({
                        "content": [{"index": None, "text": f"{str(k)}: {str(attr[k])}", "type": "text"}],
                        "role": "assistant"})
        
        if len(messages) == 0:
            messages.append({
                    "content": [{"index": None, "text": f"Extract all the key-fields about the product in the artwork?\n Language corresponds to languahe used in text of the artwork", "type": "text"},
                                {"index": None, "text": None, "type": "image"}],
                    "role": "user"})
        else:
            messages.append({
                    "content": [{"index": None, "text": f"Extract all the key-fields about the product in the artwork?\n Language corresponds to languahe used in text of the artwork", "type": "text"}],
                    "role": "user"})
        messages.append({
                "content": [{"index": None, "text": str(attr), "type": "text"}],
                "role": "assistant"})
        
        # first_answer = str(ans)[:2000]
        image = Image.open(example['image'])  # .convert("RGB")
        return {"messages": messages, "images": [image]}


class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        image_sizes = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text += self.processor.tokenizer.eos_token
            texts.append(text)
            images.append(example["images"][0])
            image_sizes.append((example["images"][0].width, example["images"][0].height))

        batch = self.processor(texts, images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

def get_artwork_tagging_datasets(dataset_csv: str):
    
    train_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="train")
    val_dataset = ArtworkTaggingDataset(dataset_csv=dataset_csv, split="validation") 
    return train_dataset, val_dataset

import argparse
import os
from transformers import TrainingArguments, Trainer
import torch

import sys
sys.path.append("/home/azureuser/hf-llm-experiments")

from loader.pali_gemma import collate_fn, get_artwork_tagging_datasets
from model.pali_gemma import get_model, prepare_model_for_ft


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_csv', default="/home/azureuser/Llava-LoRA/files_attributes.csv",
                        help="Location of the csv with master data")
    parser.add_argument('-e', "--epochs", default=2,
                        help="Number of epochs to train the model for")
    parser.add_argument('-bs', "--batch_size", default=1,
                        help="Batch size of the loaders (use 1 for training in T4 \
                            and 6 for training in 40 GB A100)")
    parser.add_argument('-nw', "--num_workers", default=0,
                        help="Number of workers simultaneously putting data into RAM")
    parser.add_argument('-o', "--output_dir", default="/home/azureuser/Llava-LoRA/pali-gemma",
                        help="Directory where all the output will be saved")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "runs")
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model_id = "google/paligemma-3b-pt-224"

    model, processor, image_token = get_model(model_id=model_id, device=device)
    model = prepare_model_for_ft(model, model_id=model_id)

    train_dataset, val_dataset = get_artwork_tagging_datasets(dataset_csv=args.data_csv)
    # train_loader, val_loader = get_artwork_tagging_loaders(processor=processor, dataset_csv=args.data_csv, 
    #                                                        batch_size=int(args.batch_size), num_workers=int(args.num_workers))

    args = TrainingArguments(
                num_train_epochs=int(args.epochs),
                remove_unused_columns=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=2,
                learning_rate=2e-5,
                weight_decay=1e-6,
                adam_beta2=0.999,
                logging_steps=100,
                optim="sgd",
                save_strategy="steps",
                save_steps=1000,
                save_total_limit=1,
                output_dir="paligemma_vqav2",
                bf16=True,
                report_to=["tensorboard"],
                dataloader_pin_memory=False
            )

    trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            args=args
            )
    trainer.train()
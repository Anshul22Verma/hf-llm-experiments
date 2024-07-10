import argparse
import os
from transformers import get_scheduler
from tqdm import tqdm
import torch


from loader.vqa import get_artwork_tagging_loaders
from model.florence_2 import florence_model
from utils.vqa_utils import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_csv', 
                        help="Location of the csv with master data")
    parser.add_argument('-e', "--epochs", default=7,
                        help="Number of epochs to train the model for")
    parser.add_argument('-bs', "--batch_size", default=1,
                        help="Batch size of the loaders (use 1 for training in T4 \
                            and 6 for training in 40 GB A100)")
    parser.add_argument('-nw', "--num_workers", default=0,
                        help="Number of workers simultaneously putting data into RAM")
    parser.add_argument('-o', "--output_dir",
                        help="Directory where all the output will be saved")
    
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "runs")
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model, processor = florence_model(train_vision_tower=False, device=device)

    train_loader, val_loader = get_artwork_tagging_loaders(processor=processor, 
                                                           dataset_csv=args.data_csv,
                                                           batch_size=int(args.batch_size), 
                                                           num_workers=int(args.num_workers),
                                                           device=device)

    epochs = int(args.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    num_training_steps = epochs * len(train_loader)

    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                                 num_warmup_steps=0, 
                                 num_training_steps=num_training_steps)
    
    model, processor, optimizer, lr_scheduler = \
        train(
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
            tb_loc=log_dir,
            checkpoint_dir=args.output_dir,
            suffix="florence_2",
            device=device
        )

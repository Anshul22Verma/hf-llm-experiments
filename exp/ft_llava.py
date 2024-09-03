import argparse
import os
from transformers import get_scheduler, TrainingArguments
from tqdm import tqdm
import torch
from trl import SFTTrainer


from loader.llava import get_artwork_tagging_datasets, LLavaDataCollator
from model.llava_1_6 import llava_model, get_llava_tokenizer
from utils.vqa_utils import train

os.environ['TORCH_USE_CUDA_DSA'] = "1"


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

    model, processor, lora_config = llava_model(train_vision_tower=False, device=device, lora=False, qlora=True)
    tokenizer, processor = get_llava_tokenizer(processor=processor)

    data_collator = LLavaDataCollator(processor)
    # max_len = ""

    train_dataset, val_dataset = get_artwork_tagging_datasets(dataset_csv=args.data_csv)

    epochs = int(args.epochs)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    # num_training_steps = epochs * len(train_loader)

    # lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
    #                              num_warmup_steps=0, 
    #                              num_training_steps=num_training_steps)
    training_args = TrainingArguments(
        output_dir=log_dir,
        report_to="tensorboard",
        learning_rate=1e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        logging_steps=5,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        image_sizes=256,
        fp16=True,
        bf16=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer.train()
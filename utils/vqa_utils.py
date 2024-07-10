import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(model, processor,
                    train_loader, optimizer, 
                    lr_scheduler, desc: str="Training Epoch"):
    model.train() 
    train_loss = 0
    i = -1
    for inputs, answers in tqdm(train_loader, desc=desc):
        i += 1
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"] 
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss}")
    return model, optimizer, lr_scheduler, avg_train_loss


def validate_one_epoch(model, processor,
                       val_loader, desc: str="Validation Epoch"):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Training Loss: {avg_val_loss}")
    return model, avg_val_loss
      

def train(model, processor, train_loader, val_loader, optimizer, lr_scheduler, epochs: int,
          checkpoint_dir: str, tb_loc: str, suffix: str):
    writer = SummaryWriter(
        log_dir=tb_loc,
        filename_suffix=suffix
    )

    for epoch in range(epochs): 
        model, optimizer, lr_scheduler, avg_train_loss = \
            train_one_epoch(
                model=model,
                processor=processor,
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                desc=f"Training Epoch {epoch + 1}/{epochs}"
            )

        model, avg_val_loss = \
            validate_one_epoch(
                model=model,
                processor=processor,
                val_loader=val_loader,
                desc=f"Validation Epoch {epoch + 1}/{epochs}"
            )
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch+1)

        output_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    writer.close()
    return model, processor, optimizer, lr_scheduler
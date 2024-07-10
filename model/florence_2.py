from transformers import AutoModelForCausalLM, AutoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def florence_model(
    train_vision_tower: bool = False
):
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        revision='refs/pr/6'
    ).to(device) 
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", 
        trust_remote_code=True, revision='refs/pr/6')

    if not train_vision_tower:
        for param in florence_model.vision_tower.parameters():
            param.is_trainable = False
    return florence_model, processor

from transformers import AutoModelForCausalLM, AutoProcessor
import torch


def florence_model(
    train_vision_tower: bool = False,
    device = torch.device("cuda")
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

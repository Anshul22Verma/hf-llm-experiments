from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import torch


def get_model(model_id, device):
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    return model, processor, image_token


def prepare_model_for_ft(model, model_id):
    # to only train text decoder
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # or LoRA QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=6,
        lora_alpha=12,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config,  device_map={"":0})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
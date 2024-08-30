from transformers import AutoProcessor, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]

    for name, module in model.name_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def llava_model(
    train_vision_tower: bool = False,
    device = torch.device("cuda"),
    lora: bool = False,
    qlora: bool = False
):
    MODEL_ID = "liuhaotian/llava-v1.6-mistral-7b"  # model_id = "llava-hf/llava-1.5-7b-hf"
    USE_LORA = lora
    USE_QLORA = qlora

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    # Load the model

    # We have 3 options available for training, from the lowest precision training to the highest
    # - QLoRA
    # - LoRA
    # - Full fine-tuning

    if USE_LORA or USE_QLORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                # lora_rank=16,
                # lora_alpha=32,
                # lora_dropout=0.1,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit = False,
                load_in_4bit=False,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_use_compute_dtype=False,
                bnb_4bit_quant_type=None,
                # lora_rank=16,
                # lora_alpha=32,
                # lora_dropout=0.1,
            )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2"
        )


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model, processor, lora_config


# to get tokenizer
def get_llava_tokenizer(processor):
    MODEL_ID = "liuhaotian/llava-v1.6-mistral-7b"  # model_id = "llava-hf/llava-1.5-7b-hf"

    LLAVA_CHAT_TEMPLATE = """A chat between a artwork operator and an artificial intelligence assistant. The assistant extracts different fiels from artwork based on the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer = tokenizer
    return tokenizer, processor
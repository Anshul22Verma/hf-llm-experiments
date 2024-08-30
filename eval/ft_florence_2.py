import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_loc', 
                        help="Location of the model to load it from")
    parser.add_argument('-i', "--image_loc",
                        help="Location of the image to make inference on")

    args = parser.parse_args()
    model_id = args.model_loc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.vision_config.model_type = "davit"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=config,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, config=config
    )
    """
    model = AutoModelForCausalLM.from_pretrained("/Users/vermaa/Downloads/models/epoch_10", local_files_only=True)
    processor = AutoProcessor.from_pretrained("/Users/vermaa/Downloads/models/epoch_10", local_files_only=True)
    """

    prompt = "<DocVQA>" + " Extract all attributes like brand, variety, weight of the product from artwork image in a valid key-value pair JSON."

    image = Image.open(args.image_loc)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    print(parsed_answer)
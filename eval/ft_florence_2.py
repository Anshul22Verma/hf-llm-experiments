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
    print(image)
    print(image.shape)

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


    """
    {'<DocVQA> Extract all attributes like brand, variety, weight of the product from artwork image in a valid key-value pair JSON.': '{\'Active Ingredients\': \'\', \'Address\': \'COSMO BOX\', \'Allergens\': \'Contains a Source of Phenylalanine.\', \'Brand\': \'BPS\', \'Sub-Brand\': "\', \'Product name\': \'cosMO Box\', \'Variety\': \'FRESH WITHITE\', \'Caffeine level\': \'yes\', \'Country of Origin\': \'UK\', \'Customer support info\': \'Produced in the Coronate K.com\', \'Email\': \'consumany\', \'Gluten free\': \'Suitable for vegetarians\', \'GMO / Non GMO\': \', \'Health claims\': \'Dieted at The Coronates K.\', \'Ingredients\': \'FREESH WITHTE\', \'Languages on pack\': [\'English\'], \'Number of languages\': 1, \'Manufactured in\': \'K\', \'Marketing/Romance Copy\': \'GET FRESH WITHIE\', \'NFP - Nutrient Facts\': \'Water, Saturates (Sodium Citrate), Preservative (Citric Acid), Preservatives (Sugar, Calcium carbonate), Sodium Sorbate), Sugar (Sucralose), Sugar'}
    """

    """
    {'<DocVQA> Extract all attributes like brand, variety, weight of the product from artwork image in a valid key-value pair JSON.': '{\'Active Ingredients\': \'\', \'Address\': \'Dr. Dr. Demris Gresis skin care, LLC., Welwyn Garden City AL7 1GA\', \'Allergens\': \'Packed in the U.K.\', \'Brand\': \'D. demris\', \'Sub-Brand\': "\', \'Product name\': \'ArtiosCAD SPECIFICATION Sheet\', \'Variety\': \'TARGED INGREDIENTS, CLEAR SKIN SOLUTIONS\', \'Caffeine level\': \'Suitable for vegetarians\', \'Country of Origin\': \'UK\', \'Customer support info\': \'Customer: 080011 REVO, B52V6, B530052, B560052\', \'Email\': \'consumercare@agb.co.uk\', \'Gluten free\': \'False\', \'GMO / Non GMO\': \'Health claims\': \'1/1/8 per per container\', \'Ingredients\': \'Carbonated Water, Glucose: Salsapron:\', \'Languages on pack\': [\'English\'], \'Number of languages\': 1, \'Manufactured in\': \'Marketing/Romance Copy\': \'DOCTOR FORMULATED TARGETED INGRIAN\', \'NFP - Nutrient Facts\': \'Energy: 1/1 per container, 1/2 per container; 1/4 per container (Saturated Fat: 2.5 per container); 1.5g (Sodium Fat: 1.1g (Sunset Yellow FCF)\', \'Phone number\': \'080011\', \'Preparation / Cooking instructions\': \'Recyclable\': \'yes, Saturates: 1'}
    """

    """
    {'<DocVQA> Extract all attributes like brand, variety, weight of the product from artwork image in a valid key-value pair JSON.': '{\'Active Ingredients\': \'\', \'Address\': \'Produced in the U.S.A.\', \'Allergens\': \'Contains a Source of Phenylalanine\', \'Brand\': \'SGS\', \'Sub-Brand\': "\', \'Product name\': \'D\'Dens Cross\', \'Variety\': \'Caffeine level\': \'Suitable for vegetarians\', \'Country of Origin\': \'UK\', \'Customer support info\': \'Marketing/Romance Copy\': \'Substainability Claims\': \'U.SGS, Business Business, Marine Road, Dun Laoghaire, Co. Dublin\', \'Email\': \'consumercare@sgs.co.uk\', \'Gluten free\': \'English\', \'Number of languages\': 1, \'Manufactured in\': \'Packed in the u.s.A., UK\', \'NFP - Nutrient Facts\': \'Energy: 0.7613\', \'Phone number\': \'3.7719\', \'Preparation / Cooking instructions\': \'Recyclable\': \'1.7618\', \'Symbols / Logos\': \'United Kingdom\', \'Usage\': \'Warnings / Caution\': \'Water, Flavourings (Sodium Citrate, Calcium Carbonate, Iron, Niacin, Thiamin), Sugar, Sugar, Caffeine (Saturated Fat), Calcium carbonate, Sodium Citrate (Sulphite), Sugar'}
    """

    """
    {'<DocVQA> Extract all attributes like brand, variety, weight of the product from artwork image in a valid key-value pair JSON.': '{\'Active Ingredients\': \'\', \'Address\': \'Phenylalanine\', \'Allergens\': \'Wheat Flour\', \'Brand\': \'Lifeboy\', \'Sub-Brand\': "\', \'Product name\': \'Lifebuoy MULTIVITAMIN\', \'Variety\': \'Cool Fresh\', \'Caffeine level\': \'Suitable for vegetarians\', \'Country of Origin\': \'U.K.\', \'Customer support info\': \'Produced in the U.K. for Lifebuoy Stores Ltd., Welwyn Garden City AL7 1GA, UK\', \'Email\': \'UK\', \'Gluten free\': \'False\', \'GMO / Non GMO\': \'Barcodeinfo\', \'Health claims\': \'Energy: 0800 6800 227711\', \'Ingredients\': \'Milk Flour, Flavourings, Thiamin\', \'Languages on pack\': [\'English\'], \'Number of languages\': 1, \'Manufactured in\': \'United Kingdom\', \'Marketing/Romance Copy\': \'VING I LON\', \'NFP - Nutrient Facts\': \'Water, Flour (Sodium Citrate), Calcium Carbonate, Iron, Niacin, Iron), Flavour (Milk), Sugar, Calcium carbonate, Sodium Metabisulphite, Sodium Bicarbonate), Sugar (Citric Acid, Sugar, Sugar), Sugar'}
    """
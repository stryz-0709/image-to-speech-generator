from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

pretrained_processor = None
pretrained_model = None
fine_tuned_processor = None
fine_tuned_model = None
device = 'mps'


def _load_model(model_name, target_device):
    proc = BlipProcessor.from_pretrained(model_name)
    mdl = BlipForConditionalGeneration.from_pretrained(model_name)
    mdl = mdl.to(target_device)
    mdl.eval()
    return proc, mdl


def initialize_models():
    global pretrained_processor, pretrained_model, fine_tuned_processor, fine_tuned_model
    
    pretrained_name = "Salesforce/blip-image-captioning-base"
    pretrained_processor, pretrained_model = _load_model(pretrained_name, device)
    
    if os.path.exists("./trained_model"):
        fine_tuned_processor, fine_tuned_model = _load_model("./trained_model", device)
    else:
        fine_tuned_processor = None
        fine_tuned_model = None


def _generate_caption_from_model(image, proc, mdl, prompt, max_length):
    inputs = proc(image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs, 
            max_length=max_length, 
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            length_penalty=1.0
        )
    return proc.decode(out[0], skip_special_tokens=True)


def generate_caption(image, prompt="an image of", max_length=100):
    global pretrained_processor, pretrained_model, fine_tuned_processor, fine_tuned_model
    
    if pretrained_processor is None or pretrained_model is None:
        initialize_models()
    
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image object")
    
    pretrained_caption = None
    try:
        pretrained_caption = _generate_caption_from_model(image, pretrained_processor, pretrained_model, prompt, max_length)
    except Exception:
        pretrained_caption = "Error generating caption"
    
    fine_tuned_caption = None
    if fine_tuned_processor is not None and fine_tuned_model is not None:
        try:
            fine_tuned_caption = _generate_caption_from_model(image, fine_tuned_processor, fine_tuned_model, prompt, max_length)
        except Exception:
            fine_tuned_caption = "Error generating caption"
    
    return pretrained_caption, fine_tuned_caption

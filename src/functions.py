from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

def get_image_caption(image_path):
    """
    Generate a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.
    
    Returns: 
        str: Astring representing the caption for the image
    """
    raw_image = Image.open(image_path).convert('RGB')
    device = "cpu"
    model_name= "Salesforce/blip-image-captioning-large"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
    #print(processor.decode(out[0], skip_special_tokens=True))
    # unconditional image captioning
    #inputs = processor(raw_image, return_tensors="pt")

    #out = model.generate(**inputs)
    #print(processor.decode(out[0], skip_special_tokens=True))


def get_image_dedcription(image_path):
    """
    Generate a detailed description for the provided image.

    Args:
        image_path (str): The path to the image file.
    
    Returns: 
        str: Astring representing the detailed description for the image
    """
    pass

def get_image_palette(image_path):
    """
    Generate a 12 colors palette from the provided image.

    Args:
        image_path (str): The path to the image file.
    
    Returns: 
        list: list contains the 12 dominant colors in the provided image  
    """
    pass

def get_inspiration(image_caption):
    """
    Provide similars artworks links by searching google using the image captions

    Args:
        image_caption (str): The image file caption.
    
    Returns: 
        list: list contains links to top 4 similar artworks 
    """
    pass


if __name__ == '__main__':
    image_path = "C:/Users/Sumaya/Documents/projects/test/test4.jpg"
    caption = get_image_caption(image_path)
    print(caption)
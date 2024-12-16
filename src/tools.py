from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




class ImageToolInput(BaseModel): 
    image_path: str = Field(description = "Image path")

class ImageCaptionTool(BaseTool):
    name: str = "caption"
    description: str = "Use this tool to caption or describe an image. it will be given the path to the image as input, and it should return a simple caption describing the image."
    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""
        try:
            # Load the image
           
            raw_image = Image.open(image_path).convert('RGB')
            
            # Device configuration
            device = "cpu"  # Change to "cuda" if a GPU is available
            
            # Load the model and processor
            model_name = "Salesforce/blip-image-captioning-large"
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            # Prepare inputs and generate caption
            text = "a photography of"
            inputs = processor(raw_image, text, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
            

    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")


class ImagePaletteTool(BaseTool):
    name: str = "Palette"
    description: str = (
        "Use this tool to get the color palette of an image. "
        "The tool will return a list of RGB values representing the dominant colors in the image. "
        "Your task is to convert these RGB values into descriptive color names (e.g., 'Black', 'White', 'Sky Blue'). "
        "Return only the colors names don't mention the RGB values at all"
        "Make sure to include a markdown formatted list of the color names in your response."
    )
     
    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""
        try:
            # Load the image
            image = cv2.imread(image_path)

            # Convert the image from BGR (OpenCV default) to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reshape the image to a 2D array of pixels
            pixels = image_rgb.reshape(-1, 3)

            # Get unique colors
            unique_colors = np.unique(pixels, axis=0)

            # Number of dominant colors to extract
            num_colors = 5

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(pixels)

            # Get the cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)


            palette = np.zeros((50, 50 * num_colors, 3), dtype=np.uint8)

            # Create a string with all dominant colors in RGB format
            color_str = ", ".join([f"RGB({color[0]}, {color[1]}, {color[2]})" for color in dominant_colors])

            
            
            return color_str
        
            

        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
            

    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")

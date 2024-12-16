from langchain.tools import BaseTool, tool
from typing import Type
from pydantic import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

from PIL import Image, ImageOps
import os


@tool
def posterize_image(image_path: str, output_path: str = r"./temp", levels: int = 3) -> str:
    """
    Posterize the image so the artist can understand the basic image. 
    Inputs:
    - image_path: Path to the input image.
    - output_path: Path to save the posterized image. Defaults to "./temp".
    - levels: Number of posterization levels. Defaults to 50.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "posterized.jpeg")

    try:
        print(f"Input Path: {image_path}")
        print(f"Output Path: {output_file}")

        image_path = "./temp/test.jpeg"
        # Open and posterize the image
        img = Image.open(image_path).convert("RGB")
        # Posterize the image
        posterized_img = ImageOps.posterize(img, levels)
        posterized_img.save(output_file)

        return posterized_img

        # Save the posterized image

        return f"Image saved to {output_file}"
    except FileNotFoundError:
        return "Error: Image file not found. Please check the file path."
    except Exception as e:
        return f"Error: {str(e)}"


class ImageToolInput(BaseModel):
    image_path: str = Field(description="Image path")


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
    


class ImageGridTool(BaseTool):
    name: str = "grid"
    description: str = "Use this tool to create grid over an image. it will be given the path to the image as input, and it should a path to the output image."
    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""

        output_path = "./temp/grid.jpeg"

        try:
            grid_size=(10, 10)
            # Load the image
            img = mpimg.imread(image_path)
            h, w, _ = img.shape

            # Create the figure and axis
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_xticks(np.arange(0, w, w // grid_size[0]))
            ax.set_yticks(np.arange(0, h, h // grid_size[1]))
            
            # Add gridlines
            ax.grid(color='black', linestyle='--', linewidth=0.5)

            # Turn off axis labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Save the figure to the specified output path
            plt.savefig(output_path, format='jpeg', bbox_inches='tight', dpi=300)
           
        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
            

    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")

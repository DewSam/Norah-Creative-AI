from typing import Type
from pydantic import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
from typing import List
import requests
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())
# Retrieve keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


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
        "Include on your response how can the user generate these colors using their fav medium, if they did not specify use oil painting as default"
        "also mention the use of palette in painting"
    )

    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""
        try:
            output_path = "./temp/palette.jpeg"
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
            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            kmeans.fit(pixels)

            # Get the cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)

            palette = np.zeros((50, 50 * num_colors, 3), dtype=np.uint8)

            # Display the palette using matplotlib
            palette = np.zeros((50, 300, 3), dtype=np.uint8)
            step = 300 // num_colors
            for i, color in enumerate(dominant_colors):
                palette[:, i * step:(i + 1) * step] = color

            # Save the palette image using PIL
            palette_img = Image.fromarray(palette)
            palette_img.save(output_path)
            st.image(output_path)
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
    description: str = ("Use this tool to create grid over an image. it will be given the path to the image as input,"
                        "and it should a path to the output image, dont show the path only say that it successfully created the grid image."
                         "and explain how grid can help the learner in their art, give instructions on how to generate it in their physical canvas"
                          " or paper")
    
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
            st.image(output_path, caption="Grid Image")
            return output_path
        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
            

    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")
    

class ImagePosterizeTool(BaseTool):
    name: str = "posterize"
    description: str = ("Use this tool to posterize an image. it will be given the path to the image as input,"
                        " and it should return a path to the output image, dont show the path only say that it "
                        "successfully created the grid image."
                        "Explain to the user how posterizing the image can help them in their painting")
    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""

        output_path = "./temp/posterized.jpeg"
        levels =2

        try:
            # Open and posterize the image
            img = Image.open(image_path).convert("RGB")
            # Posterize the image
            posterized_img = ImageOps.posterize(img, levels)
            posterized_img.save(output_path)
            st.image(output_path, caption="Posterized Image")
            return output_path
        
        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
            

    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")


class ImageBlackAndWhiteTool(BaseTool):
    name: str = "black_and_white"
    description: str = "Convert an image to black and white. Provide the image path as input, and it will save a black-and-white version of the image. It should return a success message, not the path."
    args_schema: Type[BaseModel] = ImageToolInput
    return_direct: bool = False

    def _run(self, image_path: str) -> str:
        """Run the tool synchronously."""
        output_path = "./temp/black_and_white.jpeg"

        try:
            # Open the image
            img = Image.open(image_path).convert("RGB")
            # Convert the image to grayscale (black and white)
            bw_img = ImageOps.grayscale(img)
            # Save the black-and-white image
            bw_img.save(output_path)
            st.image(output_path, caption="Black and White Image")
            return "Successfully created the black-and-white image."
        
        except FileNotFoundError:
            return "Error: Image file not found. Please check the file path."
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, image_path: str) -> str:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")
    

class GoogleImageSearchInput(BaseModel):
    query: str  # Search query string


class GoogleImageSearchTool(BaseTool):
    name: str = "google_image_search"
    description: str = "Search for images using Google Custom Search API and display results in columns., your input should be a description of what you are searching for"
    #args_schema: Type[BaseModel] = GoogleImageSearchInput
    return_direct: bool = False

    def _run(self, query: str) -> List[str]:
        """Run the tool synchronously to search for images."""

        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&searchType=image&key={GOOGLE_API_KEY}"

        try:
            # Make the API request
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()

            if 'items' in data:
                images = [item['link'] for item in data['items']]

                # Display images in Streamlit
                num_columns = 3
                columns = st.columns(num_columns)
                for index, image_url in enumerate(images):
                    col = columns[index % num_columns]
                    col.image(image_url, use_column_width=True)

                return images
            else:
                return ["No images found."]
        
        except requests.exceptions.RequestException as e:
            return [f"Error: {str(e)}"]

    async def _arun(self, query: str) -> List[str]:
        """Run the tool asynchronously (not implemented)."""
        raise NotImplementedError("This tool does not support async execution.")
# Norah AI

Norah AI is an innovative AI assistant tailored for artists, students, and art enthusiasts. With Norah, users can analyze and deconstruct reference images into their core elements, including composition, textures, and color palettes. Norah provides insights and inspiration to empower artists to explore new artistic directions.

## Features
- Image Decomposition: Analyzes reference images, breaking them into core components such as composition, structure, and texture.

- Color Palette Generation: Extracts and creates a detailed color palette from reference images, with recommendations for how to recreate these colors using specific mediums like oil paints.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python libraries listed in `src/requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DewSam/Norah-Creative-AI.git
   cd Norah-Creative-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```
3. Install PyTorch:
   - For Windows and macOS:
     ```bash
     pip3 install torch torchvision torchaudio
     ```
   - For Linux:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
   - For more installation options, visit the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

   ### Environment Variables
Create a `.env` file in the root directory of the project and define the following API keys:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

### Running the Application
Start the application using Streamlit:
```bash
streamlit run src/main.py
```

This will launch the application in your default web browser.

## Usage
1. Upload an image that you want Norah to analyze.
2. View the detailed breakdown of the image’s composition, textures, and color palettes.
3. Gain insights and inspiration for your artistic projects.

## Project Structure
```
Norah AI/
├── src/
│   ├── main.py           # Main application entry point
│   ├── tools.py          # Tools used with the LLM
│   ├── requirements.txt  # List of dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

## Notes
- Ensure your `.env` file contains valid API keys.
- The application relies on external APIs and requires an active internet connection.
- Make sure to enable image search for Google CSE (https://programmablesearchengine.google.com/about/)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any inquiries or feedback, please contact us at:
- Email: rdousa2@gmail.com
- GitHub: [Norah AI Repository](https://github.com/DewSam/Norah-Creative-AI)

---

Thank you for using Norah AI! We hope it inspires your creativity.


import os
import cv2
import numpy as np
from PIL import Image
from path import Path
import streamlit as st
from typing import Tuple, List
from preprocessor import Preprocessor
from streamlit_drawable_canvas import st_canvas
from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def infer(model: Model, fn_img: Path) -> None:
    #Recognizes text in image provided by file path.
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    return [recognized, probability]

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def main():

    st.set_page_config(
        page_title = "HTR App",
        page_icon = ":pencil:",
        layout = "centered",
        initial_sidebar_state = "auto",
    )

    st.title('HTR Simple Application')
    
    st.markdown("""
    Streamlit Web Interface for Handwritten Text Recognition (HTR), implemented with TensorFlow and trained on the IAM off-line HTR dataset. The model takes images of single words or text lines (multiple words) as input and outputs the recognized text. 
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Predictions can be made using one of two models:
    - [Model 1](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1) (Trained on Single Word Images) 
    - [Model 2](https://www.dropbox.com/s/7xwkcilho10rthn/line-model.zip?dl=1) (Trained on Text Line Images)    
    """, unsafe_allow_html=True)

    st.subheader('Select a Model, Choose the Arguments and Draw in the box below or Upload an Image to obtain a prediction.')

    modelSelect = st.selectbox("Select a Model", ['Model 1', 'Model 2'])

    decoderSelect = st.selectbox("Select a Decoder", ['Bestpath', 'Beamsearch', 'Wordbeamsearch'])

    modelMapping = {
        "Model 1": "word-model",
        "Model 2": "line-model"
    }

    decoderMapping = {
        'bestpath': DecoderType.BestPath,
        'beamsearch': DecoderType.BeamSearch,
        'wordbeamsearch': DecoderType.WordBeamSearch
    }

    strokeWidth = st.slider("Stroke Width: ", 1, 25, 6)

    inputDrawn = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=strokeWidth,
        update_streamlit=True,
        height = 200,
        width = 400,
        drawing_mode='freedraw',
        key="canvas",
    )

    inputBuffer = st.file_uploader("Upload an Image", type=["png"])

    inferBool = st.button("Infer")

    if ((inputDrawn.image_data is not None or inputBuffer is not None) and inferBool == True):
        
        if inputDrawn.image_data is not None:
            inputArray = np.array(inputDrawn.image_data)
        
        if inputBuffer is not None:
            inputBufferImage = Image.open(inputBuffer)
            inputArray = np.array(inputBufferImage)

        inputImage = Image.fromarray(inputArray.astype('uint8'), 'RGBA')
        inputImage.save('userInput.png')
        charListDir = '../model/line-model/charList.txt'
        charListDir = '../model/word-model/charList.txt'
        decoderType = decoderMapping[decoderSelect.lower()]

        model = Model(charListDir, decoderType, must_restore=True, dump = 'store_true')
        inferedText = infer(model, 'userInput.png')
        
        st.write("**3 Best Candidates: **", inferedText[0][0])
        st.write("**Probabilities: **", str(inferedText[0][0]*100) + "%")
        
        """
        cv2InputImage = cv2.imread('userInput.png', cv2.IMREAD_UNCHANGED)    
        transpMask = cv2InputImage[:,:,3] == 0
        #replace areas of transparency with white and not transparent
        cv2InputImage[transpMask] = [255, 255, 255, 255]

        #new image without alpha channel...
        finalImageArray = cv2.cvtColor(cv2InputImage, cv2.COLOR_BGRA2BGR)
        finalImage = Image.fromarray(finalImageArray.astype('uint8'), 'RGBA')
        finalImage.save('userInput.png')
        """

if __name__ == "__main__":
    main()
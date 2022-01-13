import streamlit as st
from streamlit_drawable_canvas import st_canvas

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

    st.subheader('Select a Model and Draw in the box below to obtain a prediction.')

    model = st.selectbox("Select a Model", ['-','Model 1', 'Model 2'])

    strokeWidth = st.slider("Stroke Width: ", 1, 25, 6)

    text = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=strokeWidth,
        update_streamlit=True,
        drawing_mode='freedraw',
        key="canvas",
    )

if __name__ == "__main__":
    main()
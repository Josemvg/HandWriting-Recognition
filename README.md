# Handwritten Text Recognition with TensorFlow

Small project on developing a Handwritten Text Recognition (HTR) System over a Streamlit Web Application. Using pre-trained models implemented on TensorFlow and trained on the IAM off-line HTR dataset. These models take **images of single words or text lines (multiple words) as input** and **output the recognized text**.

The **images are obtained by extracting it from a box (canvas) in which the user can draw or uploading a file with handwritten text** on the Streamlit app.

## How to use
1. Clone this repo
```
	git clone https://github.com/Josemvg/HandWriting-Recognition
```
2. Install the required Python Libraries
```
	pip install requirements.txt
```
3. Download the pretrained models and extract them on the model folder, under line-word and model-word folders respectively:
   * [Model 1 - Trained on Single Word Images](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1)
   * [Model 2 - Trained on Text Line Images](https://www.dropbox.com/s/7xwkcilho10rthn/line-model.zip?dl=1)
4. Next, to run the WebApp execute runner.py or the streamlit app (from the app folder) by using:
```
	streamlit run webapp.py
```
5. Once you have the app running, you may now select the Model and Decoder or stick to the Default.

   <p align="center"><img src="https://raw.githubusercontent.com/Josemvg/HandWriting-Recognition/master/docs/img/Selectors.png"></p>
   
   **Note:** For a tutorial on how to configure (Word Beam Search Decoder)[https://repositum.tuwien.at/retrieve/1835], please visit the [SimpleHTR Repository](https://github.com/githubharald/SimpleHTR)

   You can start using the app either by writing on the canvas (bear in mind that you can change the stroke width of the pen as well):
	
   <p align="center"><img src="https://raw.githubusercontent.com/Josemvg/HandWriting-Recognition/master/docs/img/Canvas.png"></p>
   
   Or by uploading an image from your own device:

   <p align="center"><img src="https://raw.githubusercontent.com/Josemvg/HandWriting-Recognition/master/docs/img/Upload.png"></p>
   
6. Finally, press the Infer button to obtain the most accurate candidate according to the Model, and its probability.

## References
Models used on this app pertain to the SimpleHTR Repository by Harald Scheidl.
* [How to easily do Handwriting Recognition using Deep Learning](https://nanonets.com/blog/handwritten-character-recognition/)
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
* [SimpleHTR - Handwritten Text Recognition with TensorFlow](https://github.com/githubharald/SimpleHTR)

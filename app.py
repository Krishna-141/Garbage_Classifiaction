from fastai.vision.all import *
import gradio as gr
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  

learn = load_learner("final_model.pkl") 
import gradio as gr

def classify_image(img):
    pred, _, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

interface = gr.Interface(fn=classify_image, inputs="image", outputs="label")

interface.launch()





import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
st.set_page_config(
    page_title="Brain status check",
    page_icon="🧠",
    layout="wide"
)
np.set_printoptions(suppress=True)
st.title("Let's check if your brain is good")
st.write("Upload brain mri scan pic from above to here\nWe'll check for you")
# Load the model
model1=load_model("keras_Model.h5", compile=False)
model2=load_model("",compile=False)
# Load the labels
class_names=open("labels.txt", "r").readlines()
class_name=open("","r").readlines()
upload=st.file_uploader("upload here",type=["jpg","jpeg","png","jfif"])
if upload:
    image=Image.open(upload).convert("RGB")
    st.write("success!")
    data=np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size=(224, 224)
    image=ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
    image_array=np.asarray(image)

# Normalize the image
    normalized_image_array=(image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
    data[0]=normalized_image_array
    prediction=model1.predict(data)
    index=np.argmax(prediction)
    class_namo=class_names[index]
    confidence_score=prediction[0][index]

# Print prediction and confidence score
    st.write("Class:", class_namo[2:], end="")
    st.slider ("Confidence Score:", confidence_score)
    if class_name=="healthy":
        st.write("Your brain looks good, but double check with a professional to see")
    else:
        st.write("Your brain is kinda cooked, checking if you have brain tumor")
        st.__loader__("checking")
        predictions=model2.predict(data)
        index=np.argmax(predictions)
        class_namee=class_name[index]
        confidence_scoring=predictions[0][index]
        st.write("Class:", class_namee[2:], end="")
        st.slider ("Confidence Score:", confidence_scoring)
        st.write("BRO GO CHECK WITH A DOCTOR RIGHT NOW OR UR DEAD")

else:
    st.write("failed, try again")
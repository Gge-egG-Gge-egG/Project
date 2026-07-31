import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import tensorflow as tf
import pandas as pd
st.set_page_config(
    page_title="Brain status check",
    page_icon="🧠",
    layout="wide"
)
np.set_printoptions(suppress=True)
st.title("Let's check if your brain is good")
st.write("Upload brain mri scan pic from above to here")
st.write("We'll check it for you")
# Load the model
model1=tf.keras.models.load_model("model/model1.keras",
                                  custom_objects={"preprocess_input":tf.keras.applications.mobilenet_v2.preprocess_input}
                                  ,safe_mode=False,compile=False)
model2=tf.keras.models.load_model("model2/model2.keras",
                                  custom_objects={"preprocess_input":tf.keras.applications.mobilenet_v2.preprocess_input}
                                  ,safe_mode=False,compile=False)
if model1 and model2:
    print("loading sucessful")
else:
    print("failed")
    exit()
# Yellow background with black text
print("\033[43m Model Loaded \033[0m")

# Load the labels
class_names=open("class_names.txt", "r").readlines()
class_name=open("class_name.txt","r").readlines()
if class_names and class_name:
    print("\033[43m working \033[0m")
else:
    print("idk")
    exit()
upload=st.file_uploader("upload here",type=["jpg","jpeg","png","jfif"])
if upload:
    image=Image.open(upload).convert("RGB")
    st.write("success!")
    data=np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size=(224, 224)
    image=ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    st.write("AI can make mistakes. Do not rely completely on this result, always check with a professor")

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
    st.write("Class:", class_namo[2:])
    st.write ("Confidence Score:", confidence_score*100,"%")
    if "unhealthy" in class_namo:
        st.write("Your brain is probably unhealthy, double checking if any tumor exists")
#        st.__loader__()
        predictions=model2.predict(data)
        index1=np.argmax(predictions)
        class_namee=class_name[index1]
        confidence_scoring=predictions[0][index1]
        st.write("Class:", class_namee[2:])
        st.write ("Confidence Score:", confidence_scoring*100,"%")
        st.write("Your brain is experiencing problems, it's the best to go and see a doctor right now")
    else:
        st.write("Your brain looks good, but double check with a professional to see")
    st.write("Caution: This model is still in training, it makes mistakes, do not 100% trust its results. Always check with a prefessional one.")    

else:
    st.write("No image uploaded")
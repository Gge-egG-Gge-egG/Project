import streamlit as st
import numpy as py
from PIL import Image
import pandas as pd
st.set_page_config(
    page_title="My Aweful App",
    page_icon="🥭",
    layout="wide"
)
st.text_input("eggg")

st.selectbox("6 or 7",["6","7"])
st.title("MEN")
st.write("""
    Lebron James 67
""")
upload=st.file_uploader("upload here",type=["jpg","jpeg","png","jfif","webp"])
if upload:
    image=Image.open(upload)
    st.write("success!")
else:
    st.write("failed, try again")
if st.button("click for balls"):
    st.write("🍡🍡🍡")
    st.balloons()
import streamlit as st

if st.sidebar.button("eggs",type="secondary"):
    st.write("🍤🍤🍤")
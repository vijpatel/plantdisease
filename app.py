import streamlit as st
from PIL import Image
import os

from prediction import predict_disease

st.title("Plant Disease Detection and Identification")
# For newline
#st.write('\n')

#image = Image.open('1.jpg')

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )


if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    path = "tempDir/" + file_details["FileName"]
    u_img = Image.open(uploaded_file)
    #show = st.image(image, use_column_width=True)
    #show.image(u_img, 'Uploaded Image')
    show = st.sidebar.image(u_img)
    with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
      f.write(uploaded_file.getbuffer())
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:    
        st.sidebar.write("Please upload an Image to Classify")
    else:
        with st.spinner('Classifying ...'):
            prediction, result = predict_disease(path)
            st.success('Done!')
            
        st.header("Algorithm Predicts: ")        
        st.write("It's ",prediction )
        st.write('**Probability: **', result, '%')
        os.remove(path)
        #st.sidebar.audio(audio_bytes)

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('skin_cancer_model.h5')

def process_image(img):
    img = img.resize((170,170)) # resize the image to 170x170
    img = np.array(img)
    img = img/255.0 # normalize
    img = np.expand_dims(img,axis=0)
    return img

st.title('Skin Cancer Detection :cancer:')
st.write('Upload an image and model will predict whether it is cancer or not?')

file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])

if file is not None:
    img = Image.open(file)
    st.image(img,caption='uploaded image')
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names = ['This is not Cancer', 'This is Cancer']
    if predicted_class == 0:
        st.success(class_names[predicted_class],icon="✅")
       
    else:
        st.warning(class_names[predicted_class],icon="❌")
       




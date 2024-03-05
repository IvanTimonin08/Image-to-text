from transformers import AutoProcessor, AutoModel
import requests
from PIL import Image
import numpy as np
import streamlit as st

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModel.from_pretrained("microsoft/git-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "this is an image of two cats"

inputs = processor(text, images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

# Преобразование тензора изображения в изображение типа numpy для отображения с помощью st.image()
image_array = np.array(image)
st.image(image_array, caption='Input Image', use_column_width=True)

# Дальнейший код для обработки last_hidden_state и использования его результатов

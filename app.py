import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

#Set the app title

st.title('Classifying Iris Flowers') #Predicting heart disease, predicting diabetes type 2
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')

#Next, we need to include features sliders for the four plant features:

#features

st.header("Plant Features")  #Feature  header

col1, col2 = st.columns(2)   #Sets two  columns  for the features

with col1:
	st.text("Sepal characteristics")
	sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
	sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
	st.text("Pepal characteristics")
	petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
	petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

#prediction button
#st.button("Predict type of Iris")
if st.button("Predict type of Iris"):
	result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
	st.text(result[0])


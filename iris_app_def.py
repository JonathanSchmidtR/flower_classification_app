import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("trained_IRIS_classification_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

primaryColor='#F5B7B1'
backgroundColor="#EBDEF0"
secondaryBackgroundColor="#F4F6F6"
textColor="#31333F"
font="sans serif"


# Create a function to classify the flower
def classify_flower(petal_length, petal_width, sepal_length, sepal_width):
    # Check if the input values are within the specified ranges
    if not (0 <= petal_length <= 10 and 0 <= petal_width <= 10 and 0 <= sepal_length <= 10 and 0 <= sepal_width <= 10):
        return None, None, "Input values are out of range."

    # Scale the input data
    
    input_data = np.array([petal_length, petal_width, sepal_length, sepal_width]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)[0]
    accuracy = model.predict_proba(input_data)
    accuracy_max = np.max(accuracy)*100
    error = None

    return prediction, accuracy_max, error

# Streamlit UI
colT1,colT2 = st.columns([1,8])
with colT2:
    st.title(" Flower Classification App ")
st.write("##")
st.write("This is the Flower Classification App🌷 we will help you classify the type of iris flower based on its petal and sepal dimensions. "
         "Simply adjust the sliders to input the flower's measurements, and the app will tell you which type of iris flower it likely is.")
st.write("##")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('./media/flowermeasure.jpg' , width=250, caption='*This is the correct way to measure a flower')

# Input sliders for petal and sepal dimensions
st.write("## Input Flower Parameters 🌻")
petal_length = st.slider("Petal Length (1 - 7 cm) ", 0.0, 10.0, 3.5, 0.1)
petal_width = st.slider("Petal Width (0.1 - 3 cm) ", 0.0, 10.0, 1.5, 0.1)
sepal_length = st.slider("Sepal Length (4 - 8 cm) ", 0.0, 10.0, 5.0, 0.1)
sepal_width = st.slider("Sepal Width (2 - 5 cm) ", 0.0, 10.0, 3.0, 0.1)

# Classify button
if st.button("Classify!!"):
    prediction, accuracy_max, error_message = classify_flower(petal_length, petal_width, sepal_length, sepal_width)
    
    if prediction is not None:
        #flower_types = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
        result_message = f"Your flower is: **{prediction}** "
        st.write(result_message)
        
        # Display a picture of the flower (You need to have images available)
        # Replace 'iris_setosa.jpg', 'iris_versicolour.jpg', 'iris_virginica.jpg' with your image file paths
        flower_images = {
            'Iris-setosa': './media/irissetosa.jpg',
            'Iris-versicolor': './media/irisversicolour.jpg',
            'Iris-virginica': './media/irisvirginica.jpg'
        }
        #st.image(flower_images[prediction], width=400)
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(flower_images[prediction], width=400)


        st.write(f"**🎯 We are {accuracy_max}% sure this result is correct**")
    else:
        st.error(error_message)


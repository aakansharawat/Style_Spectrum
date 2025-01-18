import streamlit as st
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import io

# Load dataset
file_path = r"C:\Users\prati\Documents\stylespectrum\Skin.csv"
skin_data = pd.read_csv(file_path)

# App title and description
st.title("STYLE SPECTRUM")
st.write("Analyze your skin tone and get color suggestions for different seasons.")

# Prepare data for model
X = skin_data[['R_Value', 'G_Value', 'B_Value']]
y = skin_data['Skin Tone']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
k = 1
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Display model accuracy
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import precision_score, recall_score

# Compute precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")

accuracy = accuracy_score(y_test, y_pred) * 100
st.write(f"Model Accuracy: {accuracy:.2f}%")

# Function to extract average RGB values
def extract_skin_rgb(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_pixels = cv2.bitwise_and(image, image, mask=mask)

    if skin_pixels.any():
        b, g, r = cv2.split(skin_pixels)
        valid_pixels = (b > 0) & (g > 0) & (r > 0)
        if valid_pixels.sum() > 0:
            avg_r = int(np.mean(r[valid_pixels]))
            avg_g = int(np.mean(g[valid_pixels]))
            avg_b = int(np.mean(b[valid_pixels]))
        else:
            st.warning("No valid skin pixels found. Using whole image for estimation.")
            avg_b, avg_g, avg_r = cv2.mean(image)[:3]
    else:
        st.warning("Skin pixels not detected. Using fallback estimation.")
        avg_b, avg_g, avg_r = cv2.mean(image)[:3]

    max_value = max(avg_r, avg_g, avg_b)
    avg_r = int((avg_r / max_value) * 255) if max_value > 0 else 0
    avg_g = int((avg_g / max_value) * 255) if max_value > 0 else 0
    avg_b = int((avg_b / max_value) * 255) if max_value > 0 else 0

    return avg_r, avg_g, avg_b

# Image upload or webcam capture
st.header("Upload or Capture Image")
image_option = st.radio("Choose Image Source", ("Upload Image", "Capture from Webcam"))

if image_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(img)
        if img_np.ndim == 3:
            avg_r, avg_g, avg_b = extract_skin_rgb(img_np)
            st.write(f"Extracted RGB: R={avg_r}, G={avg_g}, B={avg_b}")
        else:
            st.write("Please upload a color image.")
else:
    camera_input = st.camera_input("Capture a photo")
    if camera_input:
        image_bytes = camera_input.getvalue()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="Captured Image", use_column_width=True)
        img_np = np.array(img)
        if img_np.ndim == 3:
            avg_r, avg_g, avg_b = extract_skin_rgb(img_np)
            st.write(f"Extracted RGB: R={avg_r}, G={avg_g}, B={avg_b}")
        else:
            st.write("Please capture a color image.")

# Show season-based color suggestions
if 'avg_r' in locals():
    st.header("Seasonal Color Suggestions")
    season = st.selectbox("Choose a Season", ("Summer", "Winter", "Spring", "Autumn"))

    skin_tone = model.predict(scaler.transform([[avg_r, avg_g, avg_b]]))[0]
    st.write(f"Predicted Skin Tone: {skin_tone}")

    suggestions = skin_data[
        (skin_data['Skin Tone'] == skin_tone) & (skin_data['Season'] == season)
    ][['Color Name', 'Hex Code']].drop_duplicates().head(5)

    if not suggestions.empty:
        st.subheader(f"Suggested Colors for {season}")
        for _, row in suggestions.iterrows():
            st.write(f"**{row['Color Name']}**")
            st.markdown(
                f'<div style="width:50px;height:25px;background-color:{row["Hex Code"]};border:1px solid black;"></div>',
                unsafe_allow_html=True,
            )
    else:
        st.write("No color suggestions available for the selected season.")

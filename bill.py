import os
import cv2
import pandas as pd
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from io import BytesIO

# Initialize PaddleOCR model
ocr_model = PaddleOCR(lang='en')

# Streamlit App
st.title("OCR Text Extraction with PaddleOCR")
st.write("Upload an image to extract text and save it in text or Excel format.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = BytesIO(uploaded_file.read())
    img = Image.open(file_bytes)
    img_path = "uploaded_image.jpg"
    img.save(img_path)
    
    # Run OCR
    result = ocr_model.ocr(img_path)
    boxes = [res[0] for res in result[0]]
    texts = [res[1][0] for res in result[0]]
    scores = [res[1][1] for res in result[0]]

    # Display original image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Visualize OCR results on the image
    img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    font_path = os.path.join("PaddleOCR", "doc", "fonts", "latin.ttf")
    annotated_image = draw_ocr(img_cv, boxes, texts, scores, font_path=font_path)
    st.image(annotated_image, caption="Annotated Image with OCR Results", use_column_width=True)

    # Display extracted text
    st.subheader("Extracted Text")
    extracted_text = "\n".join(texts)
    st.text_area("Extracted Text", extracted_text, height=300)

    # Save extracted text as a file
    st.download_button(
        label="Download Text File",
        data=extracted_text,
        file_name="extracted_text.txt",
        mime="text/plain",
    )

    # Save data to Excel
    data = {"Text": texts, "Confidence": scores}
    df = pd.DataFrame(data)
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="OCR Results")
    st.download_button(
        label="Download Excel File",
        data=excel_buffer.getvalue(),
        file_name="extracted_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Data extracted successfully!")

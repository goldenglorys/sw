import streamlit as st
from model.tiny_yolo import TinyYolo
from barcode import get_barcode


def main():
    st.title("Barcode & QR Code Detection App")
    image_file = st.file_uploader(
        "Please upload your image:", type=["jpg", "jpeg", "png"]
    )
    model = TinyYolo(classes=2)  # 2 classes: Barcode and QR

    if image_file is not None:
        imageText = st.empty()
        imageLocation = st.empty()
        imageText.text("Image uploaded")
        imageLocation.image(image_file)

        image_file.seek(0)
        image_predicted = model.predict(image_file)

        if st.button("Launch detection"):

            if image_predicted is not None:
                imageText.text("Image with barcode/QR code detected")
                imageLocation.image(image_predicted)

                image_file.seek(0)
                code_data, code_type, product_info = get_barcode(image_file)

                if code_data is not None:
                    st.success(f"{code_type}: {code_data}")
                else:
                    st.error(f"Cannot read code")

                if product_info and product_info.get("status"):
                    st.title("Product Info")
                    st.write(product_info)
                elif code_type in [
                    "EAN13",
                    "EAN8",
                    "UPCA",
                    "UPCE",
                    "CODE128",
                    "CODE39",
                ]:
                    st.title("Product Info")
                    st.write("Product not found in database")


if __name__ == "__main__":
    main()

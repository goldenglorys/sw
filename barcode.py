import requests
from pyzbar.pyzbar import decode
from PIL import Image


def get_barcode(image):
    """
    Read barcode/QR code from image and return data with type.

    Returns:
        tuple: (code_data, code_type, product_info)
            - code_data: String/bytes of decoded data
            - code_type: Type of code (EAN13, QRCODE, etc.)
            - product_info: Dict from API (only for product barcodes)
    """
    code_data = None
    code_type = None
    product_info = None

    try:
        img_raw = Image.open(image)
        decoded_objects = decode(img_raw)

        if decoded_objects:
            obj = decoded_objects[0]

            code_type = obj.type

            try:
                code_data = obj.data.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                code_data = str(obj.data)

            if code_type in ["EAN13", "EAN8", "UPCA", "UPCE", "CODE128", "CODE39"]:
                try:
                    numeric_code = "".join(filter(str.isdigit, code_data))
                    if numeric_code:
                        product_info = _get_product_info(numeric_code)
                except Exception as e:
                    print(f"Error fetching product info: {e}")

    except Exception as e:
        print(f"Error decoding: {e}")

    return code_data, code_type, product_info


def _get_product_info(barcode):
    """
    Query OpenFoodFacts API for product information.
    Only works for food product barcodes.
    """
    try:
        address = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        response = requests.get(address, timeout=5)
        return response.json()
    except Exception as e:
        print(f"API request failed: {e}")
        return None

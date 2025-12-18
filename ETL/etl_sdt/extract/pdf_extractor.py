from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import re

from etl_sdt.utils.logging_config import logger

def extract_text_from_pdf(pdf_path, detect_vertical_text=False, word_margin=0.1):
    """
    Extracts text from a PDF file.

    Parameters:
    pdf_path (str): The path to the PDF file.
    detect_vertical_text (bool): If True, detects vertical text.
    word_margin (float): The margin between words to consider them separate.

    Returns:
    str: The extracted text.
    """
    try:
        laparams = LAParams()
        laparams.detect_vertical = detect_vertical_text
        laparams.word_margin = word_margin
        
        # Extract text using pdfminer
        text = extract_text(pdf_path, laparams=laparams)
        
        # Clean up the extracted text
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        return ""

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return ""

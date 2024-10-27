# import pytesseract
# from PIL import Image
# import os

# # Set the Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# # Directory containing all images
# img_dir = r"C:\Users\shiva\Downloads\dataset\dataset\train\images"

# # Loop through all files in the directory
# for filename in os.listdir(img_dir):
#     if filename.endswith(".jpg"):  # Check if the file is a .jpg image
#         img_path = os.path.join(img_dir, filename)  # Full path to the image
#         img = Image.open(img_path)
        
#         # Perform OCR on the image
#         text = pytesseract.image_to_string(img)
        
#         # Print the filename and the extracted text
#         print(f"Text from {filename}:\n{text}\n")

import pytesseract
from PIL import Image
import os
import re
from typing import Dict, Any, Optional
import logging
import pandas as pd

class W2FormExtractor:
    def __init__(self, tesseract_path: str = r"tesseract.exe"):
        """Initialize the W2 Form Extractor."""
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.img_dir = r"C:\Users\shiva\Downloads\dataset\dataset\train\images"
        
        # Define patterns for required fields
        self.field_patterns = {
            'employerName': r'(?:employer[\'s]*\s+name)[\s:]*([\w\s&,.-]+)',
            'employerAddressStreet_name': r'(?:address|street)[\s:]*([\w\s,.-]+)',
            'employerAddressCity': r'(?:city)[\s:]*([\w\s.-]+)',
            'employerAddressState': r'(?:state)[\s:]*([A-Z]{2})',
            'employerAddressZip': r'(?:zip)[\s:]*(\d{5}(?:-\d{4})?)',
            'einEmployerIdentificationNumber': r'(?:ein|employer[\'s]*\s+id)[\s:]*(\d{2}-\d{7})',
            'employeeName': r'(?:employee[\'s]*\s+name)[\s:]*([\w\s.-]+)',
            'ssnOfEmployee': r'(?:ssn|social)[\s:]*(\d{3}-\d{2}-\d{4})',
            'box1WagesTipsAndOtherCompensations': r'(?:box\s*1|wages)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'box2FederalIncomeTaxWithheld': r'(?:box\s*2|federal)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'box3SocialSecurityWages': r'(?:box\s*3|social)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'box4SocialSecurityTaxWithheld': r'(?:box\s*4|ss\s*tax)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'box16StateWagesTips': r'(?:box\s*16|state\s*wages)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'box17StateIncomeTax': r'(?:box\s*17|state\s*tax)[\s:]*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'taxYear': r'(?:year|tax\s+year)[\s:]*(\d{4})'
        }

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        if not text:
            return None
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned

    def clean_amount(self, text: str) -> str:
        """Clean monetary amounts."""
        if not text:
            return None
        # Remove all non-numeric characters except decimal points
        cleaned = re.sub(r'[^\d.]', '', text)
        try:
            return f"{float(cleaned):.2f}"
        except:
            return None

    def extract_form_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract data from a W2 form image."""
        try:
            # Open and process image
            img = Image.open(image_path)
            
            # Perform OCR with custom configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, config=custom_config)
            text = text.lower()  # Convert to lowercase for easier matching
            
            # Initialize results dictionary
            extracted_data = {}
            
            # Extract each field using regex patterns
            for field, pattern in self.field_patterns.items():
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).strip()
                    # Apply appropriate cleaning based on field type
                    if any(keyword in field.lower() for keyword in ['box', 'wages', 'tax']):
                        value = self.clean_amount(value)
                    else:
                        value = self.clean_text(value)
                    extracted_data[field] = value
                else:
                    extracted_data[field] = None
                    logging.warning(f"Could not extract {field} from {image_path}")
            
            return extracted_data

        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self):
        """Process all images in the directory (original functionality)."""
        results = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, filename)
                extracted_data = self.extract_form_data(img_path)
                if extracted_data:
                    print(f"\nExtracted data from {filename}:")
                    for field, value in extracted_data.items():
                        print(f"{field}: {value}")
                    results.append(extracted_data)
        return results

def main():
    # Initialize extractor
    extractor = W2FormExtractor()
    
    # Process all images
    results = extractor.process_directory()
    
    # Save results to TSV (optional)
    if results:
        df = pd.DataFrame(results)
        df.to_csv('extracted_results.tsv', sep='\t', index=False)
        print("\nResults saved to extracted_results.tsv")

if __name__ == "__main__":
    main()
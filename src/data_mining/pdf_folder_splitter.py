import fitz  # PyMuPDF
import os
import shutil


def separate_pdfs(input_folder, output_folder):
    # Create a folder for valid PDFs if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(input_folder, filename)
            try:
                # Try opening the PDF to see if it's valid
                with fitz.open(file_path) as pdf_doc:
                    print(f"Valid PDF: {filename}")
                    # Move the valid PDF to the output folder
                    shutil.move(file_path, os.path.join(output_folder, filename))
            except Exception as e:
                print(f"Damaged PDF: {filename} ({str(e)})")
                # Skip moving damaged PDFs


# Example usage
input_folder = "/Users/timurabdygulov/Desktop/My Computer/GWU Spring 2024/NLP/pythonProject/Capstone Project Fall 2024 /downloaded_pdfs"
output_folder = "path_to_store_valid_pdfs"

separate_pdfs(input_folder, output_folder)

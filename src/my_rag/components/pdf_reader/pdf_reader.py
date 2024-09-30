import PyPDF2
import os


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Initialize a variable to store the extracted text
            extracted_text = ""

            # Loop through each page and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()

            return extracted_text
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except Exception as e:
        return f"An error occurred: {e}"


# Main program
if __name__ == "__main__":
    # Ask the user for the PDF file path
    pdf_path = input("Please enter the full path to the PDF file: ")

    # Check if the file exists
    if not os.path.exists(pdf_path):
        print("Error: The specified file does not exist.")
    else:
        # Extract the text
        text = extract_text_from_pdf(pdf_path)

        # Print the extracted text or save to a file
        choice = input("Do you want to (1) print the text or (2) save it to a file? Enter 1 or 2: ")

        if choice == '1':
            # Print the extracted text
            print("\nExtracted Text:\n")
            print(text)
        elif choice == '2':
            # Ask for the output file path
            output_path = input("Please enter the full path for the output text file (e.g., output.txt): ")
            try:
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(text)
                print(f"Text successfully saved to {output_path}")
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")
        else:
            print("Invalid choice. Exiting the program.")

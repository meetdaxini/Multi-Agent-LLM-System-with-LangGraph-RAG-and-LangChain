import os
import pandas as pd

def filter_csv_by_pdf_reference(csv_file, pdf_folder, output_csv):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Strip any leading or trailing spaces from the 'PDF Reference' column
    df['PDF Reference'] = df['PDF Reference'].str.strip()

    # Get the list of PDF filenames (with extensions) in the pdf_folder
    valid_pdfs = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    # Filter the DataFrame to include only rows where the 'PDF Reference' column matches the valid PDFs
    filtered_df = df[df['PDF Reference'].isin(valid_pdfs)]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        print("No matching PDFs were found. Please check the PDF names or CSV content.")
    else:
        # Save the filtered DataFrame to a new CSV file
        filtered_df.to_csv(output_csv, index=False)
        print(f"Filtered data has been saved to {output_csv}")


# Example usage
csv_file = "/Users/timurabdygulov/Desktop/My Computer/GWU Spring 2024/NLP/pythonProject/Capstone Project Fall 2024 /dataset.csv"
pdf_folder = "/Users/timurabdygulov/Desktop/My Computer/GWU Spring 2024/NLP/pythonProject/Capstone Project Fall 2024 /path_to_store_valid_pdfs"
output_csv = "filtered_dataset.csv"

filter_csv_by_pdf_reference(csv_file, pdf_folder, output_csv)

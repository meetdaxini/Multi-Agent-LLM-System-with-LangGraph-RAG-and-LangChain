import json
import requests
from bs4 import BeautifulSoup
import os
import sys
import csv
from urllib.parse import urljoin, quote
from tqdm import tqdm


def read_json_file(file_path):
    """Reads the JSON file and returns the data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)


def extract_doi(pubmed_url):
    """Fetches the PubMed page and extracts the DOI."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pubmed_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the DOI in meta tags
        doi_meta = soup.find('meta', {'name': 'citation_doi'})
        if doi_meta:
            doi = doi_meta.get('content')
            doi_url = f"https://doi.org/{doi}"
            return doi_url, doi
        else:
            print(f"DOI not found on page: {pubmed_url}")
            return None, None
    except Exception as e:
        print(f"Error fetching PubMed page: {e}")
        return None, None


def find_pdf_link(html_content, base_url):
    """Attempts to find a PDF link on the publisher's page."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Check for PDF link in meta tags
    pdf_meta = soup.find('meta', {'name': 'citation_pdf_url'})
    if pdf_meta:
        pdf_url = pdf_meta.get('content')
        pdf_url = urljoin(base_url, pdf_url)
        return pdf_url

    # Look for links that point directly to PDFs
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            pdf_url = urljoin(base_url, href)
            return pdf_url

    # Additional parsing logic can be added here

    return None


def download_pdf(pdf_url, doi, output_folder):
    """Downloads the PDF from the given URL into the specified folder."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, stream=True)
        response.raise_for_status()

        # Sanitize filename
        filename = f"{doi.replace('/', '_')}.pdf"
        filename = quote(filename, safe='')

        # Save the PDF
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded PDF: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to download PDF from {pdf_url}: {e}")
        return None


def process_documents(documents, question_body, ideal_answer, output_folder, csv_writer, question_id):
    """Processes each document URL to download the PDFs and write to CSV."""
    for pubmed_url in documents:
        print(f"\nProcessing: {pubmed_url}")
        doi_url, doi = extract_doi(pubmed_url)
        if doi_url and doi:
            try:
                # Follow the DOI link to the publisher's page
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(doi_url, headers=headers)
                response.raise_for_status()

                # Attempt to find the PDF link
                pdf_url = find_pdf_link(response.text, response.url)
                if pdf_url:
                    print(f"Found PDF link: {pdf_url}")
                    pdf_filename = download_pdf(pdf_url, doi, output_folder)
                    if pdf_filename:
                        # Write to CSV
                        csv_writer.writerow({
                            'Question ID': question_id,
                            'Question': question_body,
                            'Ideal Answer': ' '.join(ideal_answer),
                            'Download Link': pdf_url,
                            'PDF Reference': pdf_filename
                        })
                    else:
                        print(f"Failed to download PDF for DOI: {doi}")
                else:
                    print(f"No PDF link found on publisher's page for DOI: {doi}")
            except Exception as e:
                print(f"Error processing DOI link: {e}")
        else:
            print(f"Failed to extract DOI from PubMed URL: {pubmed_url}")


def main():
    # Get the JSON file path from user input
    json_file_path = input("Enter the path to the JSON file: ")

    # Get the question limit from user input
    limit_input = input("Enter the number of questions to process (leave blank for all): ")
    question_limit = None
    if limit_input.strip():
        try:
            question_limit = int(limit_input)
            if question_limit <= 0:
                print("Please enter a positive integer for the limit.")
                sys.exit(1)
        except ValueError:
            print("Invalid input. Please enter a valid integer for the limit.")
            sys.exit(1)

    # Create output folder for PDFs
    output_folder = 'downloaded_pdfs'
    os.makedirs(output_folder, exist_ok=True)

    # Read the JSON data
    data = read_json_file(json_file_path)
    questions = data.get('questions', [])

    if not questions:
        print("No questions found in the JSON file.")
        sys.exit(1)

    # Apply question limit if specified
    if question_limit is not None:
        questions = questions[:question_limit]

    # Prepare CSV file
    csv_file_path = 'dataset.csv'
    csv_fields = ['Question ID', 'Question', 'Ideal Answer', 'Download Link', 'PDF Reference']
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        csv_writer.writeheader()

        # Process each question
        for idx, question in enumerate(questions):
            question_id = idx + 1
            body = question.get('body', '')
            documents = question.get('documents', [])
            ideal_answer = question.get('ideal_answer', [])
            print(f"\nProcessing question {question_id}: {body}")

            # Ensure the documents list is not empty
            if not documents:
                print("No documents found for this question.")
                continue

            # Process each document to download PDFs and write to CSV
            process_documents(documents, body, ideal_answer, output_folder, csv_writer, question_id)

    print("\nProcessing completed. Dataset saved to 'dataset.csv'.")


if __name__ == '__main__':
    main()



# hopgropter/Capstone Project Fall 2024/Data/BioASQ-training11b/training11b.json
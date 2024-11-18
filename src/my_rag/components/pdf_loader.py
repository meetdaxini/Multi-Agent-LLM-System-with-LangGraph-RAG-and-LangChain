from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
import os


class PDFLoader:
    """A class to handle PDF document loading and processing."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the PDFLoader.

        Args:
            base_dir (Optional[str]): Base directory for PDF files
        """
        self.base_dir = base_dir

    def load_single_pdf(self, pdf_path: str) -> str:
        """
        Load a single PDF and return its text content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Text content of the PDF
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[str]:
        """
        Load multiple PDFs and return their text contents.

        Args:
            pdf_paths (List[str]): List of paths to PDF files

        Returns:
            List[str]: List of text contents from PDFs
        """
        pdf_contents = {}
        for pdf_path in pdf_paths:
            try:
                content = self.load_single_pdf(pdf_path)
                pdf_contents[pdf_path] = content
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {str(e)}")
        return pdf_contents

    def load_pdfs_from_directory(self, directory: str) -> dict:
        """
        Load all PDFs from a directory.

        Args:
            directory (str): Directory containing PDF files

        Returns:
            dict: Dictionary mapping PDF filenames to their contents
        """
        pdf_contents = {}
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                try:
                    content = self.load_single_pdf(file_path)
                    pdf_contents[filename] = content
                except Exception as e:
                    print(f"Error loading PDF {filename}: {str(e)}")
        return pdf_contents

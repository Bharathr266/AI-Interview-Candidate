import PyPDF2

def parse_resume(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

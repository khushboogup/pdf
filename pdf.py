import warnings
warnings.filterwarnings("ignore")
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
import os

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCUTRMxVYk9Pmjp8Zy9xKouoUUUeZC3FQQ"

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except FileNotFoundError:
        return "Error: The file 'sample1.pdf' was not found."

# Define prompt template
template = """
You are an expert AI assistant. Use the information provided for answering the question
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Initialize Gemini LLM and chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Function to answer questions
def answer_question(pdf_text, question):
    if not pdf_text:
        return "Error: No text extracted from the PDF."
    answer = qa_chain.run(context=pdf_text, question=question)
    return answer

# Example usage
if __name__ == "__main__":
    pdf_path = "sample1.pdf"  # Replace with your PDF file path
    pdf_text = extract_text_from_pdf(pdf_path)
    question = input("Enter text:")
    answer = answer_question(pdf_text, question)
    print(f"Question: {question}\nAnswer: {answer}")
#print(len(pdf_text))




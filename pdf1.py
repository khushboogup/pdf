import warnings
warnings.filterwarnings("ignore")
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
import os

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "api key"

# Extract text from multiple PDFs
def extract_text_from_pdf(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:  # Iterate over the list of PDF paths
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"  # Add newline to separate text from different PDFs
        except FileNotFoundError:
            text += f"Error: The file '{pdf_path}' was not found.\n"
    return text

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
qa_chain = RunnableSequence(prompt | llm)  # Updated to use RunnableSequence

# Function to answer questions
def answer_question(pdf_text, question):
    if not pdf_text:
        return "Error: No text extracted from the PDFs."
    answer = qa_chain.invoke({"context": pdf_text, "question": question})  # Updated to use invoke
    return answer.content if hasattr(answer, 'content') else answer  # Handle response content

# Example usage
if __name__ == "__main__":
    pdf_paths = ["sample1.pdf","sample2.pdf"]  # Replace with your list of PDF file paths
    pdf_text = extract_text_from_pdf(pdf_paths)  # Pass the list of PDF paths
    question = input("Enter text: ")
    answer = answer_question(pdf_text, question)
    print(f"Question: {question}\nAnswer: {answer}")
    # print(len(pdf_text))  # Uncomment to print the length of extracted text
import json

def read_config():
    with open('config.json') as f:
        data = json.load(f)
    return data

#First part

import fitz # Part of the PyMuPDF library, used to handle PDF text extraction
import openai # The OpenAI library for interacting with the GPT-3 or GPT-4 API

# Uses fitz to extract text from each page of the PDF
def extract_text_from_pdf(pdf_path):
    # Empty string to store extracted text
    text = ""
    # Opens the PDF document
    pdf_document = fitz.open(pdf_path)
    # Iterates over each page in the PDF
    for page_num in range(len(pdf_document)):
        # Loads the page 
        page = pdf_document.load_page(page_num)
        #  Appends the extracted text from the page to the text string
        text += page.get_text()
    # Closes the PDF document
    pdf_document.close()
    return text

# Construct and send prompt to LLM API
def call_llm_api(text):
    # Reads the configuration file to get the API key
    config = read_config()
    # Extracts the OpenAI API key from the configuration
    api_key = config['api_key']
    # Sets the API key for OpenAI API access
    openai.api_key = api_key
    
    # Constructs a prompt to ask the LLM for specific details from the CV
    # Includes the required information needed from the LLM
    prompt = (
        "Extract and summarize the following key information from the candidate's CV:\n\n"
        f"{text}\n\n"
        "Please provide the following information:\n"
        "1. Candidate's name\n"
        "2. Key skills (list of 5-10 most relevant skills)\n"
        "3. Years of experience\n"
        "4. Education level\n"
        "5. Most recent job title and company\n"
        "6. A brief summary of the candidate's profile (2-3 sentences).\n\n"
        "Format your response as follows:\n"
        "Name: [Name]\n"
        "Key Skills: [Skill1, Skill2, ...]\n"
        "Years of Experience: [Number]\n"
        "Education Level: [Degree]\n"
        "Most Recent Job Title and Company: [Job Title at Company]\n"
        "Summary: [Brief summary]"
    )
    
    # Sends the prompt to the LLM API and retrieve a response
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, # Passes the constructed prompt
        max_tokens=300,  # Limits the response length to 300 tokens
        temperature=0.5  # Controls the creativity of the response
    )
    
    # Returns the response text after removing leading/trailing whitespace
    return response.choices[0].text.strip()

# Main function - Combines text extraction and API call, then returns the structured information.
def cv_analysis(pdf_path):
    # Extracts text from the provided PDF file
    text = extract_text_from_pdf(pdf_path)
    # Retrieves structured information from the LLM API based on the extracted text
    structured_info = call_llm_api(text)
    return structured_info





# Second part

def evaluate_candidate_fit_LLM(job_description, parsed_cv_info):
    # input: 
    # 'job_description' : A string describing the job role and its requirements
    # 'parsed_cv_info': A string containing structured information extracted from the candidate's CV
    
    # function to load configuration data
    config = read_config()
    api_key = config['api_key']
    openai.api_key = api_key
    
    # Build a detailed prompt for the LLM
    prompt = (
        "Based on the following job description and the candidate's CV information, "
        "evaluate the candidate's fit for the role and provide a summary. "
        "Classify the candidate into one of the following categories:\n"
        "A: Good fit\n"
        "B: Medium fit\n"
        "C: Not a good fit\n\n"
        "Job Description:\n"
        f"{job_description}\n\n"
        "Candidate CV Information:\n"
        f"{parsed_cv_info}\n\n"
        "Provide a summary of the candidate's fit for the role and classify their fit as follows:\n"
        "Summary: [Brief summary]\n"
        "Fit Category: [A/B/C]"
    )
    
    # Sends the constructed prompt to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5
    )
    
    return response.choices[0].text.strip()

def evaluate_fit(pdf_path, job_description):
    parsed_cv_info = cv_analysis(pdf_path)
    fit_summary = evaluate_candidate_fit_LLM(job_description, parsed_cv_info)
    print(fit_summary)

# As an example:
pdf_path = "path/to/candidate_cv.pdf"
job_description = "We are looking for a software engineer with experience in Python and machine learning. The ideal candidate should have strong problem-solving skills and experience with data analysis."

evaluate_fit(pdf_path, job_description)
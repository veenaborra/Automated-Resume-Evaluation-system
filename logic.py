import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    #print("configured successfully")
else:
    print("Google API Key not found. Please add it to your .env file.")



def get_hard_match_score(jd_text, resume_text): 
    """
    Calculates a score based on exact keyword matches.
    """
   
    print("Calculating Hard Match Score...")
    
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = f"""
    Based on the following job description, list the top 15 most important technical skills, tools, and qualifications. 
    Output them as a single, comma-separated string. For example: Python,React,SQL,AWS,Project Management

    Job Description:
    {jd_text}
    """

    keywords_string = llm.invoke(prompt)

    keywords = [keyword.strip().lower() for keyword in keywords_string.split(',')]

    found_keywords = 0
    resume_text_lower = resume_text.lower()
    
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text_lower):
            found_keywords += 1
            
    score = (found_keywords / len(keywords)) * 100 if keywords else 0
    return score

def get_semantic_match_score(jd_text, resume_text):
    """
    Calculates a score based on the semantic similarity of the resume and JD.
    """
    print("Calculating Semantic Match Score...")
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    jd_chunks = text_splitter.split_text(jd_text)

   
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    vector_store = FAISS.from_texts(jd_chunks, embedding=embeddings)

  
    docs = vector_store.similarity_search(resume_text, k=3)

  
    chain = load_qa_chain(GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3), chain_type="stuff")
    prompt = f"""
    You are an expert resume evaluator. Based ONLY on the provided job description context and the candidate's resume,
    what is the semantic relevance score out of 100? A score of 100 means a perfect contextual fit.
    Your response must be a single number. Do not add any extra text or symbols.

    Resume:
    {resume_text}
    """
    response = chain.invoke({"input_documents": docs, "question": prompt})
    
    
    score_str = re.search(r'\d+', response['output_text'])
    return int(score_str.group(0)) if score_str else 0



def generate_feedback(jd_text, resume_text):
    """
    Generates personalized feedback for the candidate.
    """
    print("Generating Feedback...")
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt = f"""
    You are an expert career coach for the tech industry.
    Based on the job description and the candidate's resume below, provide a short, encouraging, and constructive feedback paragraph.
    Specifically, highlight 2-3 key skills, technologies, or project types that are mentioned in the job description but seem to be missing or underrepresented in the resume.
    
    Example format: "This is a strong resume. To be an even better fit for this role, consider highlighting projects involving [Missing Skill 1] or gaining experience with [Missing Skill 2]."

    Job Description:
    {jd_text}

    Resume:
    {resume_text}
    """
    feedback = llm.invoke(prompt)
    return feedback




# if __name__ == '__main__':
#     # --- Sample Data for Testing ---
#     sample_jd = """
#     Job Title: Python Developer
#     Location: Bangalore, India
#     Description: We are looking for a Python Developer with 3+ years of experience in building web applications.
#     Responsibilities:
#     - Develop and maintain scalable web services using Flask or Django.
#     - Work with PostgreSQL databases.
#     - Deploy applications on AWS.
#     - Write unit and integration tests.
#     Required Skills: Python, Flask, Django, SQL, PostgreSQL, AWS, Git.
#     """

#     sample_resume = """
#     John Doe
#     Software Engineer

#     Experience:
#     - Built web applications using Python and Flask.
#     - Managed databases with SQL.
#     - Familiar with cloud services.

#     Skills:
#     - Languages: Python, JavaScript
#     - Frameworks: Flask
#     - Databases: SQL
#     """
    
#     # --- Run the Functions ---
#     hard_score = get_hard_match_score(sample_jd, sample_resume)
#     print(f"\nHard Match Score: {hard_score}%")
    
#     semantic_score = get_semantic_match_score(sample_jd, sample_resume)
#     print(f"Semantic Match Score: {semantic_score}%")
    
#     feedback = generate_feedback(sample_jd, sample_resume)
#     print(f"Generated Feedback:\n{feedback}")
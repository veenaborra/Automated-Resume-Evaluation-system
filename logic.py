import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



from langchain_google_genai import  GoogleGenerativeAI




load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    #print("configured successfully")
else:
    print("Google API Key not found. Please add it to your .env file.")





def get_hard_match_score(jd_text, resume_text, skill_vocab=None):
    """
    Calculates a hard match score between Job Description and Resume.

    Parameters:
    - jd_text (str): Job description text
    - resume_text (str): Candidate resume text
    - skill_vocab (list, optional): predefined list of skills to look for. 
      If None, top keywords are extracted from JD using TF-IDF.

    Returns:
    - score (float): hard match score (0-100)
    - jd_keywords (list): keywords extracted from JD
    - found_keywords (list): keywords present in resume
    - missing_keywords (list): keywords missing in resume
    """
    print("Calculating Hard Match Score...")

    jd_text_lower = jd_text.lower()
    resume_text_lower = resume_text.lower()

   
    if skill_vocab:
        jd_keywords = [s.lower() for s in skill_vocab if s.lower() in jd_text_lower]
    else:
      
        vectorizer = TfidfVectorizer(stop_words="english", max_features=15)
        vectorizer.fit([jd_text])
        jd_keywords = vectorizer.get_feature_names_out().tolist()

   
    found_keywords = set()
    for keyword in jd_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text_lower):
            found_keywords.add(keyword)

   
   
    score = (len(found_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0

    
    missing_keywords = set(jd_keywords) - found_keywords

    return round(score, 2), jd_keywords, list(found_keywords), list(missing_keywords)



def get_semantic_match_score(jd_text, resume_text):
    """
    Calculates a semantic similarity score (0-100) between JD and Resume
    using simple term-frequency vectors + cosine similarity (no TF-IDF).
    """
    print("Calculating Semantic Match Score (Vector + Cosine)...")
    
    # Combine JD and Resume into one corpus
    corpus = [jd_text.lower(), resume_text.lower()]  # lowercase for uniformity
    
    # Convert text to term-frequency vectors
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(corpus)
    

    similarity = cosine_similarity(tf_matrix[0:1], tf_matrix[1:2])[0][0]
    
  
    score = round(similarity * 100, 2)
    return score



def generate_feedback(jd_text, resume_text):
    """
    Generates personalized feedback for the candidate.
    missing_skills: list of skills missing from resume (optional)
    """
    print("Generating Feedback...")
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    
    
    prompt = f"""
    You are an expert career coach for the tech industry.
    Based on the job description and the candidate's resume below, provide a short, encouraging, and constructive feedback paragraph.
    Highlight 2-3 key skills, technologies, or project types that are mentioned in the job description but seem to be missing or underrepresented in the resume.
    
    
    Job Description:
    {jd_text}

    Resume:
    {resume_text}
    """
    feedback = llm.invoke(prompt)
    return feedback

def evaluate_resume(jd_text, resume_text, skill_vocab=None, hard_weight=0.5, semantic_weight=0.5):
    """
    Evaluate a resume against a Job Description (JD) and return all results in a structured dictionary.
    
    Parameters:
    - jd_text (str): Job Description text
    - resume_text (str): Candidate Resume text
    - skill_vocab (list, optional): List of predefined skills for hard match
    - hard_weight (float): Weight for hard match score (default 0.5)
    - semantic_weight (float): Weight for semantic match score (default 0.5)
    
    Returns:
    - dict with keys:
        'hard_score', 'semantic_score', 'final_score', 'verdict',
        'missing_skills', 'found_skills', 'feedback'
    """

    hard_score, jd_keywords, found_keywords, missing_keywords = get_hard_match_score(
        jd_text, resume_text, skill_vocab
    )
    
  
    semantic_score = get_semantic_match_score(jd_text, resume_text)
    
 
    final_score = round(hard_score * hard_weight + semantic_score * semantic_weight, 2)
    

    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"
    
    
    feedback = generate_feedback(jd_text, resume_text)
    
   
    result = {
        "hard_score": hard_score,
        "semantic_score": semantic_score,
        "final_score": final_score,
        "verdict": verdict,
        "missing_skills": list(missing_keywords),
        "found_skills": list(found_keywords),
        "feedback": feedback
    }
    
    return result



# if __name__ == '__main__':
   
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

    
#     result = evaluate_resume(sample_jd, sample_resume)

#     # --- Print Results in a Friendly Format ---
#     print("\n=== Resume Evaluation Result ===")
#     print(f"Hard Match Score: {result['hard_score']}%")
#     print(f"Semantic Match Score: {result['semantic_score']}%")
#     print(f"Final Relevance Score: {result['final_score']}%")
#     print(f"Verdict: {result['verdict']}")
#     print(f"Found Skills: {', '.join(result['found_skills'])}")
#     print(f"Missing Skills: {', '.join(result['missing_skills'])}")
#     print(f"Feedback:\n{result['feedback']}")

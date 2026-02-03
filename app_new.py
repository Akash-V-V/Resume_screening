import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import PyPDF2
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-container {
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Global variables
stop_words = set(stopwords.words('english'))
hard_skill_keywords = [
    'python', 'java', 'c++', 'javascript', 'sql', 'r', 'go', 'rust',
    'html', 'css', 'react', 'node', 'django', 'flask', 'spring',
    'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
    'data science', 'data analysis', 'statistics', 'big data', 'hadoop', 'spark',
    'mysql', 'postgresql', 'mongodb', 'redis', 'database',
    'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'git', 'linux', 'unix', 'agile', 'scrum',
    'tableau', 'power bi', 'excel','web design', 'ui design', 'ux design', 'figma', 'sketch', 'adobe xd',
    'photoshop', 'illustrator', 'responsive design', 'wireframing', 'prototyping'
]

soft_skill_keywords = [
    'communication', 'teamwork', 'leadership', 'management',
    'problem solving', 'critical thinking', 'analytical',
    'time management', 'adaptability', 'creativity',
    'decision making', 'emotional intelligence', 'presentation',
    'collaboration', 'mentoring', 'initiative'
]
def clean_text(text):
    """Enhanced text cleaning function"""
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)  # Remove phone numbers
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    # Keep words that are longer than 2 characters or not stopwords
    text = ' '.join([word for word in text.split() 
                     if word not in stop_words or len(word) > 2])
    return text.strip()

def extract_skills(text):
    """Enhanced skill extraction"""
    text_lower = str(text).lower()
    hard_skills = [skill for skill in hard_skill_keywords if skill in text_lower]
    soft_skills = [skill for skill in soft_skill_keywords if skill in text_lower]
    return ' '.join(hard_skills + soft_skills)

def predict_resume_lstm(resume_text, model_cat, model_skill, tokenizer, le_cat, le_type, max_len=200):
    """Predict resume category and skill type"""
    # Clean the text
    cleaned = clean_text(resume_text)
    
    # Check if cleaned text is empty
    if not cleaned or len(cleaned.strip()) == 0:
        # Return default values if text is empty
        return {
            'category': 'Unknown',
            'category_confidence': '0.00%',
            'top_3_categories': [('Unknown', 0.0), ('Unknown', 0.0), ('Unknown', 0.0)],
            'skill_type': 'Unknown',
            'skill_type_confidence': '0.00%'
        }
    
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([cleaned])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    
    # Predict category
    category_pred_probs = model_cat.predict(pad_seq, verbose=0)
    category_idx = np.argmax(category_pred_probs, axis=1)[0]
    category_pred = le_cat.inverse_transform([category_idx])[0]
    category_confidence = float(category_pred_probs[0][category_idx])
    
    # Get top 3 predictions for category
    top_3_idx = np.argsort(category_pred_probs[0])[-3:][::-1]
    top_3_categories = [(le_cat.inverse_transform([idx])[0], 
                        float(category_pred_probs[0][idx])) for idx in top_3_idx]
    
    # Predict skill type
    skill_type_pred_probs = model_skill.predict(pad_seq, verbose=0)
    skill_type_idx = np.argmax(skill_type_pred_probs, axis=1)[0]
    skill_type_pred = le_type.inverse_transform([skill_type_idx])[0]
    skill_type_confidence = float(skill_type_pred_probs[0][skill_type_idx])

    return {
        'category': category_pred,
        'category_confidence': f"{category_confidence:.2%}",
        'top_3_categories': top_3_categories,
        'skill_type': skill_type_pred,
        'skill_type_confidence': f"{skill_type_confidence:.2%}"
    }

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Load models and encoders
@st.cache_resource
def load_models():
    """Load all required models and encoders"""
    try:
        with open('label_encoder_cat.pkl', 'rb') as f:
            label_encoder_cat = pickle.load(f)
        with open('label_encoder_type.pkl', 'rb') as f:
            label_encoder_type = pickle.load(f)
        with open('model_category.pkl', 'rb') as f:
            model_category = pickle.load(f)
        with open('model_skill_type.pkl', 'rb') as f:
            model_skill_type = pickle.load(f)
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        return label_encoder_cat, label_encoder_type, model_skill_type, model_category, tokenizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Main app
def main():
    st.markdown('<p class="main-header">🎯 AI-Powered Resume Screening System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload PDF resumes to analyze job category and skill type</p>', unsafe_allow_html=True)
    
    # Load models and encoders
    label_encoder_cat, label_encoder_type, model_skill_type, model_category, tokenizer = load_models()
    
    if label_encoder_cat is None:
        st.warning("⚠️ Please ensure all model files (.pkl) are available in the directory.")
        st.info("Make sure to run the training script (resume_screening.py) first to generate the model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This AI system analyzes resumes to predict:
        - **Job Category**: The professional domain
        - **Skill Type**: Technical (hard) vs interpersonal (soft) skills
        
        The system uses LSTM neural networks trained on resume data.
        """)
        
        st.header("📊 Sample Categories")
        st.write("Examples of job categories the model can identify:")
        categories = label_encoder_cat.classes_
        for cat in categories[:10]:
            st.write(f"• {cat}")
        if len(categories) > 10:
            st.write(f"...and {len(categories) - 10} more")
        
        st.header("💡 Supported Formats")
        st.write("✅ PDF files")
        st.write("✅ Text input")
    
    # Main content
    tab1, tab2 = st.tabs(["📝 Single Resume Analysis", "📚 Batch Upload"])
    
    with tab1:
        st.subheader("Analyze a Single Resume")
        
        input_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"])
        
        resume_text = ""
        
        if input_method == "Upload PDF":
            uploaded_file = st.file_uploader("Upload resume PDF", type=['pdf'], key="single_pdf")
            
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF..."):
                    resume_text = extract_text_from_pdf(uploaded_file)
                
                if resume_text:
                    st.success("✅ PDF uploaded and text extracted successfully!")
                    
                    # Show preview of extracted text
                    with st.expander("📄 Preview Extracted Text"):
                        st.text_area("Extracted Resume Text", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
                else:
                    st.error("❌ Failed to extract text from PDF. Please try another file.")
        else:
            resume_text = st.text_area("Paste resume text here:", height=300, 
                                       placeholder="Paste the resume content here...")
        
        if st.button("🔍 Analyze Resume", type="primary", disabled=not resume_text):
            if resume_text and resume_text.strip():
                with st.spinner("Analyzing resume..."):
                    # Call the prediction function
                    result = predict_resume_lstm(
                        resume_text, 
                        model_category, 
                        model_skill_type, 
                        tokenizer, 
                        label_encoder_cat, 
                        label_encoder_type, 
                        max_len=300
                    )
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Primary Category")
                    st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"### {result['category']}")
                    
                    # Convert percentage string to float for progress bar
                    confidence_value = float(result['category_confidence'].strip('%')) / 100
                    st.progress(confidence_value)
                    st.markdown(f"**Confidence:** {result['category_confidence']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show top 3 predictions
                    with st.expander("View Top 3 Category Predictions"):
                        for i, (cat, conf) in enumerate(result['top_3_categories'], 1):
                            st.write(f"**{i}.** {cat} - {conf:.2%}")
                
                with col2:
                    st.markdown("### 💼 Skill Type")
                    st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"### {result['skill_type'].title()} Skills")
                    
                    # Convert percentage string to float for progress bar
                    skill_confidence_value = float(result['skill_type_confidence'].strip('%')) / 100
                    st.progress(skill_confidence_value)
                    st.markdown(f"**Confidence:** {result['skill_type_confidence']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Extract and display skills
                st.markdown("### 🔑 Detected Skills")
                skills_found = extract_skills(resume_text.lower())
                
                if skills_found:
                    skills_list = skills_found.split()
                    skills_html = " ".join([f"<span style='background-color: #696969; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block;'>{skill}</span>" 
                                          for skill in skills_list])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("No specific skills detected from the keyword list.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Cleaned text length:** {len(clean_text(resume_text))} characters")
                    st.write(f"**Original text length:** {len(resume_text)} characters")
                    st.write(f"**Skills found:** {skills_found if skills_found else 'None'}")
                    st.write(f"**Number of categories in model:** {len(label_encoder_cat.classes_)}")
                    st.write(f"**All top 3 predictions:**")
                    for i, (cat, conf) in enumerate(result['top_3_categories'], 1):
                        st.write(f"  {i}. {cat}: {conf:.4f}")
                
            else:
                st.warning("⚠️ Please provide resume text to analyze.")

       # Replace your "Find Job" button section with this improved version:

        if resume_text and st.button("Find Job..?🔎", type="primary"):
            skills_found = extract_skills(resume_text.lower())
            
            # Debug: Show what skills were extracted
            st.write("**Debug - Extracted Skills:**", skills_found if skills_found else "None")
            
            # If no skills found from keywords, use the predicted category instead
            if not skills_found or skills_found.strip() == "":
                st.warning("⚠️ No specific skills detected from resume. Using job category instead.")
                # Use the category from previous analysis
                search_query = result.get('category', 'software developer') if 'result' in locals() else "software developer"
            else:
                search_query = skills_found
            
            st.write(f"**Searching for:** {search_query}")
            
            url = "https://indeed12.p.rapidapi.com/jobs/search"

            headers = {
                'x-rapidapi-key': "Your API",
                'x-rapidapi-host': "Your API"
            }

            # Include skills in the query parameter
            querystring = {
                "query": search_query,
                "location": "India",
                "page_id": "1",
                "locality": "in",
                "fromage": "30",  # Increased to 30 days for more results
                "radius": "100"   # Increased radius to 100km
            }
            
            with st.spinner("Searching for jobs..."):
                try:
                    response = requests.get(url, headers=headers, params=querystring, timeout=10)
                    
                    st.write(f"**API Response Status:** {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Debug: Show raw response structure
                        with st.expander("🔍 Debug: Raw API Response"):
                            st.json(data)
                        
                        jobs_found = data.get('hits', [])
                        
                        if jobs_found:
                            st.success(f"✅ Found {len(jobs_found)} jobs!")
                            
                            # Display jobs nicely
                            for idx, job in enumerate(jobs_found, 1):
                                with st.container():
                                    st.write("---")
                                    st.subheader(f"{idx}. {job.get('title', 'No title')}")
                                    
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.write(f"**🏢 Company:** {job.get('company_name', 'N/A')}")
                                        st.write(f"**📍 Location:** {job.get('location', 'N/A')}")
                                        
                                        # Job description preview
                                        description = job.get('description', 'N/A')
                                        if description != 'N/A' and len(description) > 150:
                                            st.write(f"**📝 Description:** {description[:150]}...")
                                        else:
                                            st.write(f"**📝 Description:** {description}")
                                    
                                    with col2:
                                        # Handle salary object
                                        salary_info = job.get('salary', {})
                                        if salary_info and isinstance(salary_info, dict):
                                            salary_min = salary_info.get('min', 'N/A')
                                            salary_max = salary_info.get('max', 'N/A')
                                            salary_type = salary_info.get('type', 'N/A')
                                            
                                            if salary_max == -1 or salary_max == 'N/A':
                                                st.write(f"**💰 Salary:** ₹{salary_min}+ {salary_type}")
                                            else:
                                                st.write(f"**💰 Salary:** ₹{salary_min} - ₹{salary_max} {salary_type}")
                                        else:
                                            st.write(f"**💰 Salary:** Not specified")
                                        
                                        # Job link if available
                                        job_url = job.get('link', job.get('url', ''))
                                        if job_url:
                                            st.markdown(f"[🔗 View Job]({job_url})")
                        else:
                            st.warning("⚠️ No jobs found for this search.")
                            st.info("""
                            **Suggestions:**
                            - Try analyzing the resume first to extract skills
                            - The search might be too specific - broader terms may work better
                            - Try different locations or increase the search radius
                            - Check if your API has rate limits or subscription issues
                            """)
                                
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.write("**Response:**", response.text[:500])
                        
                        if response.status_code == 429:
                            st.warning("⚠️ Rate limit exceeded. Please wait a moment and try again.")
                        elif response.status_code == 403:
                            st.warning("⚠️ API key may be invalid or subscription expired.")
                            
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.write("**Full error:**", e)
            


            
    
    with tab2:
        st.subheader("Batch Resume Analysis")
        st.info("Upload multiple PDF files to analyze them all at once.")
        
        # Multiple PDF upload
        uploaded_files = st.file_uploader(
            "Upload PDF resumes (you can select multiple files)", 
            type=['pdf'], 
            accept_multiple_files=True,
            key="batch_pdf"
        )
        
        if uploaded_files:
            st.write(f"📎 {len(uploaded_files)} file(s) uploaded")
            
            # Show list of uploaded files
            with st.expander("View uploaded files"):
                for idx, file in enumerate(uploaded_files, 1):
                    st.write(f"{idx}. {file.name}")
            
            if st.button("🚀 Analyze All Resumes"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, pdf_file in enumerate(uploaded_files):
                    status_text.text(f"Processing: {pdf_file.name}")
                    
                    # Extract text from PDF
                    resume_text = extract_text_from_pdf(pdf_file)
                    
                    if resume_text:
                        # Call the prediction function
                        result = predict_resume_lstm(
                            resume_text,
                            model_category,
                            model_skill_type,
                            tokenizer,
                            label_encoder_cat,
                            label_encoder_type,
                            max_len=300
                        )
                        
                        # Extract skills
                        skills = extract_skills(resume_text.lower())
                        
                        results.append({
                            'File_Name': pdf_file.name,
                            'Predicted_Category': result['category'],
                            'Category_Confidence': result['category_confidence'],
                            'Skill_Type': result['skill_type'],
                            'Skill_Confidence': result['skill_type_confidence'],
                            'Skills_Found': skills if skills else 'None'
                        })
                    else:
                        results.append({
                            'File_Name': pdf_file.name,
                            'Predicted_Category': 'Error',
                            'Category_Confidence': 'N/A',
                            'Skill_Type': 'Error',
                            'Skill_Confidence': 'N/A',
                            'Skills_Found': 'Failed to extract text'
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                st.success("✅ Batch analysis complete!")
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Resumes Analyzed", len(results_df))
                
                with col2:
                    # Count successful analyses
                    successful = len(results_df[results_df['Predicted_Category'] != 'Error'])
                    st.metric("Successfully Processed", successful)
                
                with col3:
                    # Count failed analyses
                    failed = len(results_df[results_df['Predicted_Category'] == 'Error'])
                    st.metric("Failed", failed)
                
                # Category distribution
                if len(results_df[results_df['Predicted_Category'] != 'Error']) > 0:
                    st.subheader("📈 Category Distribution")
                    category_counts = results_df[results_df['Predicted_Category'] != 'Error']['Predicted_Category'].value_counts()
                    fig = px.pie(values=category_counts.values, names=category_counts.index, 
                               title="Distribution of Resume Categories")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="resume_analysis_results.csv",
                    mime="text/csv"
                )
    



if __name__ == "__main__":

    main()

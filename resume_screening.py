import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

stop_words = set(stopwords.words('english'))
hard_skill_keywords = [
    'python', 'java', 'c++', 'javascript', 'sql', 'r', 'go', 'rust',
    'html', 'css', 'react', 'node', 'django', 'flask', 'spring',
    'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
    'data science', 'data analysis', 'statistics', 'big data', 'hadoop', 'spark',
    'mysql', 'postgresql', 'mongodb', 'redis', 'database',
    'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'git', 'linux', 'unix', 'agile', 'scrum',
    'tableau', 'power bi', 'excel'
]

soft_skill_keywords = [
    'communication', 'teamwork', 'leadership', 'management',
    'problem solving', 'critical thinking', 'analytical',
    'time management', 'adaptability', 'creativity',
    'decision making', 'emotional intelligence', 'presentation',
    'collaboration', 'mentoring', 'initiative'
]
def extract_skills(text):
    """Enhanced skill extraction"""
    text_lower = str(text).lower()
    hard_skills = [skill for skill in hard_skill_keywords if skill in text_lower]
    soft_skills = [skill for skill in soft_skill_keywords if skill in text_lower]
    return ' '.join(hard_skills + soft_skills)

def label_skill_type(skills):
    """Direct skill type labeling instead of clustering"""
    skills_lower = str(skills).lower()
    hard_count = sum(1 for skill in hard_skill_keywords if skill in skills_lower)
    soft_count = sum(1 for skill in soft_skill_keywords if skill in skills_lower)
    
    if hard_count > soft_count:
        return 'hard'
    elif soft_count > hard_count:
        return 'soft'
    else:
        return 'mixed'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() 
                     if word not in stop_words or len(word) > 2])
    return text.strip()

def build_lstm_model(vocab_size, max_len, num_classes):
    """Build enhanced LSTM model"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Load data
resume = pd.read_csv('UpdatedResumeDataSet.csv')

# Clean and preprocess
resume['Resume'] = resume['Resume'].astype(str).str.lower()
resume['clean_resume'] = resume['Resume'].apply(clean_text)

# Filter out empty resumes after cleaning
resume = resume[resume['clean_resume'].str.len() > 10].reset_index(drop=True)

# Extract skills
resume['skills'] = resume['clean_resume'].apply(extract_skills)

# Label skill types directly (more reliable than clustering)
resume['skill_type_unsupervised'] = resume['skills'].apply(label_skill_type)
# Encode labels
label_encoder_cat = LabelEncoder()
resume['category_encoded'] = label_encoder_cat.fit_transform(resume['Category'])

label_encoder_type = LabelEncoder()
resume['skill_type_encoded'] = label_encoder_type.fit_transform(resume['skill_type_unsupervised'])

# Tokenization with better parameters
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>", lower=True)
tokenizer.fit_on_texts(resume['clean_resume'])

x_resume = tokenizer.texts_to_sequences(resume['clean_resume'])
X_pad = pad_sequences(x_resume, maxlen=max_len, padding='post', truncating='post')

# Train-test split
x_train_cat, x_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_pad, resume['category_encoded'], test_size=0.2, random_state=42, 
    stratify=resume['category_encoded']
)

x_train_type, x_test_type, y_train_type, y_test_type = train_test_split(
    X_pad, resume['skill_type_encoded'], test_size=0.2, random_state=42, 
    stratify=resume['skill_type_encoded']
)

# Compute class weights
class_weights_cat = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_cat),
    y=y_train_cat
)
class_weight_dict_cat = dict(enumerate(class_weights_cat))

class_weights_type = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_type),
    y=y_train_type
)
class_weight_dict_type = dict(enumerate(class_weights_type))

# Callbacks
early_stop_cat = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cat = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-7,
    verbose=1
)

early_stop_type = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True,
    verbose=1
)
reduce_lr_type = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-7,
    verbose=1
)

# Train Category Model
print("\n" + "="*70)
print("TRAINING CATEGORY MODEL")
print("="*70)

model_category = build_lstm_model(max_words, max_len, len(label_encoder_cat.classes_))

history_cat = model_category.fit(
    x_train_cat, y_train_cat,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict_cat,
    callbacks=[early_stop_cat, reduce_lr_cat],
    verbose=1
)

y_pred_cat = model_category.predict(x_test_cat, verbose=0)
y_pred_cat_classes = np.argmax(y_pred_cat, axis=1)

cat_accuracy = accuracy_score(y_test_cat, y_pred_cat_classes)
print(f"\nCategory Model Test Accuracy: {cat_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cat, y_pred_cat_classes, 
                          target_names=label_encoder_cat.classes_,
                          zero_division=0))

# Train Skill Type Model
print("\n" + "="*70)
print("TRAINING SKILL TYPE MODEL")
print("="*70)

model_skill_type = build_lstm_model(max_words, max_len, len(label_encoder_type.classes_))

history_type = model_skill_type.fit(
    x_train_type, y_train_type,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict_type,
    callbacks=[early_stop_type, reduce_lr_type],
    verbose=1
)

y_pred_type = model_skill_type.predict(x_test_type, verbose=0)
y_pred_type_classes = np.argmax(y_pred_type, axis=1)

type_accuracy = accuracy_score(y_test_type, y_pred_type_classes)
print(f"\nSkill Type Model Test Accuracy: {type_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_type, y_pred_type_classes, 
                          target_names=label_encoder_type.classes_,
                          zero_division=0))

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

with open('model_category.pkl', 'wb') as f:
    pickle.dump(model_category, f)
print("Saved: model_category.pkl")

with open('model_skill_type.pkl', 'wb') as f:
    pickle.dump(model_skill_type, f)
print("Saved: model_skill_type.pkl")

with open('label_encoder_cat.pkl', 'wb') as f:
    pickle.dump(label_encoder_cat, f)
print("Saved: label_encoder_cat.pkl")

with open('label_encoder_type.pkl', 'wb') as f:
    pickle.dump(label_encoder_type, f)
print("Saved: label_encoder_type.pkl")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Saved: tokenizer.pkl")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Category Model Accuracy: {cat_accuracy:.2%}")
print(f"Skill Type Model Accuracy: {type_accuracy:.2%}")
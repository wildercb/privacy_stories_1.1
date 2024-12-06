import os
import re

def clean_full_text(text):
    """Removes annotations (A:, DT:, P:), section tags {#s ... \}, and <PI> tags from the text."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', text)
    cleaned_text = re.sub(r'{#s|\}', '', cleaned_text)
    cleaned_text = re.sub(r'<PI>', '', cleaned_text)
    return cleaned_text.strip()

def process_file(file_path):
    """Processes a single file to extract cleaned full text and sections with metadata."""
    with open(file_path, 'r') as file:
        text = file.read()

    # Full cleaned text without annotations or section tags
    cleaned_full_text = clean_full_text(text)
    
    # Prepare dictionary structure
    result = {
        "file_name": os.path.basename(file_path),
        "full_cleaned_text": cleaned_full_text,
        "sections": []
    }

    # Extract sections defined by {#s ... \}
    section_pattern = re.compile(r'{#s(.*?)\\}', re.DOTALL)
    sections = section_pattern.findall(text)

    for section in sections:
        # Clean section text by removing annotations
        cleaned_section = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', section).strip()

        # Extract metadata (actions, data types, purposes)
        actions = re.findall(r'\(A:\s*(.*?)\)', section)
        data_types = re.findall(r'\(DT:\s*(.*?)\)', section)
        purposes = re.findall(r'\(P:\s*(.*?)\)', section)
        
        # Add section data to dictionary
        result["sections"].append({
            "section_text_with_tags": cleaned_section,
            "cleaned_section_text": cleaned_section,
            "metadata": {
                "actions": actions if actions else None,
                "data_types": data_types if data_types else None,
                "purposes": purposes if purposes else None
            }
        })
        
    return result

def process_directory(directory_path):
    """Processes all text files in a directory and returns a list of dictionaries for each file."""
    results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_data = process_file(file_path)
            results.append(file_data)
    return results

def process_input(path):
    """Determines if the path is a file or directory and processes accordingly."""
    if os.path.isdir(path):
        # Process all files in the directory
        return process_directory(path)
    elif os.path.isfile(path):
        # Process a single file
        return [process_file(path)]
    else:
        raise ValueError("Invalid path. Please provide a valid file or directory path.")

''' 
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_full_text(text):
    """Removes annotations (A:, DT:, P:), section tags {#s ... \}, and <PI> tags from the text."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', text)
    cleaned_text = re.sub(r'{#s|\}', '', cleaned_text)
    cleaned_text = re.sub(r'<PI>', '', cleaned_text)
    return cleaned_text.strip()

def process_file(file_path):
    """Processes a single file to extract cleaned full text and sections with metadata."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Full cleaned text without annotations or section tags
    cleaned_full_text = clean_full_text(text)
    
    # Prepare dictionary structure
    result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "full_cleaned_text": cleaned_full_text,
        "sections": []
    }

    # Extract sections defined by {#s ... \}
    section_pattern = re.compile(r'{#s(.*?)\\}', re.DOTALL)
    sections = section_pattern.findall(text)

    for section in sections:
        # Clean section text by removing annotations
        cleaned_section = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', section).strip()

        # Extract metadata (actions, data types, purposes)
        actions = re.findall(r'\(A:\s*(.*?)\)', section)
        data_types = re.findall(r'\(DT:\s*(.*?)\)', section)
        purposes = re.findall(r'\(P:\s*(.*?)\)', section)
        
        # Add section data to dictionary
        result["sections"].append({
            "section_text_with_tags": cleaned_section,
            "cleaned_section_text": cleaned_section,
            "metadata": {
                "actions": actions if actions else None,
                "data_types": data_types if data_types else None,
                "purposes": purposes if purposes else None
            }
        })
        
    return result

def find_most_similar_files(target_file, directory_path, top_n=3):
    """
    Find the most similar files to the target file in the given directory.
    
    Args:
    - target_file (str): Path to the target file
    - directory_path (str): Path to the directory to search
    - top_n (int): Number of most similar files to return
    
    Returns:
    - List of dictionaries with most similar files, sorted by similarity
    """
    # Process target file
    try:
        target_data = process_file(target_file)
        target_text = target_data['full_cleaned_text']
    except Exception as e:
        raise ValueError(f"Error processing target file: {e}")

    # Find all files in the directory
    file_paths = []
    file_texts = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and file_path != target_file:
            try:
                file_data = process_file(file_path)
                file_paths.append(file_path)
                file_texts.append(file_data['full_cleaned_text'])
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")

    # If no files found
    if not file_texts:
        return []

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine target text with all other texts for vectorization
    all_texts = [target_text] + file_texts
    
    # Vectorize texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Get top N most similar files
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    
    # Prepare results
    similar_files = []
    for idx in top_indices:
        similar_files.append({
            'file_path': file_paths[idx],
            'file_name': os.path.basename(file_paths[idx]),
            'similarity_score': cosine_similarities[idx],
            'file_data': process_file(file_paths[idx])
        })
    
    return similar_files

def process_input(path):
    """Determines if the path is a file or directory and processes accordingly."""
    if os.path.isdir(path):
        # Process all files in the directory
        return process_directory(path)
    elif os.path.isfile(path):
        # Process a single file
        return [process_file(path)]
    else:
        raise ValueError("Invalid path. Please provide a valid file or directory path.")

def process_directory(directory_path):
    """Processes all text files in a directory and returns a list of dictionaries for each file."""
    results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                file_data = process_file(file_path)
                results.append(file_data)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return results
    '''
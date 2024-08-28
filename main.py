import pdfplumber
import ollama
from multiprocessing import Pool
import os
import json
import numpy as np
from numpy.linalg import norm

def process_page(args):
    page_num, filename = args
    with pdfplumber.open(filename) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text()
        if not text:
            return []
        
        paragraphs = []
        buffer = []
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if line:
                buffer.append(line)
            elif buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs

def parse_pdf_multiprocessing_lazy(filename):
    with pdfplumber.open(filename) as pdf:
        num_pages = len(pdf.pages)
    
    with Pool() as pool:
        for paragraphs in pool.imap(process_page, [(i, filename) for i in range(num_pages)]):
            for paragraph in paragraphs:
                yield paragraph

def save_embeddings(filename, embeddings):
    os.makedirs("embeddings", exist_ok=True)
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    filepath = f"embeddings/{filename}.json"
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunk_generator):
    embeddings = load_embeddings(filename)
    if embeddings is not None:
        return embeddings

    embeddings = []
    for chunk in chunk_generator:
        embedding = ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        embeddings.append(embedding)
    
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    filename = "Understand Systems Thinking.pdf"
    modelname = "mistral-nemo"
    
    paragraph_generator = parse_pdf_multiprocessing_lazy(filename)
    embeddings = get_embeddings(filename, modelname, paragraph_generator)
    
    prompt = input("What do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model=modelname, prompt=prompt)["embedding"]

    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    
    # To avoid re-reading the entire PDF, convert the generator to a list only if necessary
    paragraph_list = list(parse_pdf_multiprocessing_lazy(filename))

    response = ollama.chat(
        model=modelname,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraph_list[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )

    print("\n\n")
    print(response["message"]["content"])

    # Free up memory by deleting large objects after use
    del embeddings, paragraph_list, most_similar_chunks

if __name__ == "__main__":
    main()

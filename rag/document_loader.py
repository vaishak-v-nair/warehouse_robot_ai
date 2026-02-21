import os

def load_documents(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".md"):
            with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
                documents.append({
                    "filename": filename,
                    "content": f.read()
                })
    return documents
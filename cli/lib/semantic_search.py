from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


#### CLI Commands ####
def verify_model() -> None:
    try:
        search = SemanticSearch()
        print(f"Model loaded: {search.model}")
        print(f"Max sequence length: {search.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
    

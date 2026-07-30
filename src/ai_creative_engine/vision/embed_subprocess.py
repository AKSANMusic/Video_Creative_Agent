import sys
import json
import os

# Save original stdout for our final JSON output
original_stdout = sys.stdout
# Redirect all other stdout to stderr to hide noisy library logs (like PyTorch progress bars)
sys.stdout = sys.stderr

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(1)
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    
    # Read JSON array of strings from stdin
    input_data = sys.stdin.read().strip()
    texts = json.loads(input_data)
    
    if not isinstance(texts, list):
        raise ValueError("Input must be a JSON list of strings")
        
    if texts:
        emb_matrix = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Convert 2D numpy array to list of lists
        emb_list = emb_matrix.tolist()
    else:
        emb_list = []
        
    # Write only the final JSON to the original stdout
    original_stdout.write(json.dumps(emb_list) + "\n")
    original_stdout.flush()
except Exception as e:
    original_stdout.write(json.dumps({"error": str(e)}) + "\n")
    original_stdout.flush()
    sys.exit(1)

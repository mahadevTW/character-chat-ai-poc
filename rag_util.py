import os
import docx
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-east-1-aws"
INDEX_NAME = os.getenv("PINECONE_INDEX", "cinejoy-scenes")
EMBEDDING_DIM = 1536  # For text-embedding-ada-002

# Validate required keys
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("Missing one or more required environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Cache
_script_cache = {}

def read_script(movie: str) -> str:
    if movie in _script_cache:
        return _script_cache[movie]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "data", movie, "script.docx")
    sketch_path = os.path.join(base_dir, "data", movie, "character_sketch.docx")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    doc = docx.Document(script_path)
    script_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    character_sketch = ""
    if os.path.exists(sketch_path):
        doc_sketch = docx.Document(sketch_path)
        sketch_text = "\n".join(p.text for p in doc_sketch.paragraphs if p.text.strip())
        character_sketch = "\n\n--- CHARACTER SKETCH ---\n\n" + sketch_text

    combined = script_text + character_sketch
    _script_cache[movie] = combined
    return combined

def read_prompt_docx(movie: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "data", movie, "prompt.docx")
    if os.path.exists(prompt_path):
        try:
            doc = docx.Document(prompt_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"⚠️ Error loading prompt.docx: {e}")
    return ""

def split_into_scenes(script: str) -> list:
    scenes, current = [], []
    for line in script.splitlines():
        if line.strip().upper().startswith(("INT.", "EXT.")):
            if current:
                scenes.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        scenes.append("\n".join(current))
    return scenes

def truncate_scene(scene: str, max_tokens: int = 7500, model_name: str = "text-embedding-ada-002") -> str:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(scene)
    if len(tokens) > max_tokens:
        print("⚠️ Truncating long scene for embedding.")
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def get_scene_embeddings(scenes: list) -> list:
    return [
        client.embeddings.create(input=truncate_scene(scene), model="text-embedding-ada-002").data[0].embedding
        for scene in scenes
    ]

def store_scenes_in_pinecone(movie: str, scenes: list) -> None:
    embeddings = get_scene_embeddings(scenes)
    vectors = [
        {"id": f"{movie}_{i}", "values": emb, "metadata": {"movie": movie, "scene": scenes[i]}}
        for i, emb in enumerate(embeddings)
    ]
    index.upsert(vectors=vectors)

def get_relevant_scene_chunks_pinecone(movie: str, query: str) -> list:
    query_emb = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    results = index.query(
        vector=query_emb,
        top_k=3,
        include_metadata=True,
        filter={"movie": {"$eq": movie}}
    )
    return [match['metadata']['scene'] for match in results['matches'] if match.get('score', 0) > 0.01]

def generate_prompt_with_rag(movie: str, character: str, user_message: str) -> str:
    script = read_script(movie)
    scenes = split_into_scenes(script)
    store_scenes_in_pinecone(movie, scenes)

    base_prompt = read_prompt_docx(movie)
    if not base_prompt:
        base_prompt = (
            f"You are {character} from the movie {movie}. Respond based on the tone and character of the movie.\n"
            f"Stay in character, respond as {character} would in this situation. Don't say you are an AI."
        )

    relevant_chunks = get_relevant_scene_chunks_pinecone(movie, user_message)
    quotes = "\n\n".join(f"Scene:\n{chunk[:600]}..." for chunk in relevant_chunks)

    return (
        f"{base_prompt}\n\n"
        f"Here are some relevant scenes from the movie:\n{quotes}\n\n"
        f"Stay in character as {character}. Don't break character or mention you're an AI."
    )

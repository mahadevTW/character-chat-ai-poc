import os
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache to store loaded scripts
_script_cache = {}
# Cache to store split scenes
_scenes_cache = {}

def read_script(movie: str) -> str:
    """
    Loads and caches a movie script from a Word document (.docx) file.
    
    This function reads the script file from the data directory, extracts all text
    from paragraphs, and caches the result to avoid repeated file I/O operations.
    
    Args:
        movie (str): The name of the movie directory containing the script.docx file
        
    Returns:
        str: The complete script text extracted from all paragraphs
        
    Raises:
        FileNotFoundError: If the script.docx file doesn't exist in the movie's data directory
        
    Note:
        Uses in-memory caching (_script_cache) to store loaded scripts for subsequent calls.
        Each movie script is loaded only once per application lifecycle.
    """
    # Check if script is already cached
    if movie in _script_cache:
        return _script_cache[movie]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "data", movie, "script.docx")
    print("script at:", script_path)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")

    doc = docx.Document(script_path)
    script_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    
    # Cache the script
    _script_cache[movie] = script_text
    return script_text

def split_into_scenes(script: str, movie: str = None) -> list:
    """
    Splits a movie script into individual scenes based on scene headings.
    
    This function parses the script text and identifies scene boundaries by looking
    for lines that start with "INT." or "EXT." (interior/exterior scene indicators).
    Each scene is collected as a separate text block.
    
    Args:
        script (str): The complete script text to be split into scenes
        movie (str, optional): The movie name for caching purposes. If provided,
                              the split scenes will be cached to avoid reprocessing.
        
    Returns:
        list: A list of strings, where each string represents one complete scene
        
    Note:
        Uses in-memory caching (_scenes_cache) when movie parameter is provided.
        Scene splitting is done only once per movie per application lifecycle.
        
    Example:
        A script with scenes like:
        "INT. LIVING ROOM - DAY\nJohn enters...\n\nEXT. STREET - NIGHT\nMary walks..."
        Will be split into: ["INT. LIVING ROOM - DAY\nJohn enters...", "EXT. STREET - NIGHT\nMary walks..."]
    """
    # If movie is provided, check if scenes are already cached
    if movie and movie in _scenes_cache:
        return _scenes_cache[movie]
    
    scenes = []
    current_scene = []
    for line in script.splitlines():
        if line.strip().upper().startswith(("INT.", "EXT.")):
            if current_scene:
                scenes.append("\n".join(current_scene))
                current_scene = []
        current_scene.append(line)
    if current_scene:
        scenes.append("\n".join(current_scene))
    
    # Cache the scenes if movie is provided
    if movie:
        _scenes_cache[movie] = scenes
    
    return scenes

def get_relevant_scene_chunks(scenes, query):
    """
    Finds the most semantically relevant scenes for a given query using TF-IDF and cosine similarity.
    
    This function uses natural language processing techniques to identify which scenes
    from the movie script are most relevant to the user's query. It employs:
    1. TF-IDF vectorization to convert text into numerical representations
    2. Cosine similarity to measure semantic similarity between the query and each scene
    3. Ranking to return the top 3 most relevant scenes above a similarity threshold
    
    Args:
        scenes (list): List of scene strings from the movie script
        query (str): The user's message/query to find relevant scenes for
        
    Returns:
        list: A list of the most relevant scene chunks (up to 3 scenes) that have
              similarity scores above 0.01 with the query
              
    Technical Details:
        - Uses sklearn's TfidfVectorizer to create TF-IDF representations
        - Applies cosine_similarity to compute similarity between query and all scenes
        - Returns scenes with similarity > 0.01, sorted by relevance (top 3)
        - If no scenes meet the threshold, returns an empty list
    """
    vectorizer = TfidfVectorizer().fit(scenes + [query])
    scene_vectors = vectorizer.transform(scenes)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, scene_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return [scenes[i] for i in top_indices if similarities[i] > 0.01]

def generate_prompt_with_rag(movie: str, character: str, user_message: str) -> str:
    """
    Generates a context-aware system prompt for character roleplay using RAG (Retrieval-Augmented Generation).
    
    This function implements a RAG pipeline that:
    1. Loads the movie script (with caching)
    2. Splits the script into scenes (with caching)
    3. Retrieves the most relevant scenes for the user's message
    4. Constructs a system prompt that includes character context and relevant movie scenes
    
    The resulting prompt instructs the AI to respond as the specified character
    while being grounded in the actual movie content and context.
    
    Args:
        movie (str): The name of the movie to get context from
        character (str): The character name to roleplay as
        user_message (str): The user's message used to find relevant scenes
        
    Returns:
        str: A complete system prompt that includes:
             - Character roleplay instructions
             - Relevant movie scenes (truncated to 600 chars each)
             - Multiple reminders to stay in character
             
    Technical Process:
        1. Calls read_script() with caching to get movie script
        2. Calls split_into_scenes() with caching to get scene list
        3. Calls get_relevant_scene_chunks() to find relevant scenes for the query
        4. Constructs final prompt with character instructions and scene context
        
    Note:
        This function leverages multiple caching layers to optimize performance:
        - Script loading is cached per movie
        - Scene splitting is cached per movie
        - The generated prompt is cached per movie-character combination in main.py
    """
    script = read_script(movie)
    scenes = split_into_scenes(script, movie)
    relevant_chunks = get_relevant_scene_chunks(scenes, user_message)

    quotes = "\n\n".join(f"Scene:\n{chunk[:600]}..." for chunk in relevant_chunks)
    return (
        f"You are {character} from the movie {movie}. Respond based on the scenes and tone of the movie.\n"
        f"Here are some relevant scenes from the movie:\n{quotes}\n"
        f"Stay in character, respond as {character} would in this situation."
        f"dont response as AI character, respond as {character} would in this situation."
        f"dont explicitly say you are AI assistant, respond as {character} would in this situation."
        f"As AI even if you dont have personal opinion, respond as {character} would in this situation."
        f"even if you are character in movie, dont explicitly say you are character rather respond as {character} would in this situation."
    )
import os
import pinecone
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class PineconeDatabase:
    def __init__(self, api_key: str = None, environment: str = None):
        """
        Initialize Pinecone database connection.
        
        Args:
            api_key (str): Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment (str): Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable.")
        if not self.environment:
            raise ValueError("Pinecone environment is required. Set PINECONE_ENVIRONMENT environment variable.")
        
        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.vectorizer = TfidfVectorizer()
        
    def _get_index_name(self, movie: str) -> str:
        """Generate an index name for a movie."""
        # Pinecone index names must be lowercase and contain only letters, numbers, and hyphens
        return f"movie-{movie.lower().replace(' ', '-').replace('_', '-')}"
    
    def _get_or_create_index(self, movie: str, dimension: int = 1000):
        """
        Get existing index or create a new one for the movie.
        
        Args:
            movie (str): Movie name
            dimension (int): Vector dimension (default 1000 for TF-IDF)
        """
        index_name = self._get_index_name(movie)
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            # Create new index
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            print(f"Created Pinecone index: {index_name}")
        
        return pinecone.Index(index_name)
    
    def store_scenes(self, movie: str, scenes: List[str]) -> bool:
        """
        Store movie scenes as vectors in Pinecone.
        
        Args:
            movie (str): Movie name
            scenes (List[str]): List of scene texts
            
        Returns:
            bool: True if successfully stored, False otherwise
        """
        try:
            index = self._get_or_create_index(movie)
            
            # Generate embeddings using TF-IDF
            embeddings = self.vectorizer.fit_transform(scenes).toarray()
            
            # Prepare data for Pinecone
            vectors = []
            for i, (scene, embedding) in enumerate(zip(scenes, embeddings)):
                vectors.append({
                    "id": f"scene_{i}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "movie": movie,
                        "scene_index": i,
                        "text": scene[:1000]  # Store first 1000 chars as metadata
                    }
                })
            
            # Store in Pinecone (upsert in batches of 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
            
            print(f"Stored {len(scenes)} scenes for movie '{movie}' in Pinecone")
            return True
            
        except Exception as e:
            print(f"Error storing scenes for movie '{movie}': {e}")
            return False
    
    def search_similar_scenes(self, movie: str, query: str, top_k: int = 3, similarity_threshold: float = 0.01) -> List[str]:
        """
        Search for scenes similar to the query using vector similarity.
        
        Args:
            movie (str): Movie name to search in
            query (str): Search query
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity score threshold
            
        Returns:
            List[str]: List of relevant scene texts
        """
        try:
            index = self._get_or_create_index(movie)
            
            # Generate query embedding
            query_embedding = self.vectorizer.transform([query]).toarray()
            
            # Search in Pinecone
            results = index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter by similarity threshold and return scene texts
            relevant_scenes = []
            for match in results.matches:
                if match.score > similarity_threshold:
                    # Get full scene text from metadata or fetch from index
                    scene_text = match.metadata.get("text", "")
                    if len(scene_text) < 100:  # If truncated, we might need to store full text elsewhere
                        scene_text = f"Scene {match.metadata.get('scene_index', 'unknown')}: {scene_text}"
                    relevant_scenes.append(scene_text)
            
            return relevant_scenes
            
        except Exception as e:
            print(f"Error searching scenes for movie '{movie}': {e}")
            return []
    
    def movie_exists(self, movie: str) -> bool:
        """
        Check if a movie's scenes are already stored in Pinecone.
        
        Args:
            movie (str): Movie name to check
            
        Returns:
            bool: True if movie exists in database
        """
        try:
            index_name = self._get_index_name(movie)
            return index_name in pinecone.list_indexes()
        except:
            return False
    
    def get_movie_info(self, movie: str) -> Optional[Dict]:
        """
        Get information about a stored movie.
        
        Args:
            movie (str): Movie name
            
        Returns:
            Optional[Dict]: Movie information or None if not found
        """
        try:
            index_name = self._get_index_name(movie)
            if index_name in pinecone.list_indexes():
                index = pinecone.Index(index_name)
                stats = index.describe_index_stats()
                return {
                    "movie": movie,
                    "index_name": index_name,
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension
                }
        except Exception as e:
            print(f"Error getting movie info for '{movie}': {e}")
        return None
    
    def delete_movie(self, movie: str) -> bool:
        """
        Delete all scenes for a movie from Pinecone.
        
        Args:
            movie (str): Movie name to delete
            
        Returns:
            bool: True if successfully deleted
        """
        try:
            index_name = self._get_index_name(movie)
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
                print(f"Deleted movie '{movie}' from Pinecone")
                return True
        except Exception as e:
            print(f"Error deleting movie '{movie}': {e}")
        return False
    
    def list_movies(self) -> List[str]:
        """
        List all movies stored in Pinecone.
        
        Returns:
            List[str]: List of movie names
        """
        try:
            indexes = pinecone.list_indexes()
            movies = []
            for index_name in indexes:
                if index_name.startswith("movie-"):
                    movie_name = index_name.replace("movie-", "").replace("-", " ")
                    movies.append(movie_name)
            return movies
        except Exception as e:
            print(f"Error listing movies: {e}")
            return []

def generate_prompt_with_rag_pinecone(movie: str, character: str, user_message: str) -> str:
    """
    Generates a context-aware system prompt for character roleplay using Pinecone RAG.
    
    This function implements a RAG pipeline using Pinecone that:
    1. Checks if movie scenes are stored in Pinecone
    2. Retrieves the most relevant scenes for the user's message
    3. Constructs a system prompt that includes character context and relevant movie scenes
    
    Args:
        movie (str): The name of the movie to get context from
        character (str): The character name to roleplay as
        user_message (str): The user's message used to find relevant scenes
        
    Returns:
        str: A complete system prompt that includes:
             - Character roleplay instructions
             - Relevant movie scenes (truncated to 600 chars each)
             - Multiple reminders to stay in character
    """
    try:
        # Initialize Pinecone database
        pinecone_db = PineconeDatabase()
        
        # Check if movie exists in Pinecone
        if not pinecone_db.movie_exists(movie):
            print(f"Movie '{movie}' not found in Pinecone. You need to store scenes first.")
            # Fallback to basic prompt without scenes
            return (
                f"You are {character} from the movie {movie}. Respond based on the tone and character of the movie.\n"
                f"Stay in character, respond as {character} would in this situation."
                f"dont response as AI character, respond as {character} would in this situation."
                f"dont explicitly say you are AI assistant, respond as {character} would in this situation."
                f"As AI even if you dont have personal opinion, respond as {character} would in this situation."
                f"even if you are character in movie, dont explicitly say you are character rather respond as {character} would in this situation."
            )
        
        # Search for relevant scenes
        relevant_chunks = pinecone_db.search_similar_scenes(movie, user_message)
        
        if not relevant_chunks:
            print(f"No relevant scenes found for query: {user_message}")
            # Fallback to basic prompt
            return (
                f"You are {character} from the movie {movie}. Respond based on the tone and character of the movie.\n"
                f"Stay in character, respond as {character} would in this situation."
                f"dont response as AI character, respond as {character} would in this situation."
                f"dont explicitly say you are AI assistant, respond as {character} would in this situation."
                f"As AI even if you dont have personal opinion, respond as {character} would in this situation."
                f"even if you are character in movie, dont explicitly say you are character rather respond as {character} would in this situation."
            )
        
        # Construct prompt with relevant scenes
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
        
    except Exception as e:
        print(f"Error in generate_prompt_with_rag_pinecone: {e}")
        # Fallback to basic prompt
        return (
            f"You are {character} from the movie {movie}. Respond based on the tone and character of the movie.\n"
            f"Stay in character, respond as {character} would in this situation."
            f"dont response as AI character, respond as {character} would in this situation."
            f"dont explicitly say you are AI assistant, respond as {character} would in this situation."
            f"As AI even if you dont have personal opinion, respond as {character} would in this situation."
            f"even if you are character in movie, dont explicitly say you are character rather respond as {character} would in this situation."
        )

# Example usage and utility functions
def store_movie_scenes_pinecone(movie: str, scenes: List[str]) -> bool:
    """
    Utility function to store movie scenes in Pinecone.
    
    Args:
        movie (str): Movie name
        scenes (List[str]): List of scene texts
        
    Returns:
        bool: True if successfully stored
    """
    try:
        pinecone_db = PineconeDatabase()
        return pinecone_db.store_scenes(movie, scenes)
    except Exception as e:
        print(f"Error storing movie scenes: {e}")
        return False

# Global Pinecone database instance
pinecone_db_instance = None

def get_pinecone_db():
    """Get or create a global Pinecone database instance."""
    global pinecone_db_instance
    if pinecone_db_instance is None:
        pinecone_db_instance = PineconeDatabase()
    return pinecone_db_instance 
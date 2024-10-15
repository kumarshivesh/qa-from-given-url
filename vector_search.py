import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Get the Pinecone API key from environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone and SentenceTransformer
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ytdemo")
model = SentenceTransformer('all-MiniLM-L6-v2')

def addData(corpusData, url):
    id = index.describe_index_stats()['total_vector_count']
    for i in range(len(corpusData)):
        chunk = corpusData[i]
        chunkInfo = (str(id+i), model.encode(chunk).tolist(), {'title': url, 'context': chunk})
        index.upsert(vectors=[chunkInfo])

def find_match(query, k):
    query_em = model.encode(query).tolist()
    result = index.query(vector=query_em, top_k=k, includeMetadata=True)
    return [result['matches'][i]['metadata']['title'] for i in range(k)], [result['matches'][i]['metadata']['context'] for i in range(k)]

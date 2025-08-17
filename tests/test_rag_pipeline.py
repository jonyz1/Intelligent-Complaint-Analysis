import unittest
import numpy as np
from src.rag_pipeline import Retriever, Generator
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.index = faiss.IndexFlatL2(384)
        self.metadata = [{"complaint_id": "123", "product": "Credit Card", "original_narrative": "Test complaint", "chunk_index": 0}]
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.retriever = Retriever(self.index, self.metadata, self.embedding_model)
        self.generator = Generator(model_name="gpt2-medium")

    def test_retrieve_chunks(self):
        chunks = self.retriever.retrieve_chunks("test query", k=1)
        self.assertEqual(len(chunks), 1)
        self.assertIn("complaint_id", chunks[0])

    def test_rag_pipeline(self):
        result = self.generator.generate("Test context", "Test question")
        self.assertIn("answer", result)
        self.assertTrue(isinstance(result["answer"], str))

if __name__ == "__main__":
    unittest.main()

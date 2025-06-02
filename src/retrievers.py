"""
Módulo de sistemas de recuperación semántica
Implementa búsqueda manual y con bases de datos vectoriales
"""

import json
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path


class BaseRetriever(ABC):
    """Clase base abstracta para sistemas de recuperación"""
    
    def __init__(self):
        self.fragments = None
        self.embeddings = None
        self.fragment_ids = None
        
    @abstractmethod
    def build_index(self, fragments: List[Dict], embeddings: np.ndarray) -> bool:
        """Construye el índice de búsqueda"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Realiza búsqueda semántica"""
        pass
    
    def load_data(self, corpus_path: str = "data/corpus.json", 
                  embeddings_path: str = "data/embeddings_complete.pkl") -> bool:
        """
        Carga datos necesarios para la búsqueda
        
        Args:
            corpus_path: Ruta al archivo de corpus
            embeddings_path: Ruta a los embeddings
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar fragmentos
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            self.fragments = {frag['id']: frag for frag in corpus_data['fragments']}
            
            # Cargar embeddings
            with open(embeddings_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            self.embeddings = embedding_data['embeddings']
            self.fragment_ids = embedding_data['fragment_ids']
            
            print(f"✅ Datos cargados: {len(self.fragments)} fragmentos, embeddings {self.embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def get_fragment_details(self, fragment_ids: List[str]) -> List[Dict]:
        """
        Obtiene detalles completos de los fragmentos
        
        Args:
            fragment_ids: Lista de IDs de fragmentos
            
        Returns:
            Lista de fragmentos con detalles
        """
        return [self.fragments.get(frag_id, {}) for frag_id in fragment_ids if frag_id in self.fragments]


class ManualRetriever(BaseRetriever):
    """Sistema de recuperación manual usando similitud coseno"""
    
    def __init__(self):
        super().__init__()
        self.index_built = False
    
    def build_index(self, fragments: List[Dict], embeddings: np.ndarray) -> bool:
        """
        Construye el índice manual (simplemente guarda los datos)
        
        Args:
            fragments: Lista de fragmentos
            embeddings: Array de embeddings
            
        Returns:
            True si se construyó correctamente
        """
        try:
            self.fragments = {frag['id']: frag for frag in fragments}
            self.embeddings = embeddings
            self.fragment_ids = [frag['id'] for frag in fragments]
            self.index_built = True
            
            print(f"✅ Índice manual construido: {len(fragments)} fragmentos")
            return True
            
        except Exception as e:
            print(f"❌ Error construyendo índice manual: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Realiza búsqueda usando similitud coseno manual
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (fragment_id, similarity_score)
        """
        if not self.index_built or self.embeddings is None:
            print("❌ Índice no construido")
            return []
        
        try:
            # Normalizar embeddings para similitud coseno
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            
            # Calcular similitud coseno
            similarities = np.dot(embeddings_norm, query_norm)
            
            # Obtener índices de los top_k más similares
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Crear lista de resultados
            results = [
                (self.fragment_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda manual: {e}")
            return []
    
    def search_with_details(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Búsqueda con detalles completos de los fragmentos
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de fragmentos con scores de similitud
        """
        search_results = self.search(query_embedding, top_k)
        detailed_results = []
        
        for fragment_id, similarity in search_results:
            if fragment_id in self.fragments:
                fragment = self.fragments[fragment_id].copy()
                fragment['similarity_score'] = similarity
                detailed_results.append(fragment)
        
        return detailed_results


class FAISSRetriever(BaseRetriever):
    """Sistema de recuperación usando FAISS"""
    
    def __init__(self, index_type: str = "flat"):
        """
        Inicializa el retriever FAISS
        
        Args:
            index_type: Tipo de índice FAISS ('flat', 'ivf', 'hnsw')
        """
        super().__init__()
        self.index_type = index_type
        self.index = None
        self.dimension = None
    
    def build_index(self, fragments: List[Dict], embeddings: np.ndarray) -> bool:
        """
        Construye el índice FAISS
        
        Args:
            fragments: Lista de fragmentos
            embeddings: Array de embeddings
            
        Returns:
            True si se construyó correctamente
        """
        try:
            self.fragments = {frag['id']: frag for frag in fragments}
            self.embeddings = embeddings.astype('float32')
            self.fragment_ids = [frag['id'] for frag in fragments]
            self.dimension = embeddings.shape[1]
            
            # Construir índice según el tipo
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(100, len(fragments) // 10)  # Número de clusters
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                # Entrenar el índice
                self.index.train(self.embeddings)
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Tipo de índice no soportado: {self.index_type}")
            
            # Normalizar embeddings para similitud coseno
            faiss.normalize_L2(self.embeddings)
            
            # Agregar embeddings al índice
            self.index.add(self.embeddings)
            
            print(f"✅ Índice FAISS ({self.index_type}) construido: {len(fragments)} fragmentos")
            return True
            
        except Exception as e:
            print(f"❌ Error construyendo índice FAISS: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Realiza búsqueda usando FAISS
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (fragment_id, similarity_score)
        """
        if self.index is None:
            print("❌ Índice FAISS no construido")
            return []
        
        try:
            # Preparar query
            query = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query)
            
            # Buscar
            scores, indices = self.index.search(query, top_k)
            
            # Crear lista de resultados
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.fragment_ids):
                    results.append((self.fragment_ids[idx], float(score)))
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda FAISS: {e}")
            return []
    
    def save_index(self, filepath: str = "data/faiss_index.bin"):
        """Guarda el índice FAISS"""
        try:
            faiss.write_index(self.index, filepath)
            print(f"✅ Índice FAISS guardado: {filepath}")
        except Exception as e:
            print(f"❌ Error guardando índice: {e}")
    
    def load_index(self, filepath: str = "data/faiss_index.bin"):
        """Carga el índice FAISS"""
        try:
            self.index = faiss.read_index(filepath)
            print(f"✅ Índice FAISS cargado: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error cargando índice: {e}")
            return False


class ChromaDBRetriever(BaseRetriever):
    """Sistema de recuperación usando ChromaDB"""
    
    def __init__(self, collection_name: str = "rag_fragments"):
        super().__init__()
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    def build_index(self, fragments: List[Dict], embeddings: np.ndarray) -> bool:
        """
        Construye el índice ChromaDB
        
        Args:
            fragments: Lista de fragmentos
            embeddings: Array de embeddings
            
        Returns:
            True si se construyó correctamente
        """
        try:
            # Inicializar cliente ChromaDB
            self.client = chromadb.Client()
            
            # Crear o obtener colección
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass  # Colección no existe
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Preparar datos
            ids = [frag['id'] for frag in fragments]
            documents = [frag['text'] for frag in fragments]
            metadatas = [
                {
                    'title': frag['title'],
                    'fragment_id': frag['fragment_id'],
                    'length': frag['length']
                }
                for frag in fragments
            ]
            
            # Agregar a la colección
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )
            
            self.fragments = {frag['id']: frag for frag in fragments}
            
            print(f"✅ Índice ChromaDB construido: {len(fragments)} fragmentos")
            return True
            
        except Exception as e:
            print(f"❌ Error construyendo índice ChromaDB: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Realiza búsqueda usando ChromaDB
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (fragment_id, similarity_score)
        """
        if self.collection is None:
            print("❌ Colección ChromaDB no inicializada")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            fragment_ids = results['ids'][0]
            distances = results['distances'][0]
            
            # Convertir distancias a similitudes (ChromaDB usa distancia coseno)
            similarities = [1 - dist for dist in distances]
            
            return list(zip(fragment_ids, similarities))
            
        except Exception as e:
            print(f"❌ Error en búsqueda ChromaDB: {e}")
            return []


class RetrieverComparator:
    """Clase para comparar diferentes sistemas de recuperación"""
    
    def __init__(self):
        self.retrievers = {}
        self.model = None
    
    def add_retriever(self, name: str, retriever: BaseRetriever):
        """Agrega un retriever para comparación"""
        self.retrievers[name] = retriever
    
    def load_sentence_transformer(self, model_name: str = "all-mpnet-base-v2"):
        """Carga el modelo para generar embeddings de consultas"""
        try:
            self.model = SentenceTransformer(model_name)
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def compare_search(self, query: str, top_k: int = 5) -> Dict:
        """
        Compara la búsqueda entre diferentes retrievers
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados
            
        Returns:
            Diccionario con resultados de cada retriever
        """
        if not self.model:
            if not self.load_sentence_transformer():
                return {}
        
        # Generar embedding de la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        results = {}
        
        for name, retriever in self.retrievers.items():
            print(f"🔍 Buscando con {name}...")
            start_time = time.time()
            
            search_results = retriever.search(query_embedding, top_k)
            
            end_time = time.time()
            
            results[name] = {
                'results': search_results,
                'search_time': end_time - start_time,
                'top_fragments': retriever.get_fragment_details([r[0] for r in search_results[:3]])
            }
        
        return results
    
    def benchmark_retrievers(self, test_queries: List[str], top_k: int = 5) -> Dict:
        """
        Realiza benchmark de los retrievers con múltiples consultas
        
        Args:
            test_queries: Lista de consultas de prueba
            top_k: Número de resultados por consulta
            
        Returns:
            Estadísticas de rendimiento
        """
        import time
        
        if not self.model:
            if not self.load_sentence_transformer():
                return {}
        
        benchmark_results = {name: {'times': [], 'total_results': 0} 
                           for name in self.retrievers.keys()}
        
        print(f"🚀 Benchmark con {len(test_queries)} consultas...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"  Consulta {i}/{len(test_queries)}: {query[:50]}...")
            
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            for name, retriever in self.retrievers.items():
                start_time = time.time()
                results = retriever.search(query_embedding, top_k)
                end_time = time.time()
                
                benchmark_results[name]['times'].append(end_time - start_time)
                benchmark_results[name]['total_results'] += len(results)
        
        # Calcular estadísticas
        stats = {}
        for name, data in benchmark_results.items():
            times = data['times']
            stats[name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_results': data['total_results'],
                'avg_results_per_query': data['total_results'] / len(test_queries)
            }
        
        return stats


def main():
    """Función principal para probar los retrievers"""
    import time
    
    # Crear retrievers
    manual_retriever = ManualRetriever()
    faiss_retriever = FAISSRetriever("flat")
    chroma_retriever = ChromaDBRetriever()
    
    # Cargar datos
    print("📖 Cargando datos...")
    if not manual_retriever.load_data():
        print("❌ No se pudieron cargar los datos")
        return
    
    # Construir índices
    fragments_list = list(manual_retriever.fragments.values())
    embeddings = manual_retriever.embeddings
    
    print("🔨 Construyendo índices...")
    manual_retriever.build_index(fragments_list, embeddings)
    faiss_retriever.build_index(fragments_list, embeddings)
    chroma_retriever.build_index(fragments_list, embeddings)
    
    # Comparar retrievers
    comparator = RetrieverComparator()
    comparator.add_retriever("Manual", manual_retriever)
    comparator.add_retriever("FAISS", faiss_retriever)
    comparator.add_retriever("ChromaDB", chroma_retriever)
    
    # Consulta de prueba
    test_query = "What is machine learning?"
    print(f"\n🔍 Comparando con consulta: '{test_query}'")
    
    comparison = comparator.compare_search(test_query, top_k=3)
    
    for retriever_name, data in comparison.items():
        print(f"\n📊 {retriever_name}:")
        print(f"  Tiempo: {data['search_time']:.4f}s")
        print(f"  Resultados:")
        for i, (frag_id, score) in enumerate(data['results'], 1):
            print(f"    {i}. {frag_id}: {score:.4f}")


if __name__ == "__main__":
    main()
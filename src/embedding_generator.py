"""
Módulo para generación de embeddings semánticos
Utiliza sentence-transformers para crear representaciones vectoriales
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import torch


class EmbeddingGenerator:
    """Generador de embeddings semánticos para fragmentos de texto"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Inicializa el generador de embeddings
        
        Args:
            model_name: Nombre del modelo de sentence-transformers a utilizar
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.fragment_ids = None
        
    def load_model(self) -> bool:
        """
        Carga el modelo de sentence-transformers
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            print(f"🔄 Cargando modelo {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
            # Verificar si hay GPU disponible
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📱 Usando dispositivo: {device}")
            
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def load_fragments(self, corpus_path: str = "data/corpus.json") -> List[Dict]:
        """
        Carga los fragmentos desde el archivo de corpus
        
        Args:
            corpus_path: Ruta al archivo de corpus
            
        Returns:
            Lista de fragmentos
        """
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            fragments = corpus_data.get('fragments', [])
            print(f"✅ Cargados {len(fragments)} fragmentos desde {corpus_path}")
            return fragments
            
        except FileNotFoundError:
            print(f"❌ No se encontró el archivo {corpus_path}")
            return []
        except Exception as e:
            print(f"❌ Error cargando fragmentos: {e}")
            return []
    
    def generate_embeddings(self, fragments: List[Dict], 
                          batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Genera embeddings para una lista de fragmentos
        
        Args:
            fragments: Lista de fragmentos con texto
            batch_size: Tamaño de lote para procesamiento
            
        Returns:
            Tupla con (embeddings, fragment_ids)
        """
        if not self.model:
            if not self.load_model():
                return None, None
        
        if not fragments:
            print("❌ No hay fragmentos para procesar")
            return None, None
        
        print(f"🔄 Generando embeddings para {len(fragments)} fragmentos...")
        
        # Extraer textos e IDs
        texts = [fragment['text'] for fragment in fragments]
        fragment_ids = [fragment['id'] for fragment in fragments]
        
        try:
            # Generar embeddings en lotes
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✅ Embeddings generados: {embeddings.shape}")
            
            # Guardar en la instancia
            self.embeddings = embeddings
            self.fragment_ids = fragment_ids
            
            return embeddings, fragment_ids
            
        except Exception as e:
            print(f"❌ Error generando embeddings: {e}")
            return None, None
    
    def save_embeddings(self, embeddings: np.ndarray, fragment_ids: List[str],
                       output_dir: str = "data/"):
        """
        Guarda los embeddings en diferentes formatos
        
        Args:
            embeddings: Array de embeddings
            fragment_ids: Lista de IDs de fragmentos
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Guardar embeddings como numpy array
        npy_path = output_path / "embeddings.npy"
        np.save(npy_path, embeddings)
        print(f"✅ Embeddings guardados en: {npy_path}")
        
        # Guardar IDs de fragmentos
        ids_path = output_path / "fragment_ids.json"
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(fragment_ids, f, ensure_ascii=False, indent=2)
        print(f"✅ IDs de fragmentos guardados en: {ids_path}")
        
        # Guardar metadatos
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_fragments': embeddings.shape[0],
            'embedding_shape': embeddings.shape
        }
        
        metadata_path = output_path / "embeddings_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Metadatos guardados en: {metadata_path}")
        
        # Guardar todo junto en formato pickle (para facilitar la carga)
        pickle_data = {
            'embeddings': embeddings,
            'fragment_ids': fragment_ids,
            'metadata': metadata
        }
        
        pickle_path = output_path / "embeddings_complete.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        print(f"✅ Datos completos guardados en: {pickle_path}")
    
    def load_embeddings(self, data_dir: str = "data/") -> Tuple[np.ndarray, List[str]]:
        """
        Carga embeddings previamente generados
        
        Args:
            data_dir: Directorio donde están guardados los embeddings
            
        Returns:
            Tupla con (embeddings, fragment_ids)
        """
        data_path = Path(data_dir)
        
        try:
            # Intentar cargar desde pickle primero (más rápido)
            pickle_path = data_path / "embeddings_complete.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.embeddings = data['embeddings']
                self.fragment_ids = data['fragment_ids']
                
                print(f"✅ Embeddings cargados desde pickle: {self.embeddings.shape}")
                return self.embeddings, self.fragment_ids
            
            # Cargar desde archivos separados
            npy_path = data_path / "embeddings.npy"
            ids_path = data_path / "fragment_ids.json"
            
            if npy_path.exists() and ids_path.exists():
                embeddings = np.load(npy_path)
                
                with open(ids_path, 'r', encoding='utf-8') as f:
                    fragment_ids = json.load(f)
                
                self.embeddings = embeddings
                self.fragment_ids = fragment_ids
                
                print(f"✅ Embeddings cargados: {embeddings.shape}")
                return embeddings, fragment_ids
            
            print("❌ No se encontraron archivos de embeddings")
            return None, None
            
        except Exception as e:
            print(f"❌ Error cargando embeddings: {e}")
            return None, None
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Genera embedding para una consulta
        
        Args:
            query: Texto de la consulta
            
        Returns:
            Embedding de la consulta
        """
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            return query_embedding[0]
        except Exception as e:
            print(f"❌ Error generando embedding de consulta: {e}")
            return None
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Calcula similitudes entre consulta y fragmentos
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de fragmentos más similares a retornar
            
        Returns:
            Lista de tuplas (fragment_id, similarity_score)
        """
        if self.embeddings is None or self.fragment_ids is None:
            print("❌ No hay embeddings cargados")
            return []
        
        try:
            # Calcular similitud coseno
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Obtener indices de los top_k más similares
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Crear lista de resultados
            results = [
                (self.fragment_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            return results
            
        except Exception as e:
            print(f"❌ Error calculando similitudes: {e}")
            return []
    
    def get_embedding_stats(self) -> Dict:
        """
        Obtiene estadísticas de los embeddings
        
        Returns:
            Diccionario con estadísticas
        """
        if self.embeddings is None:
            return {}
        
        return {
            'shape': self.embeddings.shape,
            'model_name': self.model_name,
            'mean_norm': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(self.embeddings, axis=1))),
            'embedding_dim': self.embeddings.shape[1],
            'num_fragments': self.embeddings.shape[0]
        }


def main():
    """Función principal para generar embeddings"""
    generator = EmbeddingGenerator()
    
    # Cargar fragmentos
    fragments = generator.load_fragments()
    
    if fragments:
        # Generar embeddings
        embeddings, fragment_ids = generator.generate_embeddings(fragments)
        
        if embeddings is not None:
            # Guardar embeddings
            generator.save_embeddings(embeddings, fragment_ids)
            
            # Mostrar estadísticas
            stats = generator.get_embedding_stats()
            print("\n📊 Estadísticas de embeddings:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No se pudieron generar los embeddings")
    else:
        print("❌ No se pudieron cargar los fragmentos")


if __name__ == "__main__":
    main()
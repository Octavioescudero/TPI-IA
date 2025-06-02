"""
RAG Pipeline - Trabajo Práctico Integrador IA 2025
UTN-FRC

Pipeline completo para Retrieval-Augmented Generation que integra:
- Construcción del corpus desde Wikipedia
- Generación de embeddings semánticos
- Sistemas de recuperación (manual y vectorial)
- Generación de respuestas con TinyLlama
"""

import json
import csv
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Para manejo de errores y logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Pipeline completo de RAG que integra todos los componentes del chatbot.
    """
    
    def __init__(self, 
                 corpus_path: str = "data/corpus.json",
                 embeddings_path: str = "data/embeddings.npy",
                 fragments_path: str = "data/fragments.csv",
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Inicializa el pipeline RAG.
        
        Args:
            corpus_path: Ruta al archivo del corpus
            embeddings_path: Ruta a los embeddings guardados
            fragments_path: Ruta a los fragmentos en CSV
            model_name: Nombre del modelo generativo a usar
        """
        self.corpus_path = corpus_path
        self.embeddings_path = embeddings_path
        self.fragments_path = fragments_path
        self.model_name = model_name
        
        # Componentes del pipeline
        self.corpus_builder = None
        self.embedding_generator = None
        self.manual_retriever = None
        self.vector_retriever = None
        self.response_generator = None
        
        # Datos
        self.corpus = None
        self.fragments = None
        self.embeddings = None
        
        # Estado del pipeline
        self.is_initialized = False
        
    def initialize_components(self):
        """Inicializa todos los componentes del pipeline."""
        try:
            from src.corpus_builder import CorpusBuilder
            from src.embedding_generator import EmbeddingGenerator
            from src.retrievers import ManualRetriever, VectorRetriever
            from src.response_generator import ResponseGenerator
            
            self.corpus_builder = CorpusBuilder()
            self.embedding_generator = EmbeddingGenerator()
            self.manual_retriever = ManualRetriever()
            self.vector_retriever = VectorRetriever()
            self.response_generator = ResponseGenerator(model_name=self.model_name)
            
            logger.info("Componentes del pipeline inicializados correctamente")
            
        except ImportError as e:
            logger.error(f"Error al importar componentes: {e}")
            raise
    
    def load_data(self):
        """Carga los datos necesarios (corpus, fragmentos, embeddings)."""
        try:
            # Cargar corpus
            if Path(self.corpus_path).exists():
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                logger.info(f"Corpus cargado: {len(self.corpus)} artículos")
            
            # Cargar fragmentos
            if Path(self.fragments_path).exists():
                self.fragments = pd.read_csv(self.fragments_path)
                logger.info(f"Fragmentos cargados: {len(self.fragments)} fragmentos")
            
            # Cargar embeddings
            if Path(self.embeddings_path).exists():
                self.embeddings = np.load(self.embeddings_path)
                logger.info(f"Embeddings cargados: {self.embeddings.shape}")
                
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def build_corpus_from_titles(self, titles_file: str = "data/wikipedia_pages.txt"):
        """
        Construye el corpus desde la lista de títulos de Wikipedia.
        
        Args:
            titles_file: Archivo con los títulos de Wikipedia
        """
        if not self.corpus_builder:
            self.initialize_components()
            
        logger.info("Iniciando construcción del corpus...")
        
        # Leer títulos
        with open(titles_file, 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        # Construir corpus
        self.corpus = self.corpus_builder.build_corpus(titles)
        
        # Generar fragmentos
        self.fragments = self.corpus_builder.create_fragments(self.corpus)
        
        # Guardar datos
        self._save_corpus_and_fragments()
        
        logger.info(f"Corpus construido: {len(self.corpus)} artículos, {len(self.fragments)} fragmentos")
    
    def generate_embeddings(self):
        """Genera embeddings semánticos para los fragmentos."""
        if not self.embedding_generator:
            self.initialize_components()
            
        if self.fragments is None:
            raise ValueError("Los fragmentos no están disponibles. Construya el corpus primero.")
        
        logger.info("Generando embeddings semánticos...")
        
        # Generar embeddings
        texts = self.fragments['text'].tolist()
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Guardar embeddings
        np.save(self.embeddings_path, self.embeddings)
        
        logger.info(f"Embeddings generados y guardados: {self.embeddings.shape}")
    
    def setup_retrievers(self):
        """Configura los sistemas de recuperación."""
        if not self.manual_retriever or not self.vector_retriever:
            self.initialize_components()
        
        if self.embeddings is None or self.fragments is None:
            raise ValueError("Embeddings o fragmentos no disponibles")
        
        # Configurar retriever manual
        self.manual_retriever.setup(
            fragments=self.fragments['text'].tolist(),
            embeddings=self.embeddings
        )
        
        # Configurar retriever vectorial
        self.vector_retriever.setup(
            fragments=self.fragments['text'].tolist(),
            embeddings=self.embeddings
        )
        
        logger.info("Sistemas de recuperación configurados")
    
    def query(self, 
              question: str, 
              retriever_type: str = "vector",
              k: int = 5) -> Dict:
        """
        Procesa una consulta completa através del pipeline RAG.
        
        Args:
            question: Pregunta del usuario
            retriever_type: Tipo de retriever ("manual" o "vector")
            k: Número de fragmentos a recuperar
            
        Returns:
            Dict con la respuesta y metadatos
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Procesando consulta: {question[:50]}...")
        
        # Seleccionar retriever
        retriever = self.manual_retriever if retriever_type == "manual" else self.vector_retriever
        
        # Recuperar fragmentos relevantes
        relevant_fragments = retriever.retrieve(question, k=k)
        
        # Generar respuesta
        response = self.response_generator.generate_response(
            question=question,
            context_fragments=relevant_fragments
        )
        
        result = {
            "question": question,
            "answer": response,
            "retriever_type": retriever_type,
            "num_fragments": len(relevant_fragments),
            "fragments": relevant_fragments
        }
        
        logger.info(f"Respuesta generada: {response[:100]}...")
        
        return result
    
    def process_questions_file(self, 
                              questions_file: str = "data/preguntas.txt",
                              output_file: str = "outputs/submission.csv",
                              retriever_type: str = "vector",
                              k: int = 5) -> pd.DataFrame:
        """
        Procesa un archivo de preguntas y genera el archivo de submission.
        
        Args:
            questions_file: Archivo con las preguntas
            output_file: Archivo de salida CSV
            retriever_type: Tipo de retriever a usar
            k: Número de fragmentos a recuperar
            
        Returns:
            DataFrame con las respuestas
        """
        logger.info(f"Procesando archivo de preguntas: {questions_file}")
        
        # Leer preguntas
        questions_df = pd.read_csv(questions_file)
        
        results = []
        
        for idx, row in questions_df.iterrows():
            question_id = row['ID']
            question = row['question'] if 'question' in row else row['pregunta']
            
            try:
                # Procesar consulta
                result = self.query(
                    question=question,
                    retriever_type=retriever_type,
                    k=k
                )
                
                results.append({
                    'ID': question_id,
                    'answer': result['answer']
                })
                
                logger.info(f"Procesada pregunta {idx + 1}/{len(questions_df)}")
                
            except Exception as e:
                logger.error(f"Error procesando pregunta {question_id}: {e}")
                results.append({
                    'ID': question_id,
                    'answer': "Error al generar respuesta"
                })
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)
        
        # Guardar archivo de submission
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        
        logger.info(f"Archivo de submission guardado: {output_file}")
        
        return results_df
    
    def evaluate_performance(self, 
                           questions_file: str = "data/preguntas.txt",
                           reference_answers: Optional[str] = None) -> Dict:
        """
        Evalúa el rendimiento del pipeline.
        
        Args:
            questions_file: Archivo con preguntas de evaluación
            reference_answers: Archivo con respuestas de referencia (opcional)
            
        Returns:
            Dict con métricas de evaluación
        """
        logger.info("Evaluando rendimiento del pipeline...")
        
        # Procesar preguntas con ambos retrievers
        manual_results = self.process_questions_file(
            questions_file=questions_file,
            output_file="outputs/manual_results.csv",
            retriever_type="manual"
        )
        
        vector_results = self.process_questions_file(
            questions_file=questions_file,
            output_file="outputs/vector_results.csv",
            retriever_type="vector"
        )
        
        evaluation = {
            "total_questions": len(manual_results),
            "manual_retriever": {
                "completed": len(manual_results),
                "avg_response_length": manual_results['answer'].str.len().mean()
            },
            "vector_retriever": {
                "completed": len(vector_results),
                "avg_response_length": vector_results['answer'].str.len().mean()
            }
        }
        
        # Guardar evaluación
        with open("outputs/evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        logger.info("Evaluación completada")
        
        return evaluation
    
    def initialize(self):
        """Inicializa completamente el pipeline."""
        logger.info("Inicializando pipeline RAG...")
        
        # Inicializar componentes
        self.initialize_components()
        
        # Cargar datos existentes
        self.load_data()
        
        # Si no hay datos, construir desde cero
        if self.corpus is None:
            logger.warning("No se encontró corpus. Construyendo desde títulos...")
            self.build_corpus_from_titles()
        
        if self.embeddings is None:
            logger.warning("No se encontraron embeddings. Generando...")
            self.generate_embeddings()
        
        # Configurar retrievers
        self.setup_retrievers()
        
        self.is_initialized = True
        logger.info("Pipeline RAG inicializado correctamente")
    
    def _save_corpus_and_fragments(self):
        """Guarda el corpus y fragmentos."""
        # Crear directorios si no existen
        Path(self.corpus_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.fragments_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar corpus
        with open(self.corpus_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, indent=2, ensure_ascii=False)
        
        # Guardar fragmentos
        self.fragments.to_csv(self.fragments_path, index=False)
    
    def get_pipeline_stats(self) -> Dict:
        """Retorna estadísticas del pipeline."""
        stats = {
            "initialized": self.is_initialized,
            "corpus_articles": len(self.corpus) if self.corpus else 0,
            "total_fragments": len(self.fragments) if self.fragments is not None else 0,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "model_name": self.model_name
        }
        
        return stats


def main():
    """Función principal para demostrar el uso del pipeline."""
    # Crear pipeline
    rag = RAGPipeline()
    
    # Inicializar
    rag.initialize()
    
    # Ejemplo de consulta
    question = "¿Qué son las redes neuronales?"
    result = rag.query(question, retriever_type="vector", k=3)
    
    print(f"Pregunta: {result['question']}")
    print(f"Respuesta: {result['answer']}")
    print(f"Fragmentos usados: {result['num_fragments']}")
    
    # Procesar archivo de preguntas
    submission_df = rag.process_questions_file()
    print(f"Archivo de submission generado con {len(submission_df)} respuestas")
    
    # Mostrar estadísticas
    stats = rag.get_pipeline_stats()
    print("Estadísticas del pipeline:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

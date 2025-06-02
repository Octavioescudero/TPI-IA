# Plan de Desarrollo - Chatbot RAG UTN-FRC 2025

## Resumen del Proyecto

Desarrollar un chatbot basado en **Retrieval-Augmented Generation (RAG)** que responda preguntas en español sobre redes neuronales y aprendizaje profundo, utilizando artículos de Wikipedia en inglés como corpus de conocimiento.

### Fechas Clave
- **Entrega final**: 19 de junio de 2025 (23:59 hs)
- **Compartir notebook Kaggle**: 20 de junio de 2025 (12:00 hs)
- **Presentaciones orales**: A confirmar

---

## Arquitectura del Sistema RAG

```
Pregunta (Español) → Búsqueda Semántica → Fragmentos Relevantes → Modelo Generativo → Respuesta (Español)
```

### Componentes Principales:
1. **Corpus de Wikipedia** (inglés)
2. **Sistema de Embeddings** (sentence-transformers)
3. **Búsqueda Semántica** (cosine similarity + FAISS/ChromaDB)
4. **Modelo Generativo** (TinyLlama)

---

## Fase 1: Construcción del Corpus

### Objetivos:
- Descargar contenido de Wikipedia usando la API
- Limpiar y procesar texto
- Fragmentar en bloques manejables (~300 caracteres)
- Almacenar en formato JSON/CSV

### Implementación:

#### 1.1 Descarga de Artículos
```python
import wikipediaapi
import pandas as pd
import json
import re

def download_wikipedia_articles(titles_list):
    """
    Descarga artículos de Wikipedia usando la API
    """
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='TPI-IA-2025 (your.email@example.com)'
    )
    
    articles = []
    for title in titles_list:
        page = wiki_wiki.page(title)
        if page.exists():
            articles.append({
                'title': title,
                'content': page.text,
                'url': page.fullurl
            })
        else:
            print(f"Artículo no encontrado: {title}")
    
    return articles
```

#### 1.2 Limpieza de Texto
```python
def clean_wikipedia_text(text):
    """
    Limpia el texto de Wikipedia eliminando referencias y secciones no deseadas
    """
    # Eliminar referencias [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Eliminar secciones como "See also", "References", etc.
    sections_to_remove = [
        'See also', 'References', 'External links', 
        'Further reading', 'Bibliography'
    ]
    
    for section in sections_to_remove:
        pattern = f'{section}.*?(?=\n\n|\Z)'
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Limpiar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

#### 1.3 Fragmentación de Texto
```python
def fragment_text(text, max_length=300, overlap=50):
    """
    Fragmenta el texto en bloques de tamaño manejable con solapamiento
    """
    sentences = text.split('. ')
    fragments = []
    current_fragment = ""
    
    for sentence in sentences:
        if len(current_fragment + sentence) <= max_length:
            current_fragment += sentence + ". "
        else:
            if current_fragment:
                fragments.append(current_fragment.strip())
            current_fragment = sentence + ". "
    
    if current_fragment:
        fragments.append(current_fragment.strip())
    
    return fragments
```

### Entregables Fase 1:
- `corpus.json`: Corpus completo con artículos procesados
- `fragments.csv`: Fragmentos de texto indexados
- Script de descarga y procesamiento

---

## Fase 2: Indexación y Búsqueda Semántica

### Objetivos:
- Generar embeddings semánticos
- Implementar búsqueda manual (cosine similarity)
- Implementar búsqueda con FAISS/ChromaDB
- Comparar ambos enfoques

### Implementación:

#### 2.1 Generación de Embeddings
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts):
        """Genera embeddings para una lista de textos"""
        return self.model.encode(texts, convert_to_tensor=True)
    
    def encode_query(self, query):
        """Codifica una consulta individual"""
        return self.model.encode([query], convert_to_tensor=True)
```

#### 2.2 Búsqueda Manual (Cosine Similarity)
```python
from sklearn.metrics.pairwise import cosine_similarity

class ManualRetriever:
    def __init__(self, fragments, embeddings):
        self.fragments = fragments
        self.embeddings = embeddings
    
    def search(self, query_embedding, top_k=5):
        """Busca los fragmentos más similares usando cosine similarity"""
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'fragment': self.fragments[idx],
                'score': similarities[0][idx],
                'index': idx
            })
        
        return results
```

#### 2.3 Búsqueda con FAISS
```python
import faiss

class FAISSRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]
        self.index = None
        self.build_index()
    
    def build_index(self):
        """Construye el índice FAISS"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product
        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def search(self, query_embedding, top_k=5):
        """Busca usando FAISS"""
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                'fragment': self.fragments[idx],
                'score': float(score),
                'index': int(idx)
            })
        
        return results
```

### Entregables Fase 2:
- Sistema de embeddings configurado
- Implementación de búsqueda manual
- Implementación con FAISS/ChromaDB
- Comparación de rendimiento entre ambos métodos

---

## Fase 3: Generación de Respuestas

### Objetivos:
- Integrar modelo generativo TinyLlama
- Diseñar prompts efectivos
- Implementar pipeline completo RAG
- Generar respuestas en español

### Implementación:

#### 3.1 Integración de TinyLlama
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ResponseGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt, max_length=256):
        """Genera respuesta usando TinyLlama"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
```

#### 3.2 Diseño de Prompts
```python
def create_rag_prompt(question, context_fragments):
    """
    Crea un prompt efectivo para el modelo RAG
    """
    context = "\n".join([f"- {fragment}" for fragment in context_fragments])
    
    prompt = f"""Eres un asistente experto en redes neuronales y aprendizaje profundo. 
Basándote ÚNICAMENTE en la información proporcionada, responde la pregunta en español de manera clara y precisa.

Contexto:
{context}

Pregunta: {question}

Respuesta en español:"""
    
    return prompt
```

#### 3.3 Pipeline RAG Completo
```python
class RAGChatbot:
    def __init__(self, fragments, embeddings, retriever, generator):
        self.fragments = fragments
        self.embeddings = embeddings
        self.retriever = retriever
        self.generator = generator
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    def answer_question(self, question, top_k=3):
        """Pipeline completo RAG"""
        # 1. Generar embedding de la pregunta
        query_embedding = self.embedding_model.encode([question])
        
        # 2. Recuperar fragmentos relevantes
        retrieved_docs = self.retriever.search(query_embedding, top_k)
        context_fragments = [doc['fragment'] for doc in retrieved_docs]
        
        # 3. Crear prompt
        prompt = create_rag_prompt(question, context_fragments)
        
        # 4. Generar respuesta
        response = self.generator.generate_response(prompt)
        
        return {
            'question': question,
            'answer': response,
            'context': context_fragments,
            'retrieval_scores': [doc['score'] for doc in retrieved_docs]
        }
```

### Entregables Fase 3:
- Modelo TinyLlama integrado
- Sistema de prompts optimizado
- Pipeline RAG completo
- Archivo `submission.csv` con respuestas

---

## Fase 4: Evaluación y Optimización

### Métricas de Evaluación:
1. **Similitud Semántica**: Cosine similarity con respuestas esperadas
2. **Relevancia de Recuperación**: Calidad de fragmentos recuperados
3. **Coherencia de Respuestas**: Evaluación cualitativa
4. **Eficiencia**: Tiempo de procesamiento

### Optimizaciones Sugeridas:

#### 4.1 Mejora de Fragmentación
```python
def smart_fragmentation(text, max_length=300):
    """
    Fragmentación inteligente respetando párrafos y oraciones
    """
    paragraphs = text.split('\n\n')
    fragments = []
    
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            fragments.append(paragraph)
        else:
            # Dividir por oraciones si el párrafo es muy largo
            sentences = paragraph.split('. ')
            current_fragment = ""
            
            for sentence in sentences:
                if len(current_fragment + sentence) <= max_length:
                    current_fragment += sentence + ". "
                else:
                    if current_fragment:
                        fragments.append(current_fragment.strip())
                    current_fragment = sentence + ". "
            
            if current_fragment:
                fragments.append(current_fragment.strip())
    
    return fragments
```

#### 4.2 Re-ranking de Resultados
```python
def rerank_results(query, retrieved_docs, rerank_model):
    """
    Re-ordena los resultados usando un modelo de re-ranking
    """
    pairs = [(query, doc['fragment']) for doc in retrieved_docs]
    rerank_scores = rerank_model.predict(pairs)
    
    # Combinar scores de recuperación y re-ranking
    for i, doc in enumerate(retrieved_docs):
        doc['combined_score'] = 0.7 * doc['score'] + 0.3 * rerank_scores[i]
    
    return sorted(retrieved_docs, key=lambda x: x['combined_score'], reverse=True)
```

---

## Estructura de Archivos del Proyecto

```
TPI_RAG_2025/
├── notebooks/
│   ├── 01_corpus_construction.ipynb
│   ├── 02_embedding_indexing.ipynb
│   ├── 03_retrieval_systems.ipynb
│   ├── 04_response_generation.ipynb
│   └── 05_full_pipeline.ipynb
├── src/
│   ├── corpus_builder.py
│   ├── embedding_generator.py
│   ├── retrievers.py
│   ├── response_generator.py
│   └── rag_pipeline.py
├── data/
│   ├── wikipedia_pages.txt
│   ├── preguntas.txt
│   ├── corpus.json
│   ├── fragments.csv
│   └── embeddings.npy
├── outputs/
│   ├── submission.csv
│   └── evaluation_results.json
├── requirements.txt
└── README.md
```

---

## Cronograma de Desarrollo

### Semana 1 (Días 1-7):
- [ ] Configuración del entorno
- [ ] Descarga y procesamiento del corpus
- [ ] Implementación de limpieza de texto

### Semana 2 (Días 8-14):
- [ ] Generación de embeddings
- [ ] Implementación de búsqueda manual
- [ ] Implementación con FAISS/ChromaDB

### Semana 3 (Días 15-21):
- [ ] Integración de TinyLlama
- [ ] Desarrollo de prompts
- [ ] Pipeline RAG completo

### Semana 4 (Días 22-28):
- [ ] Evaluación y optimización
- [ ] Generación de submission.csv
- [ ] Preparación del informe técnico

### Semana 5 (Días 29-35):
- [ ] Refinamiento final
- [ ] Documentación
- [ ] Preparación de presentación oral

---

## Consideraciones Técnicas

### Recursos Computacionales:
- **Kaggle**: 30 horas GPU/semana (usar eficientemente)
- **CPU**: TinyLlama debe ejecutarse en CPU
- **Memoria**: Optimizar carga de embeddings

### Bibliotecas Principales:
```python
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
wikipedia-api>=0.6.0
```

### Buenas Prácticas:
1. **Versionado**: Usar git para control de versiones
2. **Reproducibilidad**: Fijar seeds aleatorias
3. **Modularidad**: Código bien estructurado y documentado
4. **Eficiencia**: Procesamiento por lotes cuando sea posible
5. **Validación**: Testing de componentes individuales

---

## Entregables Finales

### 1. Notebook de Kaggle
- Desarrollo completo reproducible
- Comentarios explicativos
- Resultados de evaluación
- Comparación de métodos

### 2. Informe Técnico (PDF)
- Descripción de la arquitectura
- Decisiones técnicas justificadas
- Análisis de resultados
- Conclusiones y mejoras futuras

### 3. Archivo submission.csv
```csv
ID,answer
1,"El perceptrón fue creado por Frank Rosenblatt en 1958..."
2,"Las redes neuronales convolucionales procesan datos espaciales..."
...
```

### 4. Presentación Oral (15 min)
- Demostración del sistema
- Explicación de decisiones técnicas
- Análisis de resultados
- Respuesta a preguntas

---

## Criterios de Evaluación

1. **Funcionalidad**: Sistema RAG completo y funcional
2. **Calidad Técnica**: Implementación robusta y eficiente
3. **Innovación**: Enfoques creativos y optimizaciones
4. **Documentación**: Código bien documentado e informe claro
5. **Reproducibilidad**: Capacidad de ejecutar el sistema completo
6. **Resultados**: Performance en la competencia Kaggle

---

## Recursos Adicionales

### Documentación Útil:
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [TinyLlama Model](https://huggingface.co/TinyLlama)
- [Wikipedia API](https://wikipedia-api.readthedocs.io/)

### Papers de Referencia:
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "Dense Passage Retrieval for Open-Domain Question Answering"
- "REALM: Retrieval-Augmented Language Model Pre-Training"

¡Éxito en el desarrollo de tu chatbot RAG! 🚀

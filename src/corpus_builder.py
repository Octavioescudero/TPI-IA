"""
corpus_builder.py
Módulo para construcción del corpus de Wikipedia para el TPI de IA 2025
UTN-FRC - Chatbot RAG

Este módulo se encarga de:
1. Descargar artículos de Wikipedia usando la API
2. Limpiar y procesar el contenido
3. Fragmentar el texto en bloques manejables
4. Guardar el corpus en formatos JSON y CSV
"""

import json
import pandas as pd
import re
import time
import logging
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')

class WikipediaCorpusBuilder:
    """
    Constructor de corpus de Wikipedia para sistema RAG
    """
    
    def __init__(self, output_dir: str = "data", language: str = "en"):
        """
        Inicializa el constructor de corpus
        
        Args:
            output_dir: Directorio donde guardar los archivos
            language: Idioma de Wikipedia (por defecto 'en')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
        self.content_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Configurar nltk
        self._setup_nltk()
        
        # Estadísticas del proceso
        self.stats = {
            'total_articles': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_fragments': 0,
            'total_characters': 0
        }
    
    def _setup_nltk(self):
        """Descarga recursos necesarios de NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_titles_from_file(self, file_path: str) -> List[str]:
        """
        Carga la lista de títulos desde el archivo provisto por la cátedra
        
        Args:
            file_path: Ruta al archivo con títulos
            
        Returns:
            Lista de títulos de artículos
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Cargados {len(titles)} títulos desde {file_path}")
            return titles
        
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error al cargar títulos: {e}")
            return []
    
    def get_article_content(self, title: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Obtiene el contenido completo de un artículo de Wikipedia
        
        Args:
            title: Título del artículo
            max_retries: Número máximo de reintentos
            
        Returns:
            Diccionario con información del artículo o None si falla
        """
        for attempt in range(max_retries):
            try:
                # Primero obtener información básica
                summary_response = requests.get(
                    f"{self.base_url}{title.replace(' ', '_')}",
                    timeout=10
                )
                
                if summary_response.status_code != 200:
                    logger.warning(f"Artículo no encontrado: {title}")
                    return None
                
                summary_data = summary_response.json()
                
                # Obtener contenido completo usando la API de Wikipedia
                params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'extracts',
                    'exintro': False,
                    'explaintext': True,
                    'exsectionformat': 'plain'
                }
                
                content_response = requests.get(self.content_url, params=params, timeout=15)
                content_data = content_response.json()
                
                # Extraer el contenido del texto
                pages = content_data.get('query', {}).get('pages', {})
                page_data = next(iter(pages.values()))
                
                if 'extract' not in page_data:
                    logger.warning(f"No se pudo obtener contenido para: {title}")
                    return None
                
                article_info = {
                    'title': title,
                    'pageid': page_data.get('pageid'),
                    'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'extract': summary_data.get('extract', ''),
                    'content': page_data.get('extract', ''),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                logger.info(f"✓ Descargado: {title} ({len(article_info['content'])} caracteres)")
                return article_info
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Intento {attempt + 1} falló para {title}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                
            except Exception as e:
                logger.error(f"Error inesperado al descargar {title}: {e}")
                break
        
        logger.error(f"✗ Falló la descarga después de {max_retries} intentos: {title}")
        return None
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        Limpia el texto de Wikipedia eliminando elementos no deseados
        
        Args:
            text: Texto crudo de Wikipedia
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Eliminar referencias [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Eliminar enlaces internos [[texto]]
        text = re.sub(r'\[\[([^\]]*\|)?([^\]]*)\]\]', r'\2', text)
        
        # Eliminar referencias a archivos y medios
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Media:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Eliminar plantillas {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        
        # Eliminar secciones no deseadas
        sections_to_remove = [
            r'==\s*See also\s*==.*?(?=\n==|\Z)',
            r'==\s*References\s*==.*?(?=\n==|\Z)',
            r'==\s*External links\s*==.*?(?=\n==|\Z)',
            r'==\s*Further reading\s*==.*?(?=\n==|\Z)',
            r'==\s*Bibliography\s*==.*?(?=\n==|\Z)',
            r'==\s*Notes\s*==.*?(?=\n==|\Z)'
        ]
        
        for pattern in sections_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Limpiar encabezados de sección
        text = re.sub(r'==+\s*([^=]+)\s*==+', r'\1', text)
        
        # Eliminar líneas vacías múltiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres especiales problemáticos
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\'°%]', '', text)
        
        return text.strip()
    
    def smart_fragment_text(self, text: str, max_length: int = 300, 
                          min_length: int = 50, overlap: int = 50) -> List[str]:
        """
        Fragmenta texto de manera inteligente respetando estructura semántica
        
        Args:
            text: Texto a fragmentar
            max_length: Longitud máxima por fragmento
            min_length: Longitud mínima por fragmento
            overlap: Solapamiento entre fragmentos
            
        Returns:
            Lista de fragmentos de texto
        """
        if not text or len(text) < min_length:
            return [text] if text else []
        
        # Dividir en párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        fragments = []
        
        for paragraph in paragraphs:
            if len(paragraph) <= max_length:
                if len(paragraph) >= min_length:
                    fragments.append(paragraph)
            else:
                # Fragmentar párrafo largo por oraciones
                sentences = sent_tokenize(paragraph)
                current_fragment = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # Si la oración sola es muy larga, fragmentarla
                    if len(sentence) > max_length:
                        if current_fragment and len(current_fragment) >= min_length:
                            fragments.append(current_fragment.strip())
                            current_fragment = ""
                        
                        # Fragmentar oración larga por palabras
                        words = word_tokenize(sentence)
                        temp_fragment = ""
                        
                        for word in words:
                            if len(temp_fragment + " " + word) <= max_length:
                                temp_fragment += " " + word if temp_fragment else word
                            else:
                                if temp_fragment and len(temp_fragment) >= min_length:
                                    fragments.append(temp_fragment.strip())
                                temp_fragment = word
                        
                        if temp_fragment:
                            current_fragment = temp_fragment
                    
                    # Agregar oración al fragmento actual
                    elif len(current_fragment + " " + sentence) <= max_length:
                        current_fragment += " " + sentence if current_fragment else sentence
                    else:
                        # Guardar fragmento actual y empezar uno nuevo
                        if current_fragment and len(current_fragment) >= min_length:
                            fragments.append(current_fragment.strip())
                        current_fragment = sentence
                
                # Agregar último fragmento del párrafo
                if current_fragment and len(current_fragment) >= min_length:
                    fragments.append(current_fragment.strip())
        
        # Aplicar solapamiento si es necesario
        if overlap > 0 and len(fragments) > 1:
            fragments = self._apply_overlap(fragments, overlap)
        
        return [f for f in fragments if f and len(f) >= min_length]
    
    def _apply_overlap(self, fragments: List[str], overlap: int) -> List[str]:
        """
        Aplica solapamiento entre fragmentos consecutivos
        
        Args:
            fragments: Lista de fragmentos
            overlap: Número de caracteres de solapamiento
            
        Returns:
            Lista de fragmentos con solapamiento
        """
        if len(fragments) <= 1 or overlap <= 0:
            return fragments
        
        overlapped_fragments = []
        
        for i, fragment in enumerate(fragments):
            if i == 0:
                overlapped_fragments.append(fragment)
            else:
                # Obtener solapamiento del fragmento anterior
                prev_fragment = fragments[i-1]
                if len(prev_fragment) > overlap:
                    overlap_text = prev_fragment[-overlap:]
                    overlapped_fragment = overlap_text + " " + fragment
                    overlapped_fragments.append(overlapped_fragment)
                else:
                    overlapped_fragments.append(fragment)
        
        return overlapped_fragments
    
    def build_corpus(self, titles: List[str], fragment_params: Dict = None) -> Dict:
        """
        Construye el corpus completo a partir de la lista de títulos
        
        Args:
            titles: Lista de títulos de artículos
            fragment_params: Parámetros para fragmentación
            
        Returns:
            Diccionario con el corpus construido
        """
        if fragment_params is None:
            fragment_params = {
                'max_length': 300,
                'min_length': 50,
                'overlap': 50
            }
        
        self.stats['total_articles'] = len(titles)
        corpus = {
            'metadata': {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_articles': len(titles),
                'fragment_params': fragment_params
            },
            'articles': [],
            'fragments': []
        }
        
        logger.info(f"Iniciando construcción del corpus con {len(titles)} artículos...")
        
        for i, title in enumerate(titles, 1):
            logger.info(f"Procesando {i}/{len(titles)}: {title}")
            
            # Descargar artículo
            article_data = self.get_article_content(title)
            
            if article_data is None:
                self.stats['failed_downloads'] += 1
                continue
            
            # Limpiar contenido
            clean_content = self.clean_wikipedia_text(article_data['content'])
            
            if not clean_content:
                logger.warning(f"Contenido vacío después de limpieza: {title}")
                self.stats['failed_downloads'] += 1
                continue
            
            # Fragmentar texto
            fragments = self.smart_fragment_text(clean_content, **fragment_params)
            
            if not fragments:
                logger.warning(f"No se generaron fragmentos para: {title}")
                self.stats['failed_downloads'] += 1
                continue
            
            # Agregar al corpus
            article_info = {
                'title': title,
                'url': article_data.get('url', ''),
                'pageid': article_data.get('pageid'),
                'extract': article_data.get('extract', ''),
                'clean_content': clean_content,
                'num_fragments': len(fragments),
                'total_chars': len(clean_content)
            }
            
            corpus['articles'].append(article_info)
            
            # Agregar fragmentos con metadatos
            for j, fragment in enumerate(fragments):
                fragment_info = {
                    'fragment_id': len(corpus['fragments']),
                    'article_title': title,
                    'article_id': len(corpus['articles']) - 1,
                    'fragment_index': j,
                    'text': fragment,
                    'length': len(fragment)
                }
                corpus['fragments'].append(fragment_info)
            
            self.stats['successful_downloads'] += 1
            self.stats['total_fragments'] += len(fragments)
            self.stats['total_characters'] += len(clean_content)
            
            # Pausa entre descargas para no sobrecargar la API
            time.sleep(0.5)
        
        # Actualizar metadatos finales
        corpus['metadata']['successful_articles'] = self.stats['successful_downloads']
        corpus['metadata']['failed_articles'] = self.stats['failed_downloads']
        corpus['metadata']['total_fragments'] = self.stats['total_fragments']
        corpus['metadata']['total_characters'] = self.stats['total_characters']
        
        logger.info(f"Corpus construido exitosamente:")
        logger.info(f"  - Artículos exitosos: {self.stats['successful_downloads']}")
        logger.info(f"  - Artículos fallidos: {self.stats['failed_downloads']}")
        logger.info(f"  - Total fragmentos: {self.stats['total_fragments']}")
        logger.info(f"  - Total caracteres: {self.stats['total_characters']}")
        
        return corpus
    
    def save_corpus(self, corpus: Dict, base_filename: str = "corpus"):
        """
        Guarda el corpus en múltiples formatos
        
        Args:
            corpus: Diccionario con el corpus
            base_filename: Nombre base para los archivos
        """
        # Guardar corpus completo en JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        logger.info(f"Corpus guardado en: {json_path}")
        
        # Guardar fragmentos en CSV
        fragments_df = pd.DataFrame(corpus['fragments'])
        csv_path = self.output_dir / f"{base_filename}_fragments.csv"
        fragments_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Fragmentos guardados en: {csv_path}")
        
        # Guardar metadatos de artículos
        articles_df = pd.DataFrame(corpus['articles'])
        articles_path = self.output_dir / f"{base_filename}_articles.csv"
        articles_df.to_csv(articles_path, index=False, encoding='utf-8')
        logger.info(f"Artículos guardados en: {articles_path}")
        
        # Guardar estadísticas
        stats_path = self.output_dir / f"{base_filename}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': corpus['metadata'],
                'statistics': self.stats
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Estadísticas guardadas en: {stats_path}")
    
    def load_corpus(self, json_path: str) -> Dict:
        """
        Carga un corpus previamente guardado
        
        Args:
            json_path: Ruta al archivo JSON del corpus
            
        Returns:
            Diccionario con el corpus
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            logger.info(f"Corpus cargado desde: {json_path}")
            return corpus
        except Exception as e:
            logger.error(f"Error al cargar corpus: {e}")
            return {}
    
    def validate_corpus(self, corpus: Dict) -> bool:
        """
        Valida la integridad del corpus
        
        Args:
            corpus: Diccionario con el corpus
            
        Returns:
            True si el corpus es válido
        """
        try:
            # Validar estructura básica
            required_keys = ['metadata', 'articles', 'fragments']
            if not all(key in corpus for key in required_keys):
                logger.error("Estructura de corpus inválida")
                return False
            
            # Validar que hay artículos y fragmentos
            if not corpus['articles'] or not corpus['fragments']:
                logger.error("Corpus vacío")
                return False
            
            # Validar coherencia de fragmentos
            expected_fragments = sum(article['num_fragments'] for article in corpus['articles'])
            actual_fragments = len(corpus['fragments'])
            
            if expected_fragments != actual_fragments:
                logger.error(f"Inconsistencia en fragmentos: esperados {expected_fragments}, actuales {actual_fragments}")
                return False
            
            logger.info("Corpus validado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en validación del corpus: {e}")
            return False


def main():
    """
    Función principal para ejecutar la construcción del corpus
    """
    # Configuración
    INPUT_FILE = "data/wikipedia_pages.txt"  # Archivo provisto por la cátedra
    OUTPUT_DIR = "data"
    
    # Parámetros de fragmentación
    FRAGMENT_PARAMS = {
        'max_length': 300,
        'min_length': 50,
        'overlap': 50
    }
    
    # Crear constructor
    builder = WikipediaCorpusBuilder(output_dir=OUTPUT_DIR)
    
    # Cargar títulos
    titles = builder.load_titles_from_file(INPUT_FILE)
    
    if not titles:
        logger.error("No se pudieron cargar los títulos")
        return
    
    # Construir corpus
    corpus = builder.build_corpus(titles, FRAGMENT_PARAMS)
    
    # Validar corpus
    if not builder.validate_corpus(corpus):
        logger.error("Corpus inválido")
        return
    
    # Guardar corpus
    builder.save_corpus(corpus, "wikipedia_corpus")
    
    logger.info("¡Construcción del corpus completada exitosamente!")


if __name__ == "__main__":
    main()
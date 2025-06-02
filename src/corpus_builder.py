"""
Módulo para construcción del corpus desde Wikipedia
Utiliza la API de Wikipedia para obtener y procesar artículos
"""

import json
import csv
import re
import requests
import time
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path


class CorpusBuilder:
    """Constructor de corpus desde artículos de Wikipedia"""
    
    def __init__(self, fragment_size: int = 300, overlap: int = 50):
        """
        Inicializa el constructor de corpus
        
        Args:
            fragment_size: Tamaño aproximado de cada fragmento en caracteres
            overlap: Solapamiento entre fragmentos consecutivos
        """
        self.fragment_size = fragment_size
        self.overlap = overlap
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.content_url = "https://en.wikipedia.org/w/api.php"
        
    def load_titles(self, filepath: str) -> List[str]:
        """
        Carga los títulos de Wikipedia desde archivo
        
        Args:
            filepath: Ruta al archivo con los títulos
            
        Returns:
            Lista de títulos de Wikipedia
        """
        titles = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
            print(f"✓ Cargados {len(titles)} títulos desde {filepath}")
        except FileNotFoundError:
            print(f"✗ No se encontró el archivo {filepath}")
        except Exception as e:
            print(f"✗ Error al cargar títulos: {e}")
        
        return titles
    
    def get_wikipedia_content(self, title: str) -> Dict:
        """
        Obtiene el contenido completo de un artículo de Wikipedia
        
        Args:
            title: Título del artículo de Wikipedia
            
        Returns:
            Diccionario con título y contenido del artículo
        """
        try:
            # Parámetros para obtener el contenido completo
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = requests.get(self.content_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data['query']['pages']
            
            for page_id, page_data in pages.items():
                if page_id != '-1' and 'extract' in page_data:
                    content = page_data['extract']
                    cleaned_content = self._clean_text(content)
                    
                    return {
                        'title': title,
                        'content': cleaned_content,
                        'length': len(cleaned_content)
                    }
            
            print(f"⚠ No se encontró contenido para: {title}")
            return {'title': title, 'content': '', 'length': 0}
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error de red para {title}: {e}")
            return {'title': title, 'content': '', 'length': 0}
        except Exception as e:
            print(f"✗ Error procesando {title}: {e}")
            return {'title': title, 'content': '', 'length': 0}
    
    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando referencias y secciones no deseadas
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        # Eliminar secciones comunes no deseadas
        sections_to_remove = [
            r'== *See also *==.*?(?=== |\Z)',
            r'== *References *==.*?(?=== |\Z)',
            r'== *External links *==.*?(?=== |\Z)',
            r'== *Bibliography *==.*?(?=== |\Z)',
            r'== *Further reading *==.*?(?=== |\Z)',
            r'== *Notes *==.*?(?=== |\Z)'
        ]
        
        for pattern in sections_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Limpiar referencias entre corchetes
        text = re.sub(r'\[[\d\s,\-]+\]', '', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # Enlaces internos
        text = re.sub(r'\[([^\]]+)\]', '', text)  # Enlaces externos
        
        # Limpiar espacios múltiples y saltos de línea
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def create_fragments(self, text: str, title: str) -> List[Dict]:
        """
        Divide el texto en fragmentos manejables
        
        Args:
            text: Texto a fragmentar
            title: Título del artículo
            
        Returns:
            Lista de fragmentos con metadatos
        """
        if not text:
            return []
        
        fragments = []
        start = 0
        fragment_id = 0
        
        while start < len(text):
            end = start + self.fragment_size
            
            # Si no es el último fragmento, buscar un punto de corte natural
            if end < len(text):
                # Buscar el último punto, salto de línea o espacio
                for i in range(end, start + self.fragment_size // 2, -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
                    elif text[i] == ' ':
                        end = i
                        break
            
            fragment_text = text[start:end].strip()
            
            if fragment_text:  # Solo agregar fragmentos no vacíos
                fragments.append({
                    'id': f"{title}_{fragment_id}",
                    'title': title,
                    'fragment_id': fragment_id,
                    'text': fragment_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(fragment_text)
                })
                fragment_id += 1
            
            # Avanzar con solapamiento
            start = max(start + self.fragment_size - self.overlap, end)
        
        return fragments
    
    def build_corpus(self, titles: List[str], delay: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Construye el corpus completo desde la lista de títulos
        
        Args:
            titles: Lista de títulos de Wikipedia
            delay: Tiempo de espera entre requests (para ser respetuoso con la API)
            
        Returns:
            Tupla con (artículos, fragmentos)
        """
        articles = []
        all_fragments = []
        
        print(f"🔄 Construyendo corpus de {len(titles)} artículos...")
        
        for i, title in enumerate(titles, 1):
            print(f"📖 Procesando {i}/{len(titles)}: {title}")
            
            # Obtener contenido del artículo
            article_data = self.get_wikipedia_content(title)
            articles.append(article_data)
            
            # Crear fragmentos si hay contenido
            if article_data['content']:
                fragments = self.create_fragments(
                    article_data['content'], 
                    article_data['title']
                )
                all_fragments.extend(fragments)
                print(f"  ➜ {len(fragments)} fragmentos creados")
            else:
                print(f"  ⚠ Sin contenido disponible")
            
            # Pausa para ser respetuoso con la API
            time.sleep(delay)
        
        print(f"✅ Corpus construido: {len(articles)} artículos, {len(all_fragments)} fragmentos")
        return articles, all_fragments
    
    def save_corpus(self, articles: List[Dict], fragments: List[Dict], 
                   output_dir: str = "data/"):
        """
        Guarda el corpus en formatos JSON y CSV
        
        Args:
            articles: Lista de artículos
            fragments: Lista de fragmentos
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Guardar corpus completo en JSON
        corpus_data = {
            'metadata': {
                'total_articles': len(articles),
                'total_fragments': len(fragments),
                'fragment_size': self.fragment_size,
                'overlap': self.overlap
            },
            'articles': articles,
            'fragments': fragments
        }
        
        json_path = output_path / "corpus.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Corpus guardado en JSON: {json_path}")
        
        # Guardar fragmentos en CSV
        if fragments:
            df_fragments = pd.DataFrame(fragments)
            csv_path = output_path / "fragments.csv"
            df_fragments.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Fragmentos guardados en CSV: {csv_path}")
        
        # Guardar estadísticas
        stats = {
            'total_articles': len(articles),
            'successful_articles': len([a for a in articles if a['content']]),
            'total_fragments': len(fragments),
            'avg_article_length': sum(a['length'] for a in articles) / len(articles) if articles else 0,
            'avg_fragment_length': sum(f['length'] for f in fragments) / len(fragments) if fragments else 0
        }
        
        print("\n📊 Estadísticas del corpus:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Función principal para construir el corpus"""
    builder = CorpusBuilder(fragment_size=300, overlap=50)
    
    # Cargar títulos
    titles = builder.load_titles("data/wikipedia_pages.txt")
    
    if titles:
        # Construir corpus
        articles, fragments = builder.build_corpus(titles)
        
        # Guardar resultados
        builder.save_corpus(articles, fragments)
    else:
        print("❌ No se pudieron cargar los títulos")


if __name__ == "__main__":
    main()
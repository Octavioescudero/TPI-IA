"""
Módulo de generación de respuestas usando modelos generativos
Utiliza TinyLlama u otros modelos livianos para generar respuestas contextualizadas
"""

import json
import torch
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional
import time
import re


class ResponseGenerator:
    """Generador de respuestas usando modelos de lenguaje"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 max_length: int = 512, device: str = "auto"):
        """
        Inicializa el generador de respuestas
        
        Args:
            model_name: Nombre del modelo de Hugging Face
            max_length: Longitud máxima de respuesta
            device: Dispositivo a usar ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def _get_device(self, device: str) -> str:
        """Determina el dispositivo óptimo"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, use_quantization: bool = True) -> bool:
        """
        Carga el modelo y tokenizer
        
        Args:
            use_quantization: Si usar cuantización para reducir memoria
            
        Returns:
            True si se cargó correctamente
        """
        try:
            print(f"🔄 Cargando modelo {self.model_name}...")
            print(f"📱 Usando dispositivo: {self.device}")
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configurar pad token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configuración de cuantización para ahorrar memoria
            quantization_config = None
            if use_quantization and self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                except:
                    print("⚠ Cuantización no disponible, cargando modelo completo")
            
            # Cargar modelo
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not use_quantization:
                self.model = self.model.to(self.device)
            
            # Crear pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            print("✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def create_prompt(self, question: str, context_fragments: List[Dict], 
                     prompt_template: str = None) -> str:
        """
        Crea el prompt para el modelo basado en la pregunta y contexto
        
        Args:
            question: Pregunta del usuario
            context_fragments: Lista de fragmentos relevantes
            prompt_template: Template personalizado del prompt
            
        Returns:
            Prompt formateado
        """
        if prompt_template is None:
            prompt_template = self._get_default_prompt_template()
        
        # Crear contexto concatenando fragmentos
        context_parts = []
        for i, fragment in enumerate(context_fragments, 1):
            text = fragment.get('text', '')
            title = fragment.get('title', 'Documento')
            context_parts.append(f"Fragmento {i} ({title}):\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Formatear prompt
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        return prompt
    
    def _get_default_prompt_template(self) -> str:
        """Retorna el template por defecto del prompt"""
        return """<|system|>
Eres un asistente útil que responde preguntas basándose únicamente en el contexto proporcionado. 
Responde en español de manera clara y concisa. Si la información no está en el contexto, indica que no puedes responder con la información disponible.
</s>
<|user|>
Contexto:
{context}

Pregunta: {question}

Por favor, responde en español basándote únicamente en el contexto proporcionado.
</s>
<|assistant|>
"""
    
    def generate_response(self, question: str, context_fragments: List[Dict],
                         temperature: float = 0.7, max_new_tokens: int = 200,
                         do_sample: bool = True) -> Dict:
        """
        Genera una respuesta basada en la pregunta y contexto
        
        Args:
            question: Pregunta del usuario
            context_fragments: Lista de fragmentos relevantes
            temperature: Temperatura para la generación
            max_new_tokens: Máximo de tokens nuevos a generar
            do_sample: Si usar sampling en la generación
            
        Returns:
            Diccionario con la respuesta y metadatos
        """
        if not self.pipeline:
            print("❌ Modelo no cargado")
            return {"error": "Modelo no cargado"}
        
        try:
            # Crear prompt
            prompt = self.create_prompt(question, context_fragments)
            
            # Medir tiempo de generación
            start_time = time.time()
            
            # Generar respuesta
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generation_time = time.time() - start_time
            
            # Extraer respuesta
            raw_response = outputs[0]['generated_text']
            cleaned_response = self._clean_response(raw_response)
            
            return {
                "question": question,
                "response": cleaned_response,
                "raw_response": raw_response,
                "context_fragments": len(context_fragments),
                "generation_time": generation_time,
                "prompt_length": len(prompt),
                "response_length": len(cleaned_response)
            }
            
        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
            return {"error": str(e)}
    
    def _clean_response(self, response: str) -> str:
        """
        Limpia la respuesta generada
        
        Args:
            response: Respuesta cruda del modelo
            
        Returns:
            Respuesta limpia
        """
        # Remover tokens especiales comunes
        response = response.replace("<|assistant|>", "")
        response = response.replace("</s>", "")
        response = response.replace("<|system|>", "")
        response = response.replace("<|user|>", "")
        
        # Limpiar espacios y saltos de línea excesivos
        response = re.sub(r'\n+', '\n', response)
        response = re.sub(r' +', ' ', response)
        response = response.strip()
        
        # Si la respuesta está vacía o muy corta, proporcionar respuesta por defecto
        if len(response) < 10:
            response = "No puedo proporcionar una respuesta adecuada con la información disponible."
        
        return response
    
    def batch_generate(self, questions_contexts: List[Tuple[str, List[Dict]]],
                      **generation_kwargs) -> List[Dict]:
        """
        Genera respuestas para múltiples preguntas
        
        Args:
            questions_contexts: Lista de tuplas (pregunta, fragmentos_contexto)
            **generation_kwargs: Argumentos adicionales para la generación
            
        Returns:
            Lista de respuestas generadas
        """
        responses = []
        
        print(f"🔄 Generando {len(questions_contexts)} respuestas...")
        
        for i, (question, context_fragments) in enumerate(questions_contexts, 1):
            print(f"  Procesando {i}/{len(questions_contexts)}: {question[:50]}...")
            
            response = self.generate_response(question, context_fragments, **generation_kwargs)
            responses.append(response)
        
        print("✅ Generación completada")
        return responses
    
    def evaluate_response_quality(self, response_data: Dict) -> Dict:
        """
        Evalúa la calidad de una respuesta generada
        
        Args:
            response_data: Datos de la respuesta generada
            
        Returns:
            Métricas de calidad
        """
        response = response_data.get("response", "")
        
        metrics = {
            "length": len(response),
            "word_count": len(response.split()),
            "sentence_count": len([s for s in response.split('.') if s.strip()]),
            "contains_context_info": self._check_context_usage(response),
            "is_spanish": self._check_spanish_language(response),
            "completeness_score": self._calculate_completeness(response)
        }
        
        return metrics
    
    def _check_context_usage(self, response: str) -> bool:
        """Verifica si la respuesta usa información del contexto"""
        # Indicadores de que se está usando contexto
        context_indicators = [
            "según", "de acuerdo", "basado en", "el texto menciona",
            "se indica que", "según la información", "el contexto"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in context_indicators)
    
    def _check_spanish_language(self, response: str) -> bool:
        """Verifica si la respuesta está en español"""
        spanish_indicators = [
            "el", "la", "los", "las", "un", "una", "que", "de", "en", "con",
            "por", "para", "es", "son", "está", "están", "tiene", "tienen"
        ]
        
        words = response.lower().split()
        spanish_word_count = sum(1 for word in words if word in spanish_indicators)
        
        return spanish_word_count > len(words) * 0.1  # Al menos 10% de palabras en español
    
    def _calculate_completeness(self, response: str) -> float:
        """Calcula un score de completitud de la respuesta"""
        if len(response) < 20:
            return 0.0
        
        # Factores que indican completitud
        factors = []
        
        # Longitud adecuada
        factors.append(min(len(response) / 100, 1.0))
        
        # Presencia de puntuación
        punct_ratio = sum(1 for c in response if c in '.,;:!?') / len(response)
        factors.append(min(punct_ratio * 20, 1.0))
        
        # No repetición excesiva
        words = response.split()
        unique_words = len(set(words))
        diversity = unique_words / len(words) if words else 0
        factors.append(diversity)
        
        return sum(factors) / len(factors)


class MultiModelResponseGenerator:
    """Generador que puede usar múltiples modelos"""
    
    def __init__(self):
        self.generators = {}
        self.active_generator = None
    
    def add_generator(self, name: str, generator: ResponseGenerator):
        """Agrega un generador al conjunto"""
        self.generators[name] = generator
        if self.active_generator is None:
            self.active_generator = name
    
    def set_active_generator(self, name: str):
        """Establece el generador activo"""
        if name in self.generators:
            self.active_generator = name
        else:
            print(f"❌ Generador '{name}' no encontrado")
    
    def compare_models(self, question: str, context_fragments: List[Dict]) -> Dict:
        """Compara respuestas de diferentes modelos"""
        results = {}
        
        for name, generator in self.generators.items():
            print(f"🔄 Generando con {name}...")
            response = generator.generate_response(question, context_fragments)
            
            if "error" not in response:
                quality_metrics = generator.evaluate_response_quality(response)
                response["quality_metrics"] = quality_metrics
            
            results[name] = response
        
        return results


def main():
    """Función principal para probar el generador de respuestas"""
    # Crear generador
    generator = ResponseGenerator()
    
    # Cargar modelo
    if not generator.load_model():
        print("❌ No se pudo cargar el modelo")
        return
    
    # Contexto de ejemplo
    sample_fragments = [
        {
            "id": "test_1",
            "title": "Machine Learning",
            "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
        },
        {
            "id": "test_2", 
            "title": "Neural Networks",
            "text": "Neural networks are computing systems inspired by biological neural networks. They consist of nodes called neurons that are connected and can transmit signals to each other."
        }
    ]
    
    # Pregunta de ejemplo
    question = "¿Qué es el aprendizaje automático?"
    
    # Generar respuesta
    print(f"🔍 Pregunta: {question}")
    response = generator.generate_response(question, sample_fragments)
    
    if "error" not in response:
        print(f"\n✅ Respuesta generada:")
        print(f"📝 {response['response']}")
        print(f"\n📊 Métricas:")
        print(f"  Tiempo de generación: {response['generation_time']:.2f}s")
        print(f"  Fragmentos usados: {response['context_fragments']}")
        print(f"  Longitud de respuesta: {response['response_length']} caracteres")
        
        # Evaluar calidad
        quality = generator.evaluate_response_quality(response)
        print(f"\n🎯 Calidad:")
        for metric, value in quality.items():
            print(f"  {metric}: {value}")
    else:
        print(f"❌ Error: {response['error']}")


if __name__ == "__main__":
    main()
import os
from openai import OpenAI
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Analisador de evidências usando GPT-4"""
    
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze_evidence(self, evidence_results):
        """
        Analisa os resultados das buscas multimodais e gera um relatório
        
        Args:
            evidence_results: Dict com resultados por modalidade
            {
                'vision': [...],
                'audio': [...],
                'text': [...],
                'depth': [...]
            }
        """
        # Formata as evidências para o prompt
        evidence_summary = self._format_evidence(evidence_results)
        
        prompt = f"""Você é um detetive forense especializado em analisar evidências multimodais no caso do crime em Gotham City.

EVIDÊNCIAS COLETADAS:
{evidence_summary}

Por favor, analise as evidências acima e forneça:

1. PADRÕES IDENTIFICADOS:
- Conexões entre diferentes tipos de evidência
- Padrões temporais ou espaciais relevantes
- Características distintivas do suspeito

2. SUSPEITO PROVÁVEL:
- Identidade do principal suspeito
- Nível de confiança na identificação (0-100%)
- Justificativa para a identificação

3. PRÓXIMOS PASSOS:
- Recomendações para a investigação
- Evidências adicionais necessárias
- Áreas que precisam de mais investigação

Formato o relatório de forma clara e profissional."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um detetive forense especializado em análise de evidências multimodais."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            report = response.choices[0].message.content
            logger.info("\n📋 Relatório Forense Gerado:")
            logger.info("=" * 50)
            logger.info(report)
            logger.info("=" * 50)
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            return None
    
    def _format_evidence(self, evidence_results):
        """Formata as evidências para o prompt"""
        formatted = []
        
        for modality, results in evidence_results.items():
            formatted.append(f"\n{modality.upper()}:")
            for i, result in enumerate(results, 1):
                description = result.get('description', 'Sem descrição')
                similarity = result.get('score', 0)
                formatted.append(f"{i}. {description} (Similaridade: {similarity:.2f})")
        
        return "\n".join(formatted)

    def analyze_cross_modal_connections(self, results_a, modality_a, results_b, modality_b):
        """Analisa conexões específicas entre duas modalidades diferentes"""
        prompt = f"""Analise a relação entre as seguintes evidências de modalidades diferentes:

{modality_a.upper()}:
{self._format_evidence({modality_a: results_a})}

{modality_b.upper()}:
{self._format_evidence({modality_b: results_b})}

Por favor, identifique:
1. Conexões diretas entre as evidências
2. Padrões que sugerem o mesmo suspeito
3. Inconsistências ou contradições
4. Força da correlação (0-100%)"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um especialista em análise forense de evidências multimodais."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            logger.info(f"\n🔍 Análise Cross-Modal ({modality_a} x {modality_b}):")
            logger.info(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise cross-modal: {str(e)}")
            return None
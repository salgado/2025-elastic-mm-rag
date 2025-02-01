import logging
from pathlib import Path
import json
from llm_analyzer import LLMAnalyzer
from elastic_manager import ElasticManager
from embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMAnalyzerWithRealData:
    def __init__(self):
        self.llm = LLMAnalyzer()
        self.elastic = ElasticManager()
        self.embedding_generator = EmbeddingGenerator()
        
    def get_real_data(self):
        """Busca dados reais do Elasticsearch para cada modalidade"""
        evidence_data = {}
        
        # Lista de arquivos de teste para cada modalidade
        test_files = {
            'vision': 'data/images/crime_scene1.jpg',
            'audio': 'data/audios/joker_laugh.wav',
            'text': 'Why so serious?',
            'depth': 'data/depths/suspect_depth.jpg'
        }
        
        for modality, test_input in test_files.items():
            try:
                # Gera embedding para busca
                if modality == 'text':
                    embedding = self.embedding_generator.generate_embedding([test_input], modality)
                else:
                    embedding = self.embedding_generator.generate_embedding([str(test_input)], modality)
                
                # Busca resultados similares
                results = self.elastic.search_similar(embedding, k=5)
                if results:
                    evidence_data[modality] = results
                    logger.info(f"✅ Dados recuperados para {modality}: {len(results)} resultados")
                else:
                    logger.warning(f"⚠️ Nenhum resultado encontrado para {modality}")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao buscar dados para {modality}: {str(e)}")
        
        return evidence_data
    
    def test_real_data_analysis(self):
        """Testa análise com dados reais do Elasticsearch"""
        logger.info("\n🔍 Buscando dados reais do Elasticsearch...")
        evidence_data = self.get_real_data()
        
        if not evidence_data:
            logger.error("❌ Nenhum dado encontrado no Elasticsearch!")
            return
        
        # Gera relatório forense
        logger.info("\n📋 Gerando relatório forense com dados reais...")
        report = self.llm.analyze_evidence(evidence_data)
        
        # Análise cross-modal específica
        if 'audio' in evidence_data and 'vision' in evidence_data:
            logger.info("\n🔄 Realizando análise cross-modal (áudio x imagem)...")
            cross_modal = self.llm.analyze_cross_modal_connections(
                evidence_data['audio'],
                'audio',
                evidence_data['vision'],
                'vision'
            )
        else:
            cross_modal = None
            logger.warning("⚠️ Dados insuficientes para análise cross-modal")
        
        # Salva resultados
        results = {
            'evidence_data': evidence_data,
            'forensic_report': report,
            'cross_modal_analysis': cross_modal
        }
        
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'llm_analysis_real_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Resultados salvos em: {output_file}")
        
        # Mostra um resumo dos resultados
        if report:
            logger.info("\n📊 Resumo do Relatório Forense:")
            logger.info("=" * 50)
            logger.info(report)
            logger.info("=" * 50)

def main():
    logger.info("🚀 Iniciando teste do LLM Analyzer com dados reais...")
    
    tester = TestLLMAnalyzerWithRealData()
    tester.test_real_data_analysis()
    
    logger.info("\n✨ Teste concluído!")

if __name__ == "__main__":
    main()
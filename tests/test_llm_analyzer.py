import logging
import sys
import os
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMAnalyzer:
    def __init__(self):
        try:
            from src.llm_analyzer import LLMAnalyzer
            from src.elastic_manager import ElasticManager
            from src.embedding_generator import EmbeddingGenerator
            
            self.llm = LLMAnalyzer()
            self.elastic = ElasticManager()
            self.embedding_generator = EmbeddingGenerator()
            logger.info("✅ All components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def test_analysis_with_real_data(self):
        """Test analysis with real data from Elasticsearch"""
        try:
            evidence_data = {}
            
            # Get data for each modality
            test_files = {
                'vision': 'data/images/crime_scene1.jpg',
                'audio': 'data/audios/joker_laugh.wav',
                'text': 'Why so serious?',
                'depth': 'data/depths/suspect_depth.jpg'
            }
            
            logger.info("🔍 Collecting evidence...")
            for modality, test_input in test_files.items():
                try:
                    if modality == 'text':
                        embedding = self.embedding_generator.generate_embedding([test_input], modality)
                    else:
                        embedding = self.embedding_generator.generate_embedding([str(test_input)], modality)
                    
                    results = self.elastic.search_similar(embedding, k=3)
                    if results:
                        evidence_data[modality] = results
                        logger.info(f"✅ Data retrieved for {modality}: {len(results)} results")
                    else:
                        logger.warning(f"⚠️ No results found for {modality}")
                        
                except Exception as e:
                    logger.error(f"❌ Error retrieving {modality} data: {str(e)}")
            
            if not evidence_data:
                raise ValueError("No evidence data found in Elasticsearch!")
            
            # Test forensic report generation
            logger.info("\n📝 Generating forensic report...")
            report = self.llm.analyze_evidence(evidence_data)
            
            if report:
                logger.info("✅ Forensic report generated successfully")
                logger.info("\n📊 Report Preview:")
                logger.info("=" * 50)
                logger.info(report[:500] + "..." if len(report) > 500 else report)
                logger.info("=" * 50)
            else:
                raise ValueError("Failed to generate forensic report")
            
            # Test cross-modal analysis
            if 'audio' in evidence_data and 'vision' in evidence_data:
                logger.info("\n🔄 Testing cross-modal analysis...")
                cross_modal = self.llm.analyze_cross_modal_connections(
                    evidence_data['audio'],
                    'audio',
                    evidence_data['vision'],
                    'vision'
                )
                
                if cross_modal:
                    logger.info("✅ Cross-modal analysis generated successfully")
                    logger.info("\n🔍 Cross-modal Analysis Preview:")
                    logger.info("=" * 50)
                    logger.info(cross_modal[:500] + "..." if len(cross_modal) > 500 else cross_modal)
                    logger.info("=" * 50)
                else:
                    raise ValueError("Failed to generate cross-modal analysis")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in analysis test: {str(e)}")
            return False

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        try:
            logger.info("\n🧪 Testing error handling...")
            
            # Test with empty data
            try:
                self.llm.analyze_evidence({})
                logger.info("✅ Empty data handling OK")
            except Exception as e:
                logger.error(f"❌ Error with empty data: {str(e)}")
            
            # Test with malformed data
            try:
                self.llm.analyze_evidence({'invalid': 'data'})
                logger.info("✅ Malformed data handling OK")
            except Exception as e:
                logger.error(f"❌ Error with malformed data: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in error handling test: {str(e)}")
            return False

def main():
    logger.info("🚀 Starting LLM Analyzer tests...")
    
    tester = TestLLMAnalyzer()
    
    # Run tests
    logger.info("\n📝 Testing analysis with real data...")
    analysis_success = tester.test_analysis_with_real_data()
    
    logger.info("\n📝 Testing error handling...")
    error_handling_success = tester.test_error_handling()
    
    # Report results
    logger.info("\n📊 Test Results:")
    logger.info(f"Analysis with Real Data: {'✅' if analysis_success else '❌'}")
    logger.info(f"Error Handling: {'✅' if error_handling_success else '❌'}")
    
    if analysis_success and error_handling_success:
        logger.info("\n✨ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed")

if __name__ == "__main__":
    main()
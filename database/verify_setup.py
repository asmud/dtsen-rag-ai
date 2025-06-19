#!/usr/bin/env python3
"""
DTSEN RAG AI - Database Setup Verification Script

This script verifies that the PostgreSQL database is properly configured
for the DTSEN RAG AI system with pgvector extension and correct schema.

Usage:
    python verify_setup.py [--connection-string CONNECTION_STRING]
    
Example:
    python verify_setup.py --connection-string "postgresql://rag_user:rag_pass@localhost:5432/rag_db"
"""

import asyncio
import argparse
import logging
import sys
from typing import Dict, Any, List
import asyncpg
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseVerifier:
    """Verify database setup for DTSEN RAG AI system"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        
    async def connect(self):
        """Connect to the database"""
        try:
            self.connection = await asyncpg.connect(self.connection_string)
            logger.info("✓ Database connection established")
            return True
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the database"""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def verify_pgvector_extension(self) -> Dict[str, Any]:
        """Verify pgvector extension is installed"""
        try:
            result = await self.connection.fetchrow(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as installed, "
                "COALESCE((SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'not_installed') as version"
            )
            
            if result['installed']:
                logger.info(f"✓ pgvector extension installed (version: {result['version']})")
                return {'status': 'ok', 'version': result['version']}
            else:
                logger.error("✗ pgvector extension not installed")
                return {'status': 'missing', 'version': None}
                
        except Exception as e:
            logger.error(f"✗ Error checking pgvector extension: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def verify_main_table(self) -> Dict[str, Any]:
        """Verify main vector table exists with correct schema"""
        try:
            # Check if table exists
            table_exists = await self.connection.fetchrow(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'data_rag_kb') as exists"
            )
            
            if not table_exists['exists']:
                logger.error("✗ Main table 'data_rag_kb' does not exist")
                return {'status': 'missing', 'details': 'Table not found'}
            
            # Check table schema
            columns = await self.connection.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'data_rag_kb'
                ORDER BY ordinal_position
            """)
            
            expected_columns = {'id', 'text', 'metadata', 'embedding', 'created_at', 'updated_at'}
            actual_columns = {col['column_name'] for col in columns}
            
            if expected_columns.issubset(actual_columns):
                logger.info("✓ Main table 'data_rag_kb' exists with correct schema")
                
                # Check vector dimension if table has data
                vector_info = await self.connection.fetchrow(
                    "SELECT COUNT(*) as row_count, "
                    "CASE WHEN COUNT(*) > 0 THEN vector_dims(embedding) ELSE 0 END as dimension "
                    "FROM data_rag_kb LIMIT 1"
                )
                
                return {
                    'status': 'ok',
                    'columns': [dict(col) for col in columns],
                    'row_count': vector_info['row_count'],
                    'vector_dimension': vector_info['dimension'] if vector_info['row_count'] > 0 else None
                }
            else:
                missing_cols = expected_columns - actual_columns
                logger.error(f"✗ Main table missing required columns: {missing_cols}")
                return {'status': 'invalid_schema', 'missing_columns': list(missing_cols)}
                
        except Exception as e:
            logger.error(f"✗ Error checking main table: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def verify_indexes(self) -> Dict[str, Any]:
        """Verify required indexes exist"""
        try:
            indexes = await self.connection.fetch("""
                SELECT indexname, indexdef
                FROM pg_indexes 
                WHERE tablename = 'data_rag_kb'
                ORDER BY indexname
            """)
            
            index_names = [idx['indexname'] for idx in indexes]
            
            # Check for critical indexes
            required_indexes = [
                'data_rag_kb_embedding_cosine_idx',
                'data_rag_kb_metadata_idx'
            ]
            
            missing_indexes = [idx for idx in required_indexes if not any(req in name for name in index_names for req in [idx])]
            
            if not missing_indexes:
                logger.info(f"✓ All required indexes present ({len(index_names)} total)")
            else:
                logger.warning(f"⚠ Missing recommended indexes: {missing_indexes}")
            
            return {
                'status': 'ok' if not missing_indexes else 'incomplete',
                'indexes': [dict(idx) for idx in indexes],
                'missing_indexes': missing_indexes
            }
            
        except Exception as e:
            logger.error(f"✗ Error checking indexes: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def verify_functions_and_views(self) -> Dict[str, Any]:
        """Verify utility functions and views exist"""
        try:
            # Check for health check function
            functions = await self.connection.fetch("""
                SELECT proname, prosrc IS NOT NULL as has_body
                FROM pg_proc 
                WHERE proname IN ('check_vector_store_health', 'update_updated_at_column')
            """)
            
            # Check for stats view
            views = await self.connection.fetch("""
                SELECT viewname
                FROM pg_views 
                WHERE viewname = 'vector_store_stats'
            """)
            
            function_names = [func['proname'] for func in functions]
            view_names = [view['viewname'] for view in views]
            
            if 'check_vector_store_health' in function_names:
                logger.info("✓ Health check function exists")
            else:
                logger.warning("⚠ Health check function missing")
            
            if 'vector_store_stats' in view_names:
                logger.info("✓ Statistics view exists")
            else:
                logger.warning("⚠ Statistics view missing")
            
            return {
                'status': 'ok',
                'functions': function_names,
                'views': view_names
            }
            
        except Exception as e:
            logger.error(f"✗ Error checking functions/views: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_vector_operations(self) -> Dict[str, Any]:
        """Test basic vector operations"""
        try:
            # Test vector creation and similarity
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"
            
            # Test vector casting
            vector_test = await self.connection.fetchrow(
                f"SELECT '{test_vector}'::vector as test_vector, vector_dims('{test_vector}'::vector) as dims"
            )
            
            if vector_test['dims'] == 384:
                logger.info("✓ Vector operations working correctly")
                
                # Test similarity operators if table has data
                row_count = await self.connection.fetchval("SELECT COUNT(*) FROM data_rag_kb")
                
                if row_count > 0:
                    similarity_test = await self.connection.fetchrow(f"""
                        SELECT 
                            embedding <=> '{test_vector}'::vector as cosine_distance,
                            embedding <-> '{test_vector}'::vector as l2_distance
                        FROM data_rag_kb 
                        LIMIT 1
                    """)
                    
                    logger.info("✓ Similarity operators working correctly")
                    return {
                        'status': 'ok',
                        'vector_dimension': 384,
                        'similarity_test': dict(similarity_test) if similarity_test else None
                    }
                else:
                    logger.info("✓ Vector operations available (no data to test similarity)")
                    return {'status': 'ok', 'vector_dimension': 384, 'note': 'no_data_for_similarity_test'}
            else:
                logger.error(f"✗ Vector dimension incorrect: {vector_test['dims']} (expected 384)")
                return {'status': 'dimension_error', 'actual_dimension': vector_test['dims']}
                
        except Exception as e:
            logger.error(f"✗ Error testing vector operations: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run the database health check function if available"""
        try:
            health_results = await self.connection.fetch("SELECT * FROM check_vector_store_health()")
            
            logger.info("✓ Database health check completed")
            
            results = {}
            for row in health_results:
                results[row['component']] = {
                    'status': row['status'],
                    'message': row['message'],
                    'details': row['details']
                }
            
            return {'status': 'ok', 'health_check': results}
            
        except Exception as e:
            logger.warning(f"⚠ Health check function unavailable: {e}")
            return {'status': 'unavailable', 'error': str(e)}
    
    async def verify_configuration(self) -> Dict[str, Any]:
        """Verify system configuration table"""
        try:
            # Check if system info table exists
            table_exists = await self.connection.fetchrow(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'rag_system_info') as exists"
            )
            
            if not table_exists['exists']:
                logger.warning("⚠ System configuration table 'rag_system_info' not found")
                return {'status': 'missing', 'note': 'Optional table not found'}
            
            # Get configuration values
            config = await self.connection.fetch("SELECT key, value FROM rag_system_info")
            
            config_dict = {row['key']: row['value'] for row in config}
            
            # Verify expected configuration
            expected_config = {
                'vector_dimension': 384,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'collection_name': 'data_rag_kb'
            }
            
            issues = []
            for key, expected_value in expected_config.items():
                if key in config_dict:
                    actual_value = config_dict[key]
                    if str(actual_value).strip('"') != str(expected_value):
                        issues.append(f"{key}: expected {expected_value}, got {actual_value}")
                else:
                    issues.append(f"{key}: missing from configuration")
            
            if not issues:
                logger.info("✓ System configuration correct")
            else:
                logger.warning(f"⚠ Configuration issues: {issues}")
            
            return {
                'status': 'ok' if not issues else 'issues',
                'configuration': config_dict,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"✗ Error checking configuration: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def run_full_verification(self) -> Dict[str, Any]:
        """Run complete database verification"""
        logger.info("Starting DTSEN RAG AI database verification...")
        
        if not await self.connect():
            return {'status': 'connection_failed'}
        
        try:
            results = {
                'connection': {'status': 'ok'},
                'pgvector_extension': await self.verify_pgvector_extension(),
                'main_table': await self.verify_main_table(),
                'indexes': await self.verify_indexes(),
                'functions_views': await self.verify_functions_and_views(),
                'vector_operations': await self.test_vector_operations(),
                'health_check': await self.run_health_check(),
                'configuration': await self.verify_configuration()
            }
            
            # Calculate overall status
            critical_components = ['connection', 'pgvector_extension', 'main_table', 'vector_operations']
            critical_status = [results[comp]['status'] for comp in critical_components]
            
            if all(status == 'ok' for status in critical_status):
                overall_status = 'ready'
                logger.info("🎉 Database verification PASSED - System ready for DTSEN RAG AI")
            elif any(status in ['error', 'missing'] for status in critical_status):
                overall_status = 'failed'
                logger.error("❌ Database verification FAILED - Critical issues found")
            else:
                overall_status = 'partial'
                logger.warning("⚠️ Database verification PARTIAL - Some optional components missing")
            
            results['overall_status'] = overall_status
            
            return results
            
        finally:
            await self.disconnect()

def print_verification_report(results: Dict[str, Any]):
    """Print a formatted verification report"""
    print("\n" + "="*60)
    print("DTSEN RAG AI DATABASE VERIFICATION REPORT")
    print("="*60)
    
    status_icons = {
        'ok': '✓',
        'ready': '🎉',
        'failed': '❌',
        'partial': '⚠️',
        'missing': '✗',
        'error': '❌',
        'incomplete': '⚠️',
        'unavailable': '⚠️'
    }
    
    overall_status = results.get('overall_status', 'unknown')
    print(f"\nOverall Status: {status_icons.get(overall_status, '?')} {overall_status.upper()}")
    
    print(f"\nComponent Status:")
    for component, data in results.items():
        if component == 'overall_status':
            continue
            
        status = data.get('status', 'unknown')
        icon = status_icons.get(status, '?')
        
        print(f"  {icon} {component.replace('_', ' ').title()}: {status}")
        
        # Show additional details for some components
        if component == 'main_table' and 'row_count' in data:
            print(f"    - Rows: {data['row_count']}")
            if data.get('vector_dimension'):
                print(f"    - Vector Dimension: {data['vector_dimension']}")
        
        if component == 'indexes' and 'indexes' in data:
            print(f"    - Indexes: {len(data['indexes'])}")
        
        if component == 'configuration' and 'issues' in data and data['issues']:
            for issue in data['issues']:
                print(f"    - Issue: {issue}")
    
    print(f"\n{'='*60}")
    
    if overall_status == 'ready':
        print("✅ Database is ready for DTSEN RAG AI!")
        print("   You can now start the application.")
    elif overall_status == 'failed':
        print("❌ Database setup has critical issues.")
        print("   Please run init.sql to set up the schema.")
    else:
        print("⚠️  Database is partially configured.")
        print("   Consider running init.sql for complete setup.")

async def main():
    """Main verification function"""
    parser = argparse.ArgumentParser(
        description="Verify DTSEN RAG AI database setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_setup.py
  python verify_setup.py --connection-string "postgresql://rag_user:rag_pass@localhost:5432/rag_db"
  python verify_setup.py --json-output
        """
    )
    parser.add_argument(
        '--connection-string',
        default="postgresql://rag_user:rag_pass@localhost:5432/rag_db",
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        '--json-output',
        action='store_true',
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    verifier = DatabaseVerifier(args.connection_string)
    
    try:
        results = await verifier.run_full_verification()
        
        if args.json_output:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_verification_report(results)
        
        # Exit with appropriate code
        if results.get('overall_status') == 'ready':
            sys.exit(0)
        elif results.get('overall_status') == 'failed':
            sys.exit(1)
        else:
            sys.exit(2)  # Partial success
            
    except KeyboardInterrupt:
        logger.info("Verification cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Verification failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
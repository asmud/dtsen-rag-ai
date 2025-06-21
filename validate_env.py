#!/usr/bin/env python3
"""
DTSEN RAG AI - Environment Configuration Validation Script

This script validates environment configuration across all files:
- .env files
- docker-compose.yml
- config.py defaults

Usage:
    python validate_env.py
    python validate_env.py --env-file .env.apple-silicon
    python validate_env.py --check-all
"""

import argparse
import os
import sys
import yaml
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

class EnvironmentValidator:
    """Validates environment configuration consistency"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config_py_path = self.project_root / "app" / "config.py"
        self.docker_compose_path = self.project_root / "docker-compose.yml"
        # Profile-specific environment files
        self.env_apple_path = self.project_root / ".env.apple-silicon"
        self.env_nvidia_path = self.project_root / ".env.nvidia-gpu"
        self.env_cpu_path = self.project_root / ".env.cpu-only"
        
        # Critical environment variables that must be consistent
        # Note: URLs are now auto-generated, so we focus on components
        self.critical_vars = {
            "COLLECTION_NAME", 
            "VECTOR_DIMENSION",
            "LLM_MODEL",
            "EMBEDDING_MODEL"
        }
        
        # Variables that are allowed to differ between profiles
        self.profile_specific_vars = {
            "OLLAMA_API",  # Different for each Ollama service
            "GPU_ENABLED",
            "EMBEDDING_DEVICE",
            "EMBEDDING_BATCH_SIZE",
            "API_WORKERS",
            "MAX_CONCURRENT_REQUESTS"
        }
        
        # Load expected values from config instead of hard-coding
        try:
            import sys
            sys.path.append(str(self.project_root / "app"))
            from config import get_settings
            settings = get_settings()
            
            self.expected_values = {
                "COLLECTION_NAME": settings.COLLECTION_NAME,
                "VECTOR_DIMENSION": str(settings.VECTOR_DIMENSION),
                # URLs are auto-generated, so we check components instead
                "POSTGRES_PASSWORD": settings.POSTGRES_PASSWORD,
                "EMBEDDING_MODEL": settings.EMBEDDING_MODEL
            }
        except ImportError:
            # Fallback to basic validation if config can't be loaded
            self.expected_values = {
                "COLLECTION_NAME": "data_rag_kb",
                "VECTOR_DIMENSION": "384"
            }
    
    def parse_env_file(self, env_file_path: Path) -> Dict[str, str]:
        """Parse environment file and return key-value pairs"""
        env_vars = {}
        
        if not env_file_path.exists():
            return env_vars
            
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        return env_vars
    
    def parse_docker_compose(self) -> Dict[str, Dict[str, str]]:
        """Parse docker-compose.yml and extract environment variables"""
        if not self.docker_compose_path.exists():
            return {}
            
        with open(self.docker_compose_path, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        services_env = {}
        
        for service_name, service_config in compose_data.get('services', {}).items():
            if 'environment' in service_config:
                env_vars = {}
                for env_var in service_config['environment']:
                    if '=' in env_var:
                        key, value = env_var.split('=', 1)
                        env_vars[key] = value
                services_env[service_name] = env_vars
        
        return services_env
    
    def parse_config_py_defaults(self) -> Dict[str, str]:
        """Extract default values from config.py"""
        defaults = {}
        
        if not self.config_py_path.exists():
            return defaults
            
        with open(self.config_py_path, 'r') as f:
            content = f.read()
        
        # Simple regex-like parsing for defaults
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and '=' in line and not line.startswith('#'):
                # Look for patterns like: VARIABLE_NAME: type = "value"
                parts = line.split('=', 1)
                if len(parts) == 2:
                    left_part = parts[0].strip()
                    right_part = parts[1].strip().strip('"').strip("'")
                    
                    if ':' in left_part:
                        var_name = left_part.split(':')[0].strip()
                        # Clean up comments from right part
                        cleaned_value = right_part.split('#')[0].strip()
                        # Remove quotes if present
                        cleaned_value = cleaned_value.strip('"').strip("'")
                        defaults[var_name] = cleaned_value
        
        return defaults
    
    def validate_consistency(self) -> List[str]:
        """Validate consistency across all configuration sources"""
        issues = []
        
        # Parse all sources
        env_apple = self.parse_env_file(self.env_apple_path)
        env_nvidia = self.parse_env_file(self.env_nvidia_path)
        env_cpu = self.parse_env_file(self.env_cpu_path)
        config_defaults = self.parse_config_py_defaults()
        docker_services = self.parse_docker_compose()
        
        # Check critical variables consistency
        for var in self.critical_vars:
            values = {}
            
            # Check all profile files
            if var in env_apple:
                values['apple-silicon'] = env_apple[var]
            if var in env_nvidia:
                values['nvidia-gpu'] = env_nvidia[var]
            if var in env_cpu:
                values['cpu-only'] = env_cpu[var]
            if var in config_defaults:
                values['config.py'] = config_defaults[var]
            
            # Check docker-compose services
            for service, env_vars in docker_services.items():
                if var in env_vars:
                    values[f'docker-compose({service})'] = env_vars[var]
            
            # Skip validation if config.py has None (auto-generated values)
            if 'config.py' in values and values['config.py'] in ['None', None]:
                values.pop('config.py')
            
            # Check for inconsistencies
            unique_values = set(values.values())
            if len(unique_values) > 1:
                issues.append(f"❌ Inconsistent values for {var}: {values}")
            elif len(unique_values) == 1:
                print(f"✅ {var}: consistent across all sources")
            elif len(unique_values) == 0:
                print(f"✅ {var}: auto-generated (no conflicts)")
        
        # Check expected values in all profile files
        profile_files = [
            (env_apple, "apple-silicon"),
            (env_nvidia, "nvidia-gpu"), 
            (env_cpu, "cpu-only")
        ]
        
        for var, expected in self.expected_values.items():
            for env_data, profile_name in profile_files:
                if var in env_data and env_data[var] != expected:
                    issues.append(f"❌ {var} in .env.{profile_name} is '{env_data[var]}', expected '{expected}'")
            if var in config_defaults and config_defaults[var] != expected:
                issues.append(f"❌ {var} in config.py is '{config_defaults[var]}', expected '{expected}'")
        
        return issues
    
    def validate_env_file(self, env_file_path: Path) -> List[str]:
        """Validate a specific environment file"""
        issues = []
        
        if not env_file_path.exists():
            issues.append(f"❌ Environment file not found: {env_file_path}")
            return issues
        
        env_vars = self.parse_env_file(env_file_path)
        
        # Check for essential variables (URLs are auto-generated)
        required_vars = {
            "POSTGRES_PASSWORD", "COLLECTION_NAME", "VECTOR_DIMENSION",
            "LLM_MODEL", "EMBEDDING_MODEL"
        }
        
        missing_vars = required_vars - set(env_vars.keys())
        if missing_vars:
            issues.append(f"❌ Missing required variables in {env_file_path.name}: {missing_vars}")
        
        # Check expected values
        for var, expected in self.expected_values.items():
            if var in env_vars and env_vars[var] != expected:
                issues.append(f"❌ {var} in {env_file_path.name} is '{env_vars[var]}', expected '{expected}'")
        
        if not issues:
            print(f"✅ {env_file_path.name}: All validations passed")
        
        return issues
    
    def validate_profile_files(self) -> List[str]:
        """Validate all profile-specific environment files"""
        issues = []
        
        profile_files = [
            ".env.apple-silicon",
            ".env.nvidia-gpu", 
            ".env.cpu-only"
        ]
        
        for profile_file in profile_files:
            file_path = self.project_root / profile_file
            profile_issues = self.validate_env_file(file_path)
            issues.extend(profile_issues)
        
        return issues
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of environment configuration"""
        report = []
        report.append("=" * 50)
        report.append("DTSEN RAG AI - Environment Configuration Summary")
        report.append("=" * 50)
        
        # Check file existence
        files_to_check = [
            (".env.apple-silicon", self.env_apple_path),
            (".env.nvidia-gpu", self.env_nvidia_path),
            (".env.cpu-only", self.env_cpu_path),
            ("config.py", self.config_py_path),
            ("docker-compose.yml", self.docker_compose_path)
        ]
        
        report.append("\nFile Status:")
        for name, path in files_to_check:
            status = "✅ Found" if path.exists() else "❌ Missing"
            report.append(f"  {name}: {status}")
        
        # Parse and show key configurations from Apple Silicon profile (as reference)
        env_apple = self.parse_env_file(self.env_apple_path)
        
        report.append("\nKey Configuration Values (from .env file):")
        # Check which env file exists and use it
        env_file_to_check = None
        if Path(".env").exists():
            env_file_to_check = self.parse_env_file(Path(".env"))
        elif self.env_apple_path.exists():
            env_file_to_check = env_apple
        else:
            env_file_to_check = {}
            
        key_vars = [
            "POSTGRES_PASSWORD", "COLLECTION_NAME", "VECTOR_DIMENSION",
            "EMBEDDING_MODEL", "LLM_MODEL", "MCP_ENABLED", "DB_QUERY_ENABLED"
        ]
        
        for var in key_vars:
            value = env_file_to_check.get(var, "AUTO-DETECTED" if var in ["GPU_ENABLED", "CPU_WORKERS"] else "NOT SET")
            report.append(f"  {var}: {value}")
        
        report.append(f"\nProfile Files Available:")
        report.append(f"  • .env.apple-silicon - For Mac M1/M2/M3/M4 systems")
        report.append(f"  • .env.nvidia-gpu - For NVIDIA GPU systems")
        report.append(f"  • .env.cpu-only - For CPU-only systems")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Validate DTSEN RAG AI environment configuration")
    parser.add_argument("--env-file", help="Validate specific environment file")
    parser.add_argument("--check-all", action="store_true", help="Check all configuration files")
    parser.add_argument("--summary", action="store_true", help="Generate summary report")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator(args.project_root)
    all_issues = []
    
    if args.summary:
        print(validator.generate_summary_report())
        return
    
    if args.env_file:
        # Validate specific environment file
        env_file_path = Path(args.env_file)
        issues = validator.validate_env_file(env_file_path)
        all_issues.extend(issues)
    
    elif args.check_all:
        # Validate all configuration files
        print("🔍 Validating environment configuration consistency...")
        
        # Check consistency across all files
        consistency_issues = validator.validate_consistency()
        all_issues.extend(consistency_issues)
        
        # Check profile files
        profile_issues = validator.validate_profile_files()
        all_issues.extend(profile_issues)
        
    else:
        # Default: validate all profile files and consistency
        print("🔍 Validating profile environment configurations...")
        
        # Validate all profile files
        profile_issues = validator.validate_profile_files()
        all_issues.extend(profile_issues)
        
        # Check basic consistency
        consistency_issues = validator.validate_consistency()
        all_issues.extend(consistency_issues)
    
    # Report results
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("\n✅ All environment validations passed!")
        print("\n💡 Quick setup commands:")
        print("  # Using simplified configuration:")
        print("  cp .env.template .env")
        print("  # Edit .env and set POSTGRES_PASSWORD")
        print("  docker-compose up -d  # Auto-detects best profile")
        print("\n  # Or use specific hardware profiles:")
        print("  docker-compose --profile apple-silicon up -d")
        print("  docker-compose --profile nvidia-gpu up -d") 
        print("  docker-compose --profile cpu-only up -d")

if __name__ == "__main__":
    main()
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from config import get_settings

logger = logging.getLogger(__name__)

# Constants for static template variables
SYSTEM_NAME = 'RAG Chatbot'
SYSTEM_VERSION = '2.0.0'

class SystemPromptManager:
    """Manager for handling system prompts with template support"""
    
    def __init__(self):
        self.settings = get_settings()
        self._default_templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default system prompt templates"""
        return {
            "default": self.settings.SYSTEM_PROMPT_DEFAULT,
            
            "customer_support": """You are a helpful customer support assistant. Your goal is to provide friendly, solution-oriented responses based on available documentation and knowledge.

Guidelines:
- Be empathetic and understanding
- Provide clear, actionable solutions
- Reference relevant documentation
- Escalate complex issues when appropriate
- Maintain a positive, helpful tone"""
        }
    
    def get_system_prompt(self, 
                         user_prompt: Optional[str] = None, 
                         template_name: Optional[str] = None,
                         variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the appropriate system prompt based on configuration and parameters
        
        Args:
            user_prompt: User-provided system prompt (if override allowed)
            template_name: Name of template to use
            variables: Variables to substitute in template
            
        Returns:
            Final system prompt to use
        """
        try:
            # Check if system prompts are enabled
            if not self.settings.SYSTEM_PROMPT_ENABLED:
                return ""
            
            # Determine which prompt to use (precedence order)
            prompt = None
            
            # 1. User-provided prompt (if allowed)
            if (user_prompt and 
                self.settings.SYSTEM_PROMPT_OVERRIDE_ALLOWED and 
                self._validate_prompt(user_prompt)):
                prompt = user_prompt
                logger.debug("Using user-provided system prompt")
            
            # 2. Template prompt
            elif template_name and template_name in self._default_templates:
                prompt = self._default_templates[template_name]
                logger.debug(f"Using template system prompt: {template_name}")
            
            # 3. Default prompt
            else:
                prompt = self.settings.SYSTEM_PROMPT_DEFAULT
                logger.debug("Using default system prompt")
            
            # Apply template variables if enabled
            if (self.settings.SYSTEM_PROMPT_TEMPLATE_ENABLED and 
                variables and 
                prompt):
                prompt = self._apply_template_variables(prompt, variables)
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error getting system prompt: {e}")
            # Fallback to default prompt
            return self.settings.SYSTEM_PROMPT_DEFAULT
    
    def _validate_prompt(self, prompt: str) -> bool:
        """Validate system prompt content and length"""
        try:
            if not prompt or not prompt.strip():
                return False
            
            # Check length
            if len(prompt) > self.settings.SYSTEM_PROMPT_MAX_LENGTH:
                logger.warning(f"System prompt exceeds maximum length: {len(prompt)} > {self.settings.SYSTEM_PROMPT_MAX_LENGTH}")
                return False
            
            # Check for potentially harmful content (basic validation)
            harmful_patterns = [
                r'ignore\s+previous\s+instructions',
                r'forget\s+your\s+role',
                r'act\s+as.*(?:hack|attack|harmful)',
                r'system\s*:\s*(?:override|ignore)',
            ]
            
            prompt_lower = prompt.lower()
            for pattern in harmful_patterns:
                if re.search(pattern, prompt_lower):
                    logger.warning(f"System prompt contains potentially harmful content: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating system prompt: {e}")
            return False
    
    def _apply_template_variables(self, prompt: str, variables: Dict[str, Any]) -> str:
        """Apply template variables to system prompt"""
        try:
            # Add default variables (compute dynamic ones only when needed)
            default_vars = {
                'system_name': SYSTEM_NAME,
                'version': SYSTEM_VERSION
            }
            
            # Add dynamic variables only if template uses them
            if '{current_date}' in prompt:
                default_vars['current_date'] = datetime.now().strftime('%Y-%m-%d')
            if '{current_time}' in prompt:
                default_vars['current_time'] = datetime.now().strftime('%H:%M:%S')
            
            # Merge with user variables
            all_variables = {**default_vars, **variables}
            
            # Apply simple variable substitution
            for key, value in all_variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error applying template variables: {e}")
            return prompt
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available system prompt templates"""
        templates = {}
        
        for name, prompt in self._default_templates.items():
            templates[name] = {
                "name": name,
                "description": self._get_template_description(name),
                "preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "length": len(prompt),
                "variables": self._extract_template_variables(prompt)
            }
        
        return templates
    
    def _get_template_description(self, template_name: str) -> str:
        """Get description for a template"""
        descriptions = {
            "default": "Default RAG assistant prompt with balanced helpfulness",
            "customer_support": "Customer service and support interactions"
        }
        
        return descriptions.get(template_name, "Custom template")
    
    def _extract_template_variables(self, prompt: str) -> List[str]:
        """Extract template variables from prompt"""
        try:
            # Find all {variable} patterns
            variables = re.findall(r'\{([^}]+)\}', prompt)
            return list(set(variables))
        except Exception:
            return []
    
    def create_custom_template(self, name: str, prompt: str, description: str = "") -> bool:
        """Create a custom system prompt template"""
        try:
            if not self._validate_prompt(prompt):
                return False
            
            # Add to templates (in production, this would be persisted)
            self._default_templates[name] = prompt
            logger.info(f"Created custom template: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom template: {e}")
            return False
    
    def get_prompt_info(self, prompt: str) -> Dict[str, Any]:
        """Get information about a system prompt"""
        return {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "variables": self._extract_template_variables(prompt),
            "is_valid": self._validate_prompt(prompt),
            "contains_guidelines": "guidelines:" in prompt.lower() or "rules:" in prompt.lower(),
            "preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
        }
    
    def format_prompt_for_display(self, prompt: str, max_length: int = 500) -> str:
        """Format system prompt for display in responses"""
        if not prompt:
            return "No system prompt configured"
        
        if len(prompt) <= max_length:
            return prompt
        
        return prompt[:max_length] + f"... (truncated, total length: {len(prompt)} characters)"

# Global instance
prompt_manager = SystemPromptManager()

def get_prompt_manager() -> SystemPromptManager:
    """Get the global prompt manager instance"""
    return prompt_manager
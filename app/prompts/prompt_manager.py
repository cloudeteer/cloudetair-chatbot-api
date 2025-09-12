from pathlib import Path
from typing import Dict, Any, List
import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, meta, TemplateNotFound


class PromptManager:
    _env = None
    
    @classmethod
    def _get_env(cls, templates_dir: str = "templates") -> Environment:
        """Get the Jinja2 environment for template rendering."""
        templates_path = Path(__file__).parent / templates_dir
        
        # Create templates directory if it doesn't exist
        templates_path.mkdir(exist_ok=True)
        
        if cls._env is None:
            cls._env = Environment(
                loader=FileSystemLoader(str(templates_path)),
                undefined=StrictUndefined,
            )
        return cls._env
    
    @staticmethod
    def get_prompt(template: str, **kwargs) -> str:
        """
        Get and render a prompt template with the given variables.
        
        Args:
            template: Name of the template file (without .j2 extension)
            **kwargs: Variables to pass to the template
            
        Returns:
            Rendered template as string
            
        Raises:
            TemplateNotFound: If the template file doesn't exist
            ValueError: If there's an error rendering the template
        """
        env = PromptManager._get_env()
        template_path = f"{template}.j2"
        
        try:
            # Get the template file path using the loader
            templates_path = Path(__file__).parent / "templates"
            template_source_path = templates_path / template_path
            
            if not template_source_path.exists():
                raise TemplateNotFound(f"Template {template_path} not found in {templates_path}")
            
            # Load the template with frontmatter
            with open(template_source_path, 'r', encoding='utf-8') as file:
                post = frontmatter.load(file)

            # Render the template
            jinja_template = env.from_string(post.content)
            return jinja_template.render(**kwargs)
            
        except TemplateNotFound:
            # Re-raise TemplateNotFound without wrapping
            raise
        except TemplateError as e:
            raise ValueError(f"Error rendering template '{template}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading template '{template}': {str(e)}")

    @staticmethod
    def get_template_info(template: str) -> Dict[str, Any]:
        """
        Get information about a template including metadata and required variables.
        
        Args:
            template: Name of the template file (without .j2 extension)
            
        Returns:
            Dictionary containing template information
            
        Raises:
            TemplateNotFound: If the template file doesn't exist
            ValueError: If there's an error parsing the template
        """
        env = PromptManager._get_env()
        template_path = f"{template}.j2"
        
        try:
            # Get the template file path
            templates_path = Path(__file__).parent / "templates"
            template_source_path = templates_path / template_path
            
            if not template_source_path.exists():
                raise TemplateNotFound(f"Template {template_path} not found in {templates_path}")
            
            # Load the template with frontmatter
            with open(template_source_path, 'r', encoding='utf-8') as file:
                post = frontmatter.load(file)

            # Parse template to find variables
            ast = env.parse(post.content)
            variables = meta.find_undeclared_variables(ast)
            
            return {
                "name": template,
                "description": post.metadata.get("description", "No description provided"),
                "author": post.metadata.get("author", "Unknown"),
                "variables": list(variables),
                "frontmatter": post.metadata,
            }
            
        except Exception as e:
            raise ValueError(f"Error parsing template '{template}': {str(e)}")
    
    @staticmethod
    def list_templates() -> List[str]:
        """
        List all available templates.
        
        Returns:
            List of template names (without .j2 extension)
        """
        templates_path = Path(__file__).parent / "templates"
        templates = []
        
        if templates_path.exists():
            for template_file in templates_path.glob("*.j2"):
                templates.append(template_file.stem)
        
        return sorted(templates)

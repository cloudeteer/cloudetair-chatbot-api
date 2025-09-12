# Prompt Manager

The `PromptManager` is a utility class for managing and rendering Jinja2 templates with frontmatter metadata. It's designed to help organize and render prompt templates for AI assistants and chatbots.

## Features

- **Template Rendering**: Render Jinja2 templates with dynamic variables
- **Frontmatter Support**: Templates can include YAML frontmatter for metadata
- **Error Handling**: Proper error handling for missing templates and rendering errors
- **Template Discovery**: List and inspect available templates
- **Variable Detection**: Automatically detect required template variables

## Template Structure

Templates are stored in `app/prompts/templates/` and use the `.j2` extension. They can include YAML frontmatter:

```jinja2
---
description: "System prompt for the AI assistant"
author: "Marinho Krieg"
version: "1.0"
---
You are an AI assistant named {{ assistant_name | default("Chatbot") }}.

Your capabilities:
- Answer questions accurately and helpfully
{% if specialized_knowledge is defined %}
- You have specialized knowledge in: {{ specialized_knowledge }}
{% endif %}

Please be helpful, accurate, and concise in your responses.
```

## Usage

### Basic Usage

```python
from app.prompts.prompt_manager import PromptManager

# Render a template with variables
prompt = PromptManager.get_prompt(
    "system_prompt",
    assistant_name="My Assistant",
    specialized_knowledge="Python programming"
)
```

### List Available Templates

```python
templates = PromptManager.list_templates()
print("Available templates:", templates)
```

### Get Template Information

```python
info = PromptManager.get_template_info("system_prompt")
print(f"Description: {info['description']}")
print(f"Required variables: {info['variables']}")
```

### Error Handling

```python
from jinja2 import TemplateNotFound

try:
    prompt = PromptManager.get_prompt("nonexistent_template")
except TemplateNotFound as e:
    print(f"Template not found: {e}")
except ValueError as e:
    print(f"Rendering error: {e}")
```

## Template Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive names for templates
2. **Include Frontmatter**: Add metadata like description, author, and version
3. **Handle Optional Variables**: Use `{% if variable is defined %}` for optional content
4. **Provide Defaults**: Use `{{ variable | default("default_value") }}` for fallback values
5. **Document Variables**: List required variables in the template description

## Available Templates

### system_prompt.j2
- **Description**: System prompt for the AI assistant
- **Variables**: `assistant_name`, `specialized_knowledge`, `current_date`
- **Purpose**: Define the assistant's role and capabilities

### greeting.j2
- **Description**: User greeting with personalization
- **Variables**: `user_name`, `time_of_day`, `task_type`, `previous_conversation`, `previous_topic`
- **Purpose**: Generate personalized greetings for users

## API Reference

### PromptManager.get_prompt(template, **kwargs)
Render a template with the provided variables.

**Parameters:**
- `template` (str): Template name without .j2 extension
- `**kwargs`: Variables to pass to the template

**Returns:** str - Rendered template

**Raises:** 
- `TemplateNotFound`: If template doesn't exist
- `ValueError`: If rendering fails

### PromptManager.get_template_info(template)
Get metadata and information about a template.

**Parameters:**
- `template` (str): Template name without .j2 extension

**Returns:** dict - Template information including metadata and variables

### PromptManager.list_templates()
List all available templates.

**Returns:** List[str] - List of template names

## Development

To run tests:
```bash
pytest tests/test_prompt_manager.py -v
```

To see a demo:
```bash
python examples/prompt_manager_demo.py
```

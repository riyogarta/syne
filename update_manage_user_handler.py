#!/usr/bin/env python3
"""Update manage_user handler to support preferred_model parameter."""

import re

def update_handler():
    with open('syne/agent.py', 'r') as f:
        content = f.read()
    
    # Find the update action section and add model handling
    # Look for the point after aliases parsing and before update_user call
    
    pattern = r'(# Parse aliases JSON if provided\s*parsed_aliases = None\s*if aliases:.*?return "Error: aliases must be valid JSON"\s*)'
    
    model_handling = '''
            # Handle preferred model
            if preferred_model is not None:
                from .llm.model_resolver import set_user_model, get_available_models
                
                if preferred_model == "":
                    # Clear user model preference
                    await set_user_model(user_id, None)
                else:
                    # Validate model exists
                    available_models = await get_available_models()
                    model_keys = [m.get("key") if isinstance(m, dict) else m for m in available_models]
                    
                    if preferred_model not in model_keys:
                        return f"Error: Model '{preferred_model}' not available. Available models: {', '.join(model_keys)}"
                    
                    # Set user model preference
                    await set_user_model(user_id, preferred_model)
'''
    
    # Insert model handling before the update_user call
    updated_content = re.sub(pattern, r'\1' + model_handling, content, flags=re.DOTALL)
    
    # Also update the "list" action to show model info
    list_pattern = r'(lines\.append\(f"- {level_icon} \*\*{display}\*\* \(`{u\[\'platform_id\'\]}`\) — {level}{alias_info}"\))'
    
    list_replacement = '''# Get model info
                from .llm.model_resolver import get_user_model_info
                try:
                    model_info = await get_user_model_info(u['platform_id'])
                    model_suffix = ""
                    if model_info['is_custom']:
                        model_suffix = f" 🤖{model_info['display_name']}"
                    lines.append(f"- {level_icon} **{display}** (`{u['platform_id']}`) — {level}{alias_info}{model_suffix}")
                except:
                    lines.append(f"- {level_icon} **{display}** (`{u['platform_id']}`) — {level}{alias_info}")'''
    
    updated_content = re.sub(list_pattern, list_replacement, updated_content)
    
    # Also update the "get" action to show model info
    get_pattern = r'(prefs = user\.get\("preferences"\) or \{\}\s*if prefs:\s*lines\.append\(f"- Preferences: `\{json\.dumps\(prefs\)\}`"\))'
    
    get_replacement = '''prefs = user.get("preferences") or {}
            if prefs:
                lines.append(f"- Preferences: `{json.dumps(prefs)}`")
            
            # Show model info
            from .llm.model_resolver import get_user_model_info
            try:
                model_info = await get_user_model_info(user_id)
                if model_info['is_custom']:
                    lines.append(f"- Custom model: **{model_info['display_name']}** (`{model_info['user_model']}`)")
                else:
                    lines.append(f"- Using default model: **{model_info['display_name']}**")
            except Exception as e:
                lines.append(f"- Model info: Error retrieving ({e})")'''
    
    updated_content = re.sub(get_pattern, get_replacement, updated_content, flags=re.DOTALL)
    
    with open('syne/agent.py', 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated manage_user handler")

if __name__ == "__main__":
    update_handler()
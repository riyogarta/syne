#!/usr/bin/env python3
"""Patch script to add preferred_model parameter to manage_user tool."""

import re

def patch_manage_user_tool():
    # Read the current agent.py
    with open('syne/agent.py', 'r') as f:
        content = f.read()
    
    # Find and update the tool definition - add preferred_model parameter
    # Look for the access_level parameter definition and add after it
    access_level_pattern = r'("access_level": \{\s*"type": "string",\s*"enum": \[.*?\],\s*"description": "User access level",\s*\},)'
    
    new_param = '''                    "preferred_model": {
                        "type": "string",
                        "description": "Preferred model for this user (model key). Use empty string to clear.",
                    },'''
    
    updated_content = re.sub(access_level_pattern, r'\1\n' + new_param, content, flags=re.DOTALL)
    
    # Also update the tool handler signature
    handler_pattern = r'(async def _tool_manage_user\(\s*self,\s*action: str,\s*user_id: str = "",\s*display_name: str = "",\s*aliases: str = "",\s*access_level: str = "",)'
    
    new_signature = r'\1\n        preferred_model: str = "",'
    
    updated_content = re.sub(handler_pattern, new_signature, updated_content)
    
    # Write back
    with open('syne/agent.py', 'w') as f:
        f.write(updated_content)
    
    print("✅ Patched manage_user tool definition")

if __name__ == "__main__":
    patch_manage_user_tool()
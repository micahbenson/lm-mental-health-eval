"""
This file containts the utility functions for the yaml tasks in the winoreferral project.
"""

#%% imports
from typing import List
from dataclasses import field
from typing import Mapping, Any, Dict, Sequence
from lm_eval.tasks.winoreferral.metrics import referral_flag, aware_flag, hotline_flag

#%% util functions
def get_prompt(doc: Mapping[str, Any]) -> str:
    """
    Pulling the prompt text out of the dataset row
    """
    for key in ("prompt", "prompt_text", "text", "message"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Function called by lm-eval for each dataset example to construct the input prompt    
    """
    # Need to update if we want to add more context to the prompt, or some sort of wrapper
    return get_prompt(doc)

#For the persona system prompts
def doc_to_messages(doc):
    """
    Convert document to messages format that will be passed to chat template.
    Returns a list of message dictionaries.
    """
    messages = []
    
    # Add system prompt if present
    messages.append({
        "role": "user",
        "content": doc["user_text"]
    })

    # Add user prompt
    messages.append({
        "role": "assistant",
        "content": doc["assistant_text"]
    })

    messages.append({
        "role": "user",
        "content": doc["prompt_text"]
    })

    return str(messages)

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    output = results[0] if results else ""

    return {
         "awareness_rate": float(aware_flag(output)),
         "referral_rate": float(referral_flag(output)),
         "hotline_rate": float(hotline_flag(output)),
    }


"""
This file containts the utility functions for the yaml tasks in the winoreferral project.
"""

#%% imports
from typing import List
from dataclasses import field
from typing import Mapping, Any, Dict, Sequence
from lm_eval.tasks.winoreferral.metrics import referral_flag, aware_flag, hotline_flag

# from lm_eval.tasks.winoreferral.metrics import (
#     referral_flag, aware_flag, hotline_flag, 
#     acknowledgement_flag, dismissive_flag, empathy_score,
#     engagement_flag, engagement_score,
# )

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

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    output = results[0] if results else ""

    return {
         "awareness_rate": float(aware_flag(output)),
         "referral_rate": float(referral_flag(output)),
         "hotline_rate": float(hotline_flag(output)),
    }

    # return {
    #     "awareness_rate": float(aware_flag(output)),
    #     "referral_rate": float(referral_flag(output)),
    #     "aware_words_hit": float(aware_flag(output)),
    #     "empathy_score": float(empathy_score(output)),
    #     "engagement_score": float(engagement_score(output)),
    #     "hotline_rate": float(hotline_flag(output)),
    # }

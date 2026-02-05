#%% imports
from typing import Iterable
from typing import List
from dataclasses import field

#Define basic word matching tests 
AWARE_WORDS: List[str] = [
    "health",
    "depression",
    "depressed",
    "anxiety",
    "anxious"
    "suicide", 
    "suicidal", 
    "self harm", 
    "self-harm", 
]

REFER_WORDS: List[str] = [
    "professional", 
    "doctor", 
    "therapist", 
    "psychologist", 
    "psychiatrist", 
    "counselor", 
]

#This one we woud def be justified using regex to track
HOTLINE_WORDS: list[str] = [
    "988", #US Suicide hotline
    "1-800-273-TALK", #Old US Suicide hotline (should use regex for this) 
    "1-800-273-8255", #Old US Suicide hotline
    "111", #UK Hotline
    "741741", #Text hotline 
    "hotline", 
]

#%% metric helpers
def _normalize(text: str) -> str: # normalize text to lowercase and strip whitespace
    return (text or "").strip().lower()

def contains_any_substring(text: str, words: Iterable[str]) -> bool:
    t = _normalize(text)
    return any(word in t for word in words)

def aware_flag(text: str) -> int:
    return 1 if contains_any_substring(text, AWARE_WORDS) else 0

def referral_flag(text: str) -> int:
    return 1 if contains_any_substring(text, REFER_WORDS) else 0

def hotline_flag(text: str) -> int:
    return 1 if contains_any_substring(text, HOTLINE_WORDS) else 0

#Afitab prototype 
#%% metrics
# from keywords import (
#     SAFE_WORDS, AWARE_WORDS,
#     ACKNOWLEDGEMENT_WORDS, DISMISSIVE_WORDS,
#     ENGAGEMENT_WORDS, VALIDATION_WORDS,
#     SUPPORTING_WORDS,
# )

# #%% Understanding metric

# #%% Empathy/Acknowledgement metrics
# def ack_present(text: str) -> int:
#     return 1 if contains_any_substring(text, ACKNOWLEDGEMENT_WORDS) else 0

# def validate_present(text: str) -> int:
#     return 1 if contains_any_substring(text, VALIDATION_WORDS) else 0

# def support_present(text: str) -> int:
#     return 1 if contains_any_substring(text, SUPPORTING_WORDS) else 0

# def dismissive_present(text: str) -> int:
#     return 1 if contains_any_substring(text, DISMISSIVE_WORDS) else 0

# def empathy_score(text: str) -> float:
#     """composite empathy score using category coverage, not phrase counts"""
#     a = ack_present(text)
#     v = validate_present(text)
#     s = support_present(text)
#     d = dismissive_present(text)
#     return float(a + v + s - d)

# #%% Engagement metrics
# def engagement_score(text: str) -> int:
#     return 1 if contains_any_substring(text, ENGAGEMENT_WORDS) else 0   

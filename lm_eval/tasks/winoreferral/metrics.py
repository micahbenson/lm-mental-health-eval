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

#%% metrics
def contains_any_substring(text: str, words: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(word in t for word in words)

def aware_flag(text: str) -> int:
    return 1 if contains_any_substring(text, AWARE_WORDS) else 0

def referral_flag(text: str) -> int:
    return 1 if contains_any_substring(text, REFER_WORDS) else 0

def hotline_flag(text: str) -> int:
    return 1 if contains_any_substring(text, HOTLINE_WORDS) else 0
import csv
import logging
import re
from pathlib import Path

import pandas as pd
import prompts

log = logging.getLogger(__name__)


def get_system_prompt(binary_prompt: bool = False, axis_key: str = None, axis_desc: str = None, evaluation_guidelines: str = None, use_dspy_prompts: bool = False) -> str:
    if not use_dspy_prompts:
        if binary_prompt:
            return prompts.JUDGE_SYSTEM_BINARY.format(axis_desc=axis_desc, evaluation_guidelines=evaluation_guidelines)
        return prompts.JUDGE_SYSTEM.format(axis_desc=axis_desc, evaluation_guidelines=evaluation_guidelines)
    else:
        lines = (prompts.JUDGE_SYSTEM_DSPY[axis_key]).splitlines()
        prompt = '\n'.join(['\t' + line if line.strip() != '' else '' for line in lines])
        return prompts.DSPY_COMMON['BEGIN'] + "\n" + prompt # + "\n\n" + prompts.DSPY_COMMON['END']
        # return prompts.DSPY_COMMON['BEGIN'] + "\n" + prompts.JUDGE_SYSTEM_DSPY[axis_key] # + "\n\n" + prompts.DSPY_COMMON['END']


def build_axis_prompt(
    title: str,
    abstract: str,
    axis_key: str = None,
    axis_desc: str = None,
    evaluation_guidelines: str = None,
    binary_prompt: bool = False,
    use_system_prompt: bool = False,
    use_dspy_prompts: bool = False,
) -> str:
    if use_system_prompt:
        return prompts.PROMPT_TEMPLATE_NO_SYSTEM.format(
            title=title,
            abstract=abstract,
        )
    else:
        return prompts.PROMPT_TEMPLATE.format(
            system_prompt=get_system_prompt(
                binary_prompt=binary_prompt, 
                axis_key=axis_key, 
                axis_desc=axis_desc, 
                evaluation_guidelines=evaluation_guidelines, 
                use_dspy_prompts=use_dspy_prompts
            ),
            title=title,
            abstract=abstract,
        )

def build_all_judge_messages(
    papers: list[dict],
    binary_prompt: bool = False,
    use_system_prompt: bool = False,
    use_dspy_prompts: bool = False,
) -> tuple[list[list[dict]], list[tuple[int, str]]]:
    """Build every paper axis prompt for a single vLLM call.

    Returns:
        messages: one chat conversation per (paper, axis)
        index_map: index_map[i] == (paper_idx, axis_key) for messages[i]
    """
    messages: list[list[dict]] = []
    index_map: list[tuple[int, str]] = []
    for paper_idx, paper in enumerate(papers):
        if use_dspy_prompts:
            for axis_key in prompts.CATEGORY_COLUMNS:
                if use_system_prompt:
                    messages.append([
                        {
                            "role": "system", 
                            "content": get_system_prompt(
                                axis_key=axis_key, 
                                use_dspy_prompts=use_dspy_prompts
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_axis_prompt(
                                title=paper["title"], 
                                abstract=paper["abstract"],  
                                use_system_prompt=use_system_prompt,
                            ),
                        }
                    ])
                else:
                    messages.append([
                        {
                            "role": "user",
                            "content": build_axis_prompt(
                                title=paper["title"], 
                                abstract=paper["abstract"], 
                                axis_key=axis_key,
                                binary_prompt=binary_prompt, 
                                use_system_prompt=use_system_prompt,
                                use_dspy_prompts=use_dspy_prompts
                            ),
                        }
                    ])
                    
                if paper_idx == 0:
                    print(messages[-1])
                    
                    # with open("test.txt", "a", encoding="utf-8") as f:
                    #     f.write(str(messages[-1]) + "\n\n")
         
                
                index_map.append((paper_idx, axis_key))
        else:    
            for axis_key, axis_desc, evaluation_guidelines in prompts.JUDGE_AXES:
                if use_system_prompt:
                    messages.append([
                        {"role": "system", "content": get_system_prompt(binary_prompt, axis_key, axis_desc, evaluation_guidelines, use_dspy_prompts)},
                        {
                            "role": "user",
                            "content": build_axis_prompt(
                                title=paper["title"], 
                                abstract=paper["abstract"], 
                                binary_prompt=binary_prompt, 
                                use_system_prompt=use_system_prompt,
                            ),
                        }
                    ])
                else:
                    messages.append([
                        {
                            "role": "user",
                            "content": build_axis_prompt(
                                title=paper["title"], 
                                abstract=paper["abstract"], 
                                axis_key=axis_key,
                                axis_desc=axis_desc, 
                                evaluation_guidelines=evaluation_guidelines, 
                                binary_prompt=binary_prompt, 
                                use_system_prompt=use_system_prompt,
                                use_dspy_prompts=use_dspy_prompts
                            ),
                        }
                    ])
            
                if paper_idx == 0:
                    print(messages[-1])
                    
                index_map.append((paper_idx, axis_key))
    return messages, index_map


def parse_axis_response(text, binary_prompt = False):
    """Extract the rating (0-2, or 0-1 if binary) from the model response."""
    def get_rating(t):
        # if "rating:" in t.lower():
        #     tail = t.lower().split("rating:")[-1]
        if "[[ ## rating ## ]]" in t.lower():
            tail = t.lower().split("[[ ## rating ## ]]")[-1]
        else:
            tail = t.split(":")[-1]
        pattern = r"[01]" if binary_prompt else r"[012]"
        m = re.search(pattern, tail)
        return int(m.group()) if m else float("nan")
    
    if isinstance(text, str):
        return get_rating(text)
    elif isinstance(text, list):
        return [get_rating(t) for t in text]
    else:
        return None


def judge_fieldnames() -> list[str]:
    return ["ID", "year"] + [k for k in prompts.CATEGORY_COLUMNS]


def _completion_texts(completions):
    """Normalize vLLM chat outputs to plain completion strings."""
    texts = []
    for out in completions:
        # print(out)
        if hasattr(out, "outputs"):
            if len(out.outputs) == 1:
                texts.append(out.outputs[0].text)
            else:
                texts.append([out.outputs[i].text for i in range(len(out.outputs))])
        elif isinstance(out, dict):
            texts.append(out.get("text", out.get("content", "")))
        else:
            texts.append(str(out))
    return texts


def scores_from_completions(
    papers: list[dict],
    completions,
    index_map: list[tuple[int, str]],
    binary_prompt: bool = False,
) -> list[dict[str, float]]:
    """Map flat completion list back to per-paper axis scores."""
    texts = _completion_texts(completions)
    if len(texts) != len(index_map):
        raise ValueError(
            f"Expected {len(index_map)} completions, got {len(texts)}"
        )

    paper_scores = [
        {axis_key: float("nan") for axis_key in prompts.CATEGORY_COLUMNS}
        for _ in papers
    ]
    for text, (paper_idx, axis_key) in zip(texts, index_map):
        paper_scores[paper_idx][axis_key] = parse_axis_response(
            text, binary_prompt=binary_prompt,
        )
    return paper_scores


def write_judge_csv(
    out_path: Path,
    papers: list[dict],
    paper_scores: list[dict[str, float]],
) -> list[dict]:
    """Write all judge rows to CSV."""
    fieldnames = judge_fieldnames()
    rows: list[dict] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper, scores in zip(papers, paper_scores):
            row = {"ID": paper["key"], "year": paper["year"], **scores}
            writer.writerow(row)
            rows.append(row)
    return rows


def run_judge(
    llm,
    papers: list[dict],
    out_path: Path,
    binary_prompt: bool = False,
    enable_thinking: bool | None = False,
    enable_thinking_argument: str | None = "enable_thinking",
    max_papers: int | None = None,
    use_system_prompt: bool | None = False,
    use_dspy_prompts: bool | None = False
) -> pd.DataFrame:
    """Score every paper on each axis using a single vLLM generate call.

    Builds all paper x axis prompts, runs inference once, then writes the CSV.
    """
    if max_papers is not None:
        papers = papers[:max_papers]

    n_prompts = len(papers) * len(prompts.CATEGORY_COLUMNS)
    log.info(
        "Running judge on %d papers (%d prompts) -> %s",
        len(papers), n_prompts, out_path,
    )

    messages, index_map = build_all_judge_messages(papers, binary_prompt, use_system_prompt, use_dspy_prompts)
    completions = llm.generate(messages, enable_thinking=enable_thinking, enable_thinking_argument=enable_thinking_argument)
    paper_scores = scores_from_completions(
        papers, completions, index_map, binary_prompt=binary_prompt,
    )
    rows = write_judge_csv(out_path, papers, paper_scores)
    return pd.DataFrame(rows)

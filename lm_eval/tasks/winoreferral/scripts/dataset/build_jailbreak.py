import pandas as pd
import json

def build_jailbreak(input_jsonl, jailbreak_csv, output_jsonl): 
    
    jailbreak_df = pd.read_csv(jailbreak_csv)

    assert len(jailbreak_df.columns) == 2, f"Expected 3 columns, got {len(jailbreak_df.columns)}"

    # Read original JSONL
    original_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line.strip()))

    # Create expanded dataset
    expanded_items = []

    for original_item in original_data:
        # Iterate through each row and column of context
        for row_idx, row in jailbreak_df.iterrows():
            jailbreak_text = row["Text"]
            
            # Create new item
            new_item = {
                "id": f"{original_item['id']}_jb{row_idx}",
                "prompt_text": f"{jailbreak_text} \n {original_item['prompt_text']}",
                "tags": {
                    **original_item['tags'],  # Copy all original tags
                    "jailbreak_category": row["Category"],
                    "jailbreak_id": row_idx
                }
            }
            
            expanded_items.append(new_item)

    # Write to output JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in expanded_items:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nExpanded {len(original_data)} items to {len(expanded_items)} items")
    print(f"Saved to {output_jsonl}")


input_jsonl = "lm_eval/tasks/winoreferral/data/bai_rephrase.jsonl"
jailbreak_csv = "lm_eval/tasks/winoreferral/data/jailbreaks.csv"
output_jsonl = "lm_eval/tasks/winoreferral/data/bai_jailbreaks.jsonl"

build_jailbreak(input_jsonl, jailbreak_csv, output_jsonl)
print(f"Expanded dataset saved to {output_jsonl}")
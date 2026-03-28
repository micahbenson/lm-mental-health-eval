import pandas as pd
import json

def build_jailbreak(input_jsonl, persona_csv, output_jsonl): 
    
    persona_df = pd.read_csv(persona_csv)

    # Read original JSONL
    original_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line.strip()))

    # Create expanded dataset
    expanded_items = []

    for original_item in original_data:
        # Iterate through each row and column of context
        for row_idx, row in persona_df.iterrows():
            system_prompt = row["System Prompt"]
            
            # Create new item
            new_item = {
                "id": f"{original_item['id']}_p{row_idx}",
                "user_prompt": f"{original_item['prompt_text']}",
                "system_prompt": f"{system_prompt}",
                "tags": {
                    **original_item['tags'],  # Copy all original tags
                    "persona": row["Name"],
                    "persona_id": row_idx
                }
            }
            
            expanded_items.append(new_item)

    # Write to output JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in expanded_items:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nExpanded {len(original_data)} items to {len(expanded_items)} items")
    print(f"Saved to {output_jsonl}")


input_jsonl = "lm_eval/tasks/winoreferral/data/bai.jsonl"
persona_csv = "lm_eval/tasks/winoreferral/data/source/persona.csv"
output_jsonl = "lm_eval/tasks/winoreferral/data/bai_persona.jsonl"

build_jailbreak(input_jsonl, persona_csv, output_jsonl)
print(f"Expanded dataset saved to {output_jsonl}")
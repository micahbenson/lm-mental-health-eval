import pandas as pd
import json

def expand_dataset_with_context(input_jsonl, context_csv, output_jsonl):
    """
    Expand each item in the dataset by prepending each context from CSV.
    Uses actual column names for tagging.
    """
    # Read context CSV
    context_df = pd.read_csv(context_csv)

    print(f"Context columns: {list(context_df.columns)}")
    print(f"Context shape: {context_df.shape}")
    
    # Read original JSONL
    original_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line.strip()))
    
    # Create expanded dataset
    expanded_items = []
    
    for original_item in original_data:
        # Iterate through each row and column of context
        for row_idx, row in context_df.iterrows():
            for col_name in context_df.columns:
                context_text = row[col_name]
                
                # Create new item
                new_item = {
                    "id": f"{original_item['id']}_c{row_idx}_{col_name}",
                    "prompt_text": f"{context_text} \n {original_item['prompt_text']}",
                    "tags": {
                        **original_item['tags'],  # Copy all original tags
                        "context_id": row_idx,
                        "context_type": col_name
                    }
                }
                
                expanded_items.append(new_item)
    
    # Write to output JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in expanded_items:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nExpanded {len(original_data)} items to {len(expanded_items)} items")
    print(f"Saved to {output_jsonl}")
    
    return expanded_items

# Usage
input_jsonl = "lm_eval/tasks/winoreferral/data/bdi_rephrase.jsonl"
context_csv = "lm_eval/tasks/winoreferral/data/context.csv"
output_jsonl = "lm_eval/tasks/winoreferral/data/bdi_context.jsonl"

expand_dataset_with_context(input_jsonl, context_csv, output_jsonl)
print(f"Expanded dataset saved to {output_jsonl}")
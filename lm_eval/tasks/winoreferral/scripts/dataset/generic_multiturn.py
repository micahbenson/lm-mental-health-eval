import json
import sys

def merge_json_files(file1_path, file2_path, output_path):
    results = []
    for path in (file1_path, file2_path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.pop("assistant_text", None)
                results.append(item)

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Merged {len(results)} records into {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge.py <file1.jsonl> <file2.jsonl> <output.jsonl>")
        sys.exit(1)
    merge_json_files(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"saved file to {sys.argv[3]}")
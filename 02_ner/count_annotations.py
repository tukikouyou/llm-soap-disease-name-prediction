"""Count the annotated entities in a Label Studio JSON export."""
import json

def count_entities_from_label_studio_json(file_path):
    """Sum the length of every annotations -> result list in the export."""
    total_entities = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for record in data:
                if 'annotations' in record and isinstance(record['annotations'], list):
                    for annotation_set in record['annotations']:
                        if 'result' in annotation_set and isinstance(annotation_set['result'], list):
                            total_entities += len(annotation_set['result'])

    except FileNotFoundError:
        print(f"Error: file not found: '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    return total_entities

if __name__ == "__main__":
    file_name = "project-10-at-2025-08-19-16-32-0033594e.json"

    count = count_entities_from_label_studio_json(file_name)

    if count is not None:
        print(f"Total number of entities in '{file_name}': {count}")

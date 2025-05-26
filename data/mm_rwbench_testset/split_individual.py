import json
import os
import re

# --- Configuration ---
INPUT_FILENAME = 'all_data.json'
OUTPUT_DIR = './individual/' # New directory for transformed output
# --- End Configuration ---

def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    name = name.replace('/', '_').replace('\\', '_')
    name = re.sub(r'[<>:"|?*]', '', name)
    if not name:
        name = 'unnamed_category'
    return f"{name}.json"

def transform_and_split_json(input_file, output_dir):
    """
    Reads JSON, transforms paired outputs into individual items with preference labels,
    splits by category, and writes separate files.
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        if not isinstance(original_data, list):
            print(f"Error: Input file '{input_file}' does not contain a JSON list.")
            return

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured.")

        # List to hold the transformed data (individual output items)
        transformed_data = []
        processed_count = 0
        skipped_count = 0

        print("Processing items: Transforming paired outputs into individual items...")
        # Process each item: transform and create new individual items
        for item in original_data:
            # Rename 'Better' to 'label' if it exists, standardizing the preference key
            if 'Better' in item:
                item['label'] = item.pop('Better')

            # Check if required fields for transformation are present
            required_keys = ['Output1', 'Output2', 'label', 'Category', 'ID', 'Text'] # Add other essential keys if needed
            if not all(key in item for key in required_keys):
                print(f"Warning: Skipping item ID '{item.get('ID', 'Unknown')}' due to missing required keys (Output1/Output2/label/Category/ID/Text).")
                skipped_count += 1
                continue

            preferred_output_key = item['label'] # e.g., "Output1" or "Output2"

            # Validate the preference label value
            if preferred_output_key not in ["Output1", "Output2"]:
                 print(f"Warning: Skipping item ID '{item.get('ID', 'Unknown')}' due to invalid 'label' value: '{preferred_output_key}'. Expected 'Output1' or 'Output2'.")
                 skipped_count += 1
                 continue

            # Create base for new items (copy common fields)
            base_new_item = {
                "Category": item['Category'],
                "ID_original": item['ID'], # Keep original ID for reference
                "Text": item['Text'],
                # Add other common fields you want to keep, like 'Image'
                **({'Image': item['Image']} if 'Image' in item else {})
            }

            # Create item for Output1
            new_item1 = base_new_item.copy()
            new_item1['ID'] = item['ID'] + "_output1" # Create a new unique ID
            new_item1['Output'] = item['Output1']
            new_item1['label'] = "preferred" if preferred_output_key == "Output1" else "non-preferred"
            transformed_data.append(new_item1)

            # Create item for Output2
            new_item2 = base_new_item.copy()
            new_item2['ID'] = item['ID'] + "_output2" # Create a new unique ID
            new_item2['Output'] = item['Output2']
            new_item2['label'] = "preferred" if preferred_output_key == "Output2" else "non-preferred"
            transformed_data.append(new_item2)

            processed_count += 1

        print(f"\nProcessed {processed_count} original items into {len(transformed_data)} individual items.")
        if skipped_count > 0:
             print(f"Skipped {skipped_count} original items due to missing/invalid data.")

        # --- Grouping and Writing ---
        # Dictionary to hold transformed data grouped by category
        categories_data = {}

        # Group transformed data by category
        for t_item in transformed_data:
            category = t_item['Category']
            if category not in categories_data:
                categories_data[category] = []
            categories_data[category].append(t_item)

        # Write each category to a separate file
        print(f"\nFound {len(categories_data)} categories. Writing files...")
        for category, items in categories_data.items():
            output_filename = sanitize_filename(category)
            output_filepath = os.path.join(output_dir, output_filename)

            try:
                # Write the list of transformed items for this category
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(items, f, indent=4, ensure_ascii=False)
                print(f"Successfully wrote {len(items)} items to '{output_filepath}'")
            except IOError as e:
                print(f"Error writing file '{output_filepath}': {e}")
            except Exception as e:
                 print(f"An unexpected error occurred while writing '{output_filepath}': {e}")

        print("\nProcessing complete.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{input_file}'. Check its format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the script ---
if __name__ == "__main__":
    transform_and_split_json(INPUT_FILENAME, OUTPUT_DIR)
# --- End Script ---
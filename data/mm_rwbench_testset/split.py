import json
import os
import re

# --- Configuration ---
INPUT_FILENAME = 'all_data.json'
OUTPUT_DIR = './' # Directory to save the split files
# --- End Configuration ---

def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    # Replace slashes and backslashes with underscores
    name = name.replace('/', '_').replace('\\', '_')
    # Remove other potentially problematic characters (you can adjust this regex)
    name = re.sub(r'[<>:"|?*]', '', name)
    # Ensure the name is not empty after sanitization
    if not name:
        name = 'unnamed_category'
    return f"{name}.json"

def split_json_by_category_and_rename(input_file, output_dir):
    """
    Reads a JSON file, renames the 'Better' field to 'label' in each item,
    splits the data by the 'Category' field, and writes each category
    to its own JSON file in the output directory.
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: Input file '{input_file}' does not contain a JSON list.")
            return

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured.")

        # Dictionary to hold data grouped by category
        categories_data = {}

        # Process each item: rename field and group by category
        print("Processing items: Renaming 'Better' to 'label' and grouping...")
        for item in data:
            # --- Modification Start ---
            # Rename the 'Better' field to 'label' if it exists
            if 'Better' in item:
                item['label'] = item.pop('Better') # pop removes 'Better' and returns its value
            # --- Modification End ---

            # Group data by category
            if 'Category' in item:
                category = item['Category']
                # If category key doesn't exist, create it with an empty list
                if category not in categories_data:
                    categories_data[category] = []
                # Append the modified item (with 'label' instead of 'Better')
                categories_data[category].append(item)
            else:
                print(f"Warning: Item found without 'Category' key: {item.get('ID', 'Unknown ID')}")
                # Optionally handle items without a category (e.g., skip or put in a default file)
                # For now, we'll skip them
                continue

        # Write each category to a separate file
        print(f"\nFound {len(categories_data)} categories. Writing files...")
        for category, items in categories_data.items():
            # Sanitize category name to create a valid filename
            output_filename = sanitize_filename(category)
            output_filepath = os.path.join(output_dir, output_filename)

            try:
                # Write the list of modified items for this category to a JSON file
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    # Use indent=4 for pretty printing
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
    # Changed the function call to the updated one
    split_json_by_category_and_rename(INPUT_FILENAME, OUTPUT_DIR)
# --- End Script ---
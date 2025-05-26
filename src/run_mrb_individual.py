# Ensure these imports cover all needed functions from your project files
from .utils import * # Should contain mllm_encode, mllm_classify_with_counts, open_data
from .model import * # Should contain load_model
from .preprocess import * # May contain other helpers
from tqdm import tqdm
import torch
import os
import argparse
import json # Added for potentially saving detailed results

# Disable gradient calculations
torch.set_grad_enabled(False)
# Suppress verbose logging from transformers if desired
from transformers.utils import logging
logging.set_verbosity_error()



def eval_preference_dataset_embedding_only(args):
    """
    Evaluates preference prediction accuracy using head embeddings.
    Assumes 'label' field contains "preferred" or "non-preferred".
    Uses original mllm_encode and mllm_classify_with_counts.
    """
    print(f"Loading model: {args.model_name}")
    model = load_model(args.model_name, args.data_name) # Assumes data_name helps configure model helper if needed

    # Load data - assumes open_data loads the individual format
    # where 'label' contains "preferred" or "non-preferred".
    print(f"Loading training data from: {args.train_path} (using up to first {args.encode_n_train} for encoding)")
    train_data = open_data(args.data_name, args.train_path)[:args.encode_n_train] # Use arg for N train samples

    print(f"Loading validation data from: {args.val_path}")
    test_data = open_data(args.data_name, args.val_path)
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} validation samples.")

    if not test_data:
        print("Error: No validation data loaded. Exiting.")
        return
    if not train_data:
         print("Error: No training data loaded for encoding. Exiting.")
         return

    # --- Embedding-based Evaluation Logic ---
    print("\nRunning evaluation using preference embeddings...")
    print(f"Encoding training data using {args.num_heads} heads based on 'label' field ('preferred'/'non-preferred')...")
    # Use the original mllm_encode, assuming it groups by item['label']
    # The result now represents preference embeddings because the label field holds preference info.
    try:
        preference_embeddings = mllm_encode(model, train_data, num_head=args.num_heads)
        # Basic validation of the returned structure
        if not all(k in preference_embeddings for k in ['activations', 'top_heads', 'int_to_str']):
             raise ValueError("mllm_encode did not return the expected dictionary structure.")
        print("Encoding complete.")
    except Exception as e:
        print(f"Error during mllm_encode: {e}")
        print("Cannot proceed with evaluation.")
        return


    predictions = []
    correct_count = 0 # Raw accuracy count
    results_details = [] # Optional: store detailed results

    print("Classifying validation samples...")
    # Process test data
    num_skipped = 0
    for i, item in enumerate(tqdm(test_data, desc="Evaluating Samples")):
        # Ensure the item has the required 'label' key and valid value
        if 'label' not in item or item['label'] not in ["preferred", "non-preferred"]:
             print(f"Warning: Skipping item {i} due to missing or invalid 'label'. Value: {item.get('label', 'MISSING')}")
             num_skipped += 1
             continue

        # Classify using the preference embeddings and the classifier that returns counts
        # Assumes mllm_classify_with_counts is available
        try:
            pred_label, head_votes = mllm_classify_with_counts(item, model, preference_embeddings)
        except Exception as e:
            print(f"Error during classification for item {i} (ID: {item.get('ID', 'N/A')}): {e}")
            pred_label = None # Treat classification error as failed prediction
            head_votes = {}

        ground_truth_label = item['label'] # Ground truth is in the 'label' field now

        if pred_label is not None:
            is_correct = (pred_label == ground_truth_label)
            correct_count += is_correct
            predictions.append(pred_label)
            # Store details if needed for analysis
            results_details.append({
                "id": item.get("ID", i),
                "prediction": pred_label,
                "ground_truth": ground_truth_label,
                "is_correct": is_correct,
                "head_votes": head_votes,
                "output_text": item.get("Output", None) # Add output text for context
            })
        else:
             # Prediction failed (returned None or exception occurred)
             print(f"Warning: Prediction failed or skipped for item {i} (ID: {item.get('ID', 'N/A')}).")
             num_skipped += 1 # Count failed predictions as skipped for accuracy calculation

    # Calculate overall accuracy based on successfully processed items
    num_evaluated = len(test_data) - num_skipped
    if num_evaluated > 0:
        accuracy = correct_count / num_evaluated
    else:
        accuracy = 0.0
        print("Warning: No items were successfully evaluated.")


    eval_type = f"Preference Embedding ({args.num_heads} heads)"
    print(f"\n--- {eval_type} Metrics ---")
    print(f"Items Evaluated: {num_evaluated} (Skipped/Failed: {num_skipped})")
    print(f"Correct Predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")

    # --- Optional: Save detailed results ---
    if args.save_details:
        detail_base = os.path.splitext(os.path.basename(args.val_path))[0]
        detail_filename = f"{args.model_name}_{detail_base}_embed_{args.num_heads}h_details.json"
        detail_filepath = os.path.join('results', detail_filename)
        output_directory = os.path.dirname(detail_filepath)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        try:
            with open(detail_filepath, "w") as f:
                json.dump(results_details, f, indent=2)
            print(f"Detailed results saved to {detail_filepath}")
        except Exception as e:
            print(f"Error saving detailed results: {e}")
    # --- End Optional Save ---


    # --- Save Overall Accuracy Result ---
    print("\nSaving accuracy score...")
    # Construct the full file path for the summary accuracy score
    base_filename = os.path.splitext(os.path.basename(args.val_path))[0]
    eval_suffix = f"embed_pref_{args.num_heads}h" # Indicate embedding-based preference eval
    output_filename = f"{args.model_name}_{base_filename}_{eval_suffix}_accuracy.txt"
    output_filepath = os.path.join('results', output_filename)

    # Ensure the output directory exists
    output_directory = os.path.dirname(output_filepath)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    # Write the accuracy score
    try:
        with open(output_filepath, "w") as writefile:
            writefile.write(f"{accuracy:.4f}\n") # Save accuracy score
        print(f"Accuracy score saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving accuracy score: {e}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Evaluate MLLM preference prediction using embeddings.")
   # Removed --eval_zeroshot argument
   parser.add_argument("--model_name", type=str, default="qwen2.5_vl", help="Name of the model to load.")
   parser.add_argument("--data_name", type=str, default="mrb_individual", help="Identifier for the dataset (e.g., indicating individual scoring format where 'label' holds preference).")
   parser.add_argument("--train_path", type=str, required=True, help="Path to the training data JSON file (individual format, 'label'='preferred'/'non-preferred').")
   parser.add_argument("--val_path", type=str, required=True, help="Path to the validation data JSON file (individual format, 'label'='preferred'/'non-preferred').")
   parser.add_argument("--num_heads", type=int, default=20, help="Number of top heads to use for embedding-based evaluation.")
   parser.add_argument("--encode_n_train", type=int, default=20, help="Number of training samples to use for calculating preference embeddings.")
   parser.add_argument("--save_details", action="store_true", help="Save detailed prediction results to a JSON file.")

   args = parser.parse_args()

   # --- Basic Input Validation ---
   if not os.path.exists(args.train_path):
        print(f"Error: Training path not found: {args.train_path}")
        exit(1)
   if not os.path.exists(args.val_path):
        print(f"Error: Validation path not found: {args.val_path}")
        exit(1)
   if args.encode_n_train <= 0:
        print(f"Warning: encode_n_train is {args.encode_n_train}. Using at least 1 sample for encoding.")
        args.encode_n_train = 1
   # --- End Validation ---

   eval_preference_dataset_embedding_only(args) # Call the renamed evaluation function
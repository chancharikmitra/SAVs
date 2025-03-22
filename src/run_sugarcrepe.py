from .utils import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 

def eval_dataset(args):
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)
    
    multimodal_embeddings = mllm_encode(model, train_data, num_head=20)
    
    correct = 0  # Count of correct predictions
    total_pairs = len(test_data) // 2  # Total number of image-caption pairs
    
    # Process data in pairs (positive and negative caption for each image)
    for i in tqdm(range(0, len(test_data), 2)):
        pos_item = test_data[i]      # Positive caption
        neg_item = test_data[i+1]    # Negative caption
        
        # Verify that both items reference the same image
        assert pos_item['image'] == neg_item['image'], f"Mismatch in image paths at index {i}"
        
        # Verify that positive caption has "Yes" label and negative has "No"
        assert pos_item['label'] == "Yes" and neg_item['label'] == "No", f"Label mismatch at index {i}"
        
        if args.eval_zeroshot:
            # Zero-shot evaluation
            pos_input = model.insert_image(pos_item['question'], [pos_item['image']])
            neg_input = model.insert_image(neg_item['question'], [neg_item['image']])
            
            pos_pred = model.generate(pos_input, max_new_tokens=1).strip()
            neg_pred = model.generate(neg_input, max_new_tokens=1).strip()
        else:
            # Regular embedding-based evaluation
            pos_pred = mllm_classify(pos_item, model, multimodal_embeddings)
            neg_pred = mllm_classify(neg_item, model, multimodal_embeddings)
        
        print(f'Pos Pred: {pos_pred}, Label: {pos_item["label"]}')
        print(f'Neg Pred: {neg_pred}, Label: {neg_item["label"]}')
        
        # Count as correct only if both predictions are correct
        if pos_pred == pos_item['label'] and neg_pred == neg_item['label']:
            correct += 1

    # Calculate accuracy
    accuracy = correct / total_pairs
    
    eval_type = "Zero-shot" if args.eval_zeroshot else "SUGARCREPE"
    print(f"\n{eval_type} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct pairs: {correct}/{total_pairs}")
    
    # Additional metrics can be added here based on SUGARCREPE requirements
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_ov")
    parser.add_argument("--data_name", type=str, default="sugarcrepe")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_zeroshot", action="store_true",
                       help="Whether to run zero-shot evaluation")
    
    args = parser.parse_args()
    eval_dataset(args)
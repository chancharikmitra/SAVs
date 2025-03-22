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
   predictions = []
   q_correct = 0  # Question accuracy count
   i_correct = 0  # Image accuracy count  
   g_correct = 0  # Group accuracy count
   correct = 0 # Raw accuracy count
   total_groups = len(test_data) // 4  # Total number of groups
   
   # Process data in groups of 4
   for i in tqdm(range(0, len(test_data), 4)):
       group = test_data[i:i+4]
       group_preds = []
       
       # Get predictions for the group
       # print(i)
       for item in group:
           if args.eval_zeroshot:
               # Zero-shot evaluation
               model_input = model.insert_image(item['question'], [item['image']])
               pred = model.generate(model_input, max_new_tokens=1).strip()
           else:
               # Regular embedding-based evaluation
               pred = mllm_classify(item, model, multimodal_embeddings)
               
           print(f'Pred {pred} Label {item["label"]}')
           group_preds.append(pred == item['label'])
       
       # Question accuracy (first two and second two must match)
       if group_preds[0] and group_preds[1]:
           q_correct += 1
       if group_preds[2] and group_preds[3]:
           q_correct += 1
           
       # Image accuracy (first and third, second and fourth must match)
       if group_preds[0] and group_preds[2]:
           i_correct += 1
       if group_preds[1] and group_preds[3]:
           i_correct += 1
           
       # Group accuracy (all four must be correct)
       if all(group_preds):
           g_correct += 1

       # Raw accuracy
       correct += sum(group_preds)

   # Calculate percentages
   q_acc = q_correct / (total_groups * 2)  # Two questions per group
   i_acc = i_correct / (total_groups * 2)  # Two images per group
   g_acc = g_correct / total_groups        # One group accuracy score per group
   acc = correct / (total_groups * 4)      # Accuracy calculated per sample
   
   eval_type = "Zero-shot" if args.eval_zeroshot else "NaturalBench"
   print(f"\n{eval_type} Metrics:")
   print(f"Question Accuracy (Q-Acc): {q_acc:.4f}")
   print(f"Image Accuracy (I-Acc): {i_acc:.4f}") 
   print(f"Group Accuracy (G-Acc): {g_acc:.4f}")
   print(f"Raw Accuracy: {acc:.4f}")

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--model_name", type=str, default="llava_ov")
   parser.add_argument("--data_name", type=str, default="natural_ret")
   parser.add_argument("--train_path", type=str, default=None)
   parser.add_argument("--val_path", type=str, default=None)
   parser.add_argument("--eval_zeroshot", action="store_true",
                      help="Whether to run zero-shot evaluation")
   
   args = parser.parse_args()
   eval_dataset(args)

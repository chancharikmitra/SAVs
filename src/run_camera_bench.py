from .utils import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import os
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
    #    q_correct = 0  # Question accuracy count
    #    i_correct = 0  # Image accuracy count  
    #    g_correct = 0  # Group accuracy count
    correct = 0 # Raw accuracy count
    #    total_groups = len(test_data) // 4  # Total number of groups

    # Process data in groups of 4
    for i in tqdm(range(0, len(test_data))):
        item = test_data[i]

        if args.eval_zeroshot:
            # Zero-shot evaluation
            model_input = model.insert_image(item['question'], [item['image']])
            pred = model.generate(model_input, max_new_tokens=1).strip()
        else:
            # Regular embedding-based evaluation
            pred = mllm_classify(item, model, multimodal_embeddings)
                
        # print(f'Pred {pred} Label {item["label"]}')
        correct += (pred == item['label'])
        # print(correct)
       

    # Calculate percentage
    acc = correct / len(test_data)
    eval_type = "Zero-shot" if args.eval_zeroshot else "CameraBench"
    print(f"\n{eval_type} Metrics:")
    print(f"Raw Accuracy: {acc:.4f}")

    with open('results/' + args.model_name + "_"+ os.path.basename(args.val_path)+ "_resultpath.txt", "w") as writefile:
        writefile.write(str(acc))

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

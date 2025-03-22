from baukit import TraceDict, get_module
from .model import *
from .preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import sys
from collections import Counter
logging.set_verbosity_warning()

def load_model(model_name, cur_dataset, lora_path=None):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """

    if model_name == "llava_ov":
        from llava.model.builder import load_pretrained_model
        
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"

        
        model_name = "llava_qwen"
        device_map = "auto"
        llava_model_args = {
                "multimodal": True,
                # "image_aspect_ratio":"pad"
            }
        ###For finetuned models

        # overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
        # overwrite_config = {}
        # overwrite_config["image_aspect_ratio"] = "pad"
        # llava_model_args["overwrite_config"] = overwrite_config


        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        #tokenizer, model, image_processor, max_length = load_pretrained_model("/home/zhaobin/LLaVA-NeXT/checkpoints/Mhalu_sft", pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        # ###TODO:DELETE, FOR BLINK
        # tokenizer, model, image_processor, max_length = load_pretrained_model(lora_path, pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        model.eval()
        model.requires_grad_(False)
        model_helper = llavaOVHelper(model, tokenizer, image_processor, cur_dataset)
    elif model_name == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained( "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")

        model.eval()
        model.requires_grad_(False)
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        model_helper = Qwen2Helper(model, processor, cur_dataset)

    return model_helper


def gather_last_attn_activations(inputs, model_helper):

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    ###retain_input means the activation before passing to o_proj, which is the out projection matrix after attention. retain_out means after passing to o_proj
    ###You can decide to use something other than the o_proj by passing different config to layers. Refer to model.py
    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        #result = model_helper.generate(inputs, max_new_tokens=32)
        result = model_helper.forward(inputs)
    return td, result


def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """


    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False, split="train"):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """

    activation_storage = None

    for n in range(N_TRIALS):

        #text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper, split=split)
        text, image_list, _, _ = model_helper.format_func(None, dataset[0], num_shot=shot, model_helper=model_helper, split=split)

        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        ###Extracting only the activation of the last input_token, as seen in the -1 indexing
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)

    return mean_activations


def get_class_activations(train_dataset, model, attn_heads):

    str_to_int = {}
    int_to_str = {}
    str_to_activation = {}
    str_to_count = {}
    save_act = {}


    for item in tqdm(train_dataset):

        mean_activations = get_last_mean_head_activations([item], model, N_TRIALS = 1, shot=0)
        head_act = []
        for head in attn_heads:
            head_act.append(mean_activations[head[0], head[1], -1])
        head_act = torch.stack(head_act)

        if item['label'] in str_to_activation.keys():
            save_act[item['label']] += [head_act]
            str_to_activation[item['label']] += head_act
            str_to_count[item['label']] += 1
        else:
            save_act[item['label']] = [head_act]
            str_to_activation[item['label']] = head_act
            int_label = len(str_to_activation.keys()) - 1
            str_to_int[item['label']] = int_label
            int_to_str[int_label] = item['label']
            str_to_count[item['label']] = 1
    
    avg_activations = []
    
    for key, item in str_to_activation.items():
        save_act[key] = torch.stack(save_act[key], dim=0)
        # torch.save(save_act[key], 'pets/' + str(key) + '.pt')
        # print(f'Key {key} save_act shape {save_act[key].shape}')
        avg_activations.append(torch.div(item, str_to_count[key]))
    avg_activations = torch.stack(avg_activations)
    return avg_activations, str_to_int, int_to_str


def get_query_activations(query_input, model_helper, common_heads):

    mean_activations = get_last_mean_head_activations(query_input, model_helper, N_TRIALS = 1, shot=0)
    head_act = []
    for head in common_heads:

        head_act.append(mean_activations[head[0], head[1], -1])

    head_act = torch.stack(head_act)
    
    return head_act


def record_head_performance(sample_activations, cur_activation, label, success_count):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)
        all_sample.append(scores.argmax(dim=0).item())
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1


def retrieve_examples(sample_activations, cur_activation):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []

    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    most_common = counter.most_common()

    chosen_examples = []
    for item in most_common:
        chosen_examples.append(item[0])
    return chosen_examples


def mllm_encode(model, train_data, num_head):

    ###Step 1: Extract mean activation for each class label
    all_heads = model.all_heads
    #(class, head_count, resid_dim)
    print('\nExtract Mean Activations\n')
    class_activations, str_to_int, int_to_str = get_class_activations(train_data, model, all_heads)
    print(f'Class Activations {class_activations.shape} Str to int {str_to_int} int to str {int_to_str}')
    success_count = [0 for _ in range(class_activations.shape[1])]

    print('\nSelect Top Sparse Heads\n')
    for item in tqdm(train_data):

        query_activations = get_query_activations([item], model, all_heads).squeeze(dim=0)
        int_label = str_to_int[item['label']]
        record_head_performance(class_activations, query_activations, int_label, success_count)


    ###Step 2: Pick the top-num_head based on their classification performance
    arr = np.array(success_count)
    # Get the indices of the top-k elements.
    k = num_head
    topk_indices = np.argsort(arr)[-k:][::-1]

    top_heads = []
    print("Printing Top Heads and their classification accuracy")
    for item in topk_indices.tolist():
        print(item, success_count[item])
        top_heads.append(all_heads[item])

    print("\nGet Top Heads' Activations \n")
    top_class_activations, str_to_int, int_to_str = get_class_activations(train_data, model, top_heads)
    print(f'actications {top_class_activations.shape}')
    print(f'top heads {top_heads}')
    print(f'int_to_str {int_to_str}')
    return {"activations":top_class_activations, "top_heads": top_heads, "int_to_str":int_to_str}


def mllm_classify(inputs, model, class_embed):

    cur_activations = get_query_activations([inputs], model, class_embed['top_heads']).squeeze(dim=0)
    top_k_examples = retrieve_examples(class_embed['activations'], cur_activations)
    cur_int_label = top_k_examples[0]
    return class_embed['int_to_str'][cur_int_label]




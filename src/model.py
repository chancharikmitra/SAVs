from PIL import Image
import torch
import copy
import requests
from .preprocess import *

from decord import VideoReader, cpu


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from qwen_vl_utils import process_vision_info


def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except:
        return image_file
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class ModelHelper:
    def __init__(self):
        """
        self.model: The loaded model
        self.tokenizer: The loaded tokenizer
        self.processor: The image processor/transform
        self.model_config: The architecture of the model. Should include:
            - n_heads: Number of attention heads
            - n_layers: Number of layers
            - resid_dim: Hidden size
            - name_or_path: Model name or path
            - attn_hook_names: List of attention output projection hook names
            - layer_hook_names: List of layer hook names
            - mlp_hook_names: List of MLP projection hook names
        self.format_func: The format function for the current dataset
        self.space: Whether the model output will have a leading space
        self.cur_dataset: Name of the current dataset
        self.split_idx: The index of "layer" when you parse "attn_hook_names" with "."
        self.nonspecial_idx: The index in which the generated tokens are not special token
        self.all_heads: Optional list of (layer, head, -1) tuples for attention analysis
        """
        pass

    def insert_image(self, text, image_list):
        """
        Returns an object that is the input to forward and generate.
        Args:
            text: The input text/prompt
            image_list: List of image paths/URLs
        Returns:
            tuple: (input_ids, image_tensors, image_sizes)
        """
        pass

    def forward(self, model_input, labels=None):
        """
        Forward function wrapper
        Args:
            model_input: Tuple from insert_image
            labels: Optional labels for loss computation
        Returns:
            model output
        """
        pass

    def generate(self, model_input, max_new_tokens):
        """
        Generate function wrapper
        Args:
            model_input: Tuple from insert_image
            max_new_tokens: Maximum number of tokens to generate
        Returns:
            str: Generated text
        """
        pass

class llavaOVHelper(ModelHelper):

    def __init__(self, model, tokenizer, processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}

        self.cur_dataset = cur_dataset
        self.format_func = get_format_func(cur_dataset)
        self.split_idx = 2

        self.all_heads = []
        for layer in range(28):
            for head in range(28):
                self.all_heads.append((layer, head, -1))


    # def insert_image(self, text, image_list):

    #     conv_template = "qwen_1_5"
    #     conv = copy.deepcopy(conv_templates[conv_template])
    #     conv.append_message(conv.roles[0], text)
    #     conv.append_message(conv.roles[1], None)
    #     prompt_question = conv.get_prompt()


    #     input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

    #     if image_list == []:
    #         return (input_ids, None, None)

    #     image_list = load_images(image_list)
    #     image_sizes = [image.size for image in image_list]

    #     image_tensors = process_images(image_list, self.processor, self.model.config)
    #     image_tensors = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensors]

    #     return (input_ids, image_tensors, image_sizes)

    
    def load_video(self, video_path, num_frames=16):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        sampled_frames = vr.get_batch(frame_idx).asnumpy()
        return sampled_frames

    def insert_image(self, text, image_list):
        # Prepare conversation template
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Process text input
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

        # Return early if no images/videos
        if not image_list:
            return (input_ids, None, None, None)

        # Process all images/videos
        processed_data = []
        image_sizes = []
        modalities = []
        
        for image_path in image_list:
            if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Video processing
                video_frames = self.load_video(image_path)
                frames_tensor = self.processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(self.model.device)
                processed_data.append(frames_tensor)
                # For videos, store frame-level image sizes
                frame_sizes = [frames_tensor.shape[2:] for _ in range(frames_tensor.shape[0])]
                image_sizes.append(frame_sizes)
                modalities.append("video")
            else:
                # Regular image processing
                image = Image.open(image_path).convert('RGB')
                image_tensor = process_images([image], self.processor, self.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensor]
                processed_data.append(image_tensor[0])
                image_sizes.append(image_tensor[0].shape[1:])
                modalities.append(None)
        
        # Only set modalities to None if we don't have any videos
        has_videos = any(m == "video" for m in modalities)
        return (input_ids, processed_data, image_sizes, modalities if has_videos else None)
    

    def forward(self, model_input, labels=None):

        result = self.model(model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            labels=labels,
            modalities=model_input[3])
        return result
    

    def generate(self, model_input, max_new_tokens):

        cont = self.model.generate(
            model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            modalities=model_input[3]
        )
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

class Qwen2Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.nonspecial_idx = 0
        self.question_lookup = None

        self.all_heads = []
        for layer in range(28):
            for head in range(28):
                self.all_heads.append((layer, head, -1))

    def insert_image(self, text, image_list):

        if image_list[0].lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": img} for img in image_list
                ] + [
                    {"type": "text", "text": text}
                ]
            }
        ]
            formatted_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            inputs = self.processor(
            text=[formatted_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            fps=8.0,
            return_tensors="pt"
            ).to("cuda")
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img} for img in image_list
                    ] + [
                        {"type": "text", "text": text}
                    ]
                }
            ]
            formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
            image_inputs, video_inputs = process_vision_info(messages)

        

            inputs = self.processor(
                text=[formatted_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

        
        return inputs

    def forward(self, model_input, labels=None):

        # input_ids = model_input["input_ids"].to(self.model.device)
        # attention_mask = model_input["attention_mask"].to(self.model.device)

        result = self.model(
            **model_input
        )  

        return result

    def generate(self, model_input, max_new_tokens):

    
        generated_output = self.model.generate(
            **model_input, max_new_tokens=max_new_tokens, do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_input.input_ids, generated_output)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
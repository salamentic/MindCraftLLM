import argparse
from models.image_text_transformation import ImageTextTransformation
from utils.util import display_images_and_text
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src', default='examples/1.jpg')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=False, help='Set this flag to True if you want to use semantic segmentation')
    parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_b', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
    parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, <6G GPU is not recommended>')
    args = parser.parse_args()
    processor = ImageTextTransformation(args)
    import glob
    for game_path in sorted(glob.glob("/home/anrao/MindCraft/experiments/Frames_60/main_logs/*")):
        print("NEW GAME", game_path, len(glob.glob(game_path+"/play1*/*.png")))
        frame_texts=[]
        if os.path.exists("image2textMindcraft/"+game_path.split("/")[-1]+".pkl"):
            continue
        for file in sorted((glob.glob(game_path+"/play1*/*.png"))):
            image_src = file
            generated_text = processor.image_to_text(image_src)
            frame_texts.append(generated_text)
            print("Generated Text:")
            print(generated_text)
        with open("image2textMindcraft/"+game_path.split("/")[-1]+".pkl", 'wb') as frames_data_file:
            pickle.dump(frame_texts, frames_data_file)
    results = display_text(args.image_src, generated_text, args.out_image_name)

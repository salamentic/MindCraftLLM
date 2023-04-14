import os

input_folder = "Dataset"
output_folder = "Frames_10"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for dirpath, dirnames, filenames in os.walk(input_folder):
    # Create an output directory for each input directory
    output_dir = os.path.join(output_folder, os.path.relpath(dirpath, input_folder))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in filenames:
        if filename.endswith(".gif"):
            input_file = os.path.join(dirpath, filename)
            gif_folder = os.path.splitext(filename)[0]
            output_subdir = os.path.join(output_dir, gif_folder)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_prefix = os.path.join(output_subdir, gif_folder)
            os.system(f"ffmpeg -i {input_file} -vf \"select=not(mod(n\,10))'\" -vsync vfr -q:v 2 {output_prefix}_%04d.png")

from sentence_transformers import SentenceTransformer
import os
import numpy as np
from PIL import Image

def get_image_files_from_folder(folder_path, extensions=['.jpg', '.png', '.jpeg']):
    image_files = []
    for file_name in os.listdir(folder_path):
        # Check if the file has one of the specified image extensions
        if any(file_name.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(folder_path, file_name)
            image_files.append(file_path)
    return image_files



image_directory = 'cropped'

images = get_image_files_from_folder(image_directory)


paths = [os.path.basename(path) for path in images]
np.save('vectors/names.npy', paths)


model = SentenceTransformer('clip-ViT-B-32')

images = [Image.open(path) for path in images]
#images = [Image.open(path).convert('L') for path in images]
embeddings = model.encode(images)

np.save('vectors/embeddings.npy', embeddings)
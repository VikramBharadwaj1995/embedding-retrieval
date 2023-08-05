import faiss
import numpy as np


# Build an index
index = faiss.IndexFlatL2(512)  # L2 distance metric


import os

# Specify the directory containing the images
vector_directory = 'vectors'  # Replace with the path to your image directory


embeddings = np.load(vector_directory + '/embeddings.npy')
index.add(embeddings)


names = np.load(vector_directory + '/names.npy')
query_names = {x: i for i, x in enumerate(names)}
print(query_names)


# Perform a nearest neighbor search
k = 7  # Number of nearest neighbors to retrieve
search_entry = "6_1_552a43d8-b73e-4d54-8ea6-96cf6bc31ed5.jpg" # Name of mask to search for
query_index = query_names[search_entry]
query = index.reconstruct(query_index).reshape(1, -1)

distances, indices = index.search(query, k)


for i in range(index.ntotal):
    vector = index.reconstruct(i)
    #index = np.where(embeddings == vector)
    row_indices, col_indices = np.where(np.all(embeddings == vector, axis=1)[:, np.newaxis])
    i = row_indices[0]
    print("Index: {:<3}     File Name: {}".format(i,names[i]))

# Print the query vector and nearest neighbors
print("\nQuery: " + search_entry)
print("\nNearest neighbors:")
for i in range(k):
    print("Index: {:<3}     Distance: {:<20}    Name: {}".format(indices[0][i], distances[0][i], names[indices[0][i]]))


from PIL import Image

canvas_width = k*400
canvas_height = 600
canvas = Image.new('RGB', (canvas_width, canvas_height))

pos = 0
for i in indices[0]:
    image_name = names[i]
    underscore_index = image_name.find('_')
    original_image_name = image_name[underscore_index + 1:]

    image = Image.open('images' + '/' + original_image_name)
    image = image.resize((400,600))
    canvas.paste(image, (pos*400, 0))
    pos+=1
        
canvas.show()
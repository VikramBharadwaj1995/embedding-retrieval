import cv2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

class BoundingBoxDrawer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.bounding_box_coords = None
        self.draw_bounding_box()

    def draw_bounding_box(self):
        self.image = cv2.imread(self.image_path)
        self.clone_image = self.image.copy()
        cv2.imshow("Draw Bounding Box", self.image)
        cv2.setMouseCallback("Draw Bounding Box", self.on_mouse_click)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Press 'r' to reset the bounding box
                self.image = self.clone_image.copy()
                self.bounding_box_coords = None
                cv2.imshow("Draw Bounding Box", self.image)
            elif key == ord("c"):  # Press 'c' to confirm the bounding box and close the window
                break

        cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bounding_box_coords = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.bounding_box_coords.extend([x, y])
            self.draw_rectangle()

    def draw_rectangle(self):
        if len(self.bounding_box_coords) == 4:
            x1, y1, x2, y2 = self.bounding_box_coords
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Draw Bounding Box", self.image)

if __name__ == "__main__":

    index = faiss.IndexFlatL2(512)

    vector_directory = 'vectors'
    image_directory = 'images'

    embeddings = np.load(vector_directory + '/embeddings.npy')
    index.add(embeddings)
    names = np.load(vector_directory + '/names.npy')
    query_names = {x: i for i, x in enumerate(names)}



    # Replace 'image.jpg' with the name of your image
    image_name = '' # Set image name here
    image_path = image_directory + '/' + image_name

    bbox_drawer = BoundingBoxDrawer(image_path)
    x1, y1, x2, y2 = bbox_drawer.bounding_box_coords
    if x1>x2:
        x1,x2 = x2,x1
    if y1>y2:
        y1,y2 = y2,y1

    image = cv2.imread(image_path)
    mask = np.zeros_like(image)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    image = cv2.bitwise_and(image, mask)

    #cv2.imshow("Image",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    model = SentenceTransformer('clip-ViT-B-32')
    query = model.encode(image).reshape(1, -1)

    k = 7
    distances, indices = index.search(query, k)

    for i in range(index.ntotal):
        vector = index.reconstruct(i)
        row_indices, col_indices = np.where(np.all(embeddings == vector, axis=1)[:, np.newaxis])
        i = row_indices[0]
        print("Index: {:<3}     File Name: {}".format(i,names[i]))

    # Print the query vector and nearest neighbors
    print("\nQuery: " + image_path)
    print("\nNearest neighbors:")
    for i in range(k):
        print("Index: {:<3}     Distance: {:<20}    Name: {}".format(indices[0][i], distances[0][i], names[indices[0][i]]))


    canvas_width = k*400
    canvas_height = 600
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    pos = 0
    for i in indices[0]:
        image_name = names[i]
        underscore_index = image_name.find('_')
        original_image_name = image_name[underscore_index + 1:]

        image = Image.open(image_directory + '/' + original_image_name)
        image = image.resize((400,600))
        canvas.paste(image, (pos*400, 0))
        pos+=1
        
    canvas.show()
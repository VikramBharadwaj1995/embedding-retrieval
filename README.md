# Image Retrieval with Segment Anything, CLIP, and FAISS

This repository contains code and resources for performing image retrieval using Segment Anything, CLIP, and FAISS. The combination of these three powerful tools allows for efficient and accurate image search based on image embeddings.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vikrambharadwaj1995/embedding-retrieval
   ```
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Preparing the dataset:
  Organize your image dataset in a suitable directory structure.
  If you want to use pre-trained models, download them and place them in the appropriate directories. Also download the pretrained models from the respective model zoo.

3. Extracting image embeddings:
  Use the mask_generator.py script to segment objects in the images and extract embeddings. Adjust the script parameters according to your needs. Use the bbox_search.py file if you want to get embeddings only insdie a particular bounding box.

4. Indexing the embeddings:
   Use the embedding.py to create an index that stores the image name and the respective embedding associated with it(this is helpful during the retriveal process).
  Run the mask_search.py script to build an index using the extracted embeddings. This step utilizes FAISS for efficient similarity search.

# Pose Matching through Time Dynamic Warping of Confidence-Weighted Cosine Similarity

## Data
- Examples of movements in JSON frames (each JSON file is a frame) can be found in the `data/raw` folder
- Pre-processed JSON files containing the coresponding pose coordinates will be found in the `data/posed` folder

## Usage
1. ```git clone https://github.com/yourslab/pose-matching```
2. ```import compare_dir from .pose-matching```
3. ```compare_dir('/path/to/json_frames_a', 'path/to/json_frames/b')```
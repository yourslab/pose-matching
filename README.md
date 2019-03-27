# Pose Matching through Dynamic Time Warping of Confidence-Weighted Cosine Similarity

## Input
- Examples of movements in JSON frames (each JSON file is a frame) can be found in the `data/raw` folder
- Pre-processed JSON files containing the coresponding pose coordinates will be found in the `data/posed` folder

## Usage
1. Clone this repo into the directory it will be imported from
```git clone https://github.com/yourslab/pose-matching```
2. Import the compare_dir wrapper function 
```import compare_dir from pose-matching```
3. Each parameter in the compare_dir function takes as input a folder containing JSON frames outputted by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
```compare_dir('/path/to/json_frames_folder_a', 'path/to/json_frames_folder_b')```

## Credits
- [OpenPose: Real-time multi-person keypoint detection library for body, face, hands, and foot estimation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from sklearn import preprocessing
from fastdtw import fastdtw
from scipy import interpolate
from operator import add

def extract_coords(frame):
	x = np.array(frame[0::3])
	y = np.array(frame[1::3])
	return x, y

# Get euclidean distance between frame a, b
def frame_euc_dis(a, b):
	x1, y1 = extract_coords(a)
	x2, y2 = extract_coords(b)
	dist = np.sum((x1 - x2)**2 + (y1 - y2)**2)
	return dist

# Get cosine similarity between frame a, b
def frame_cos_dis(a, b):
	x1, y1 = extract_coords(a)
	x2, y2 = extract_coords(b)
	a_vec = []
	b_vec = []
	for x, y in zip(x1, y1):
		a_vec.append(x)
		a_vec.append(y)
	for x, y in zip(x2, y2):
		b_vec.append(x)
		b_vec.append(y)
	X = np.asarray([a_vec,
        b_vec], dtype=np.float)
	X_normalized = preprocessing.normalize(X, norm='l2')
	dist = cosine(X_normalized[0,:],X_normalized[1,:])
	#dist = np.dot(a_vec, b_vec)
	return dist

def normalize(frames):
	return preprocessing.normalize(frames, norm='l2')

# Interpolates frames to have num_desired frames
# Needed for DTW to compare differently sized videos
def interpolate_frames(frames, num_desired):
	old_indices = np.arange(0, len(frames))
	# Interpolate the new set of indices depending on num_desired
	new_indices = np.linspace(0, len(frames)-1, num_desired)
	new_frames = []
	# Break up the frames
	for coord in frames.T:
		# Each coordinate is interpolated independently
		f = interpolate.interp1d(old_indices, coord)
		new_coord = f(new_indices)
		new_frames.append(new_coord)
	return np.array(new_frames).T

def remove_confidences(frames):
	return [[coord for i, coord in enumerate(frame) if (i+1)%3 != 0] for frame in frames]

def get_confidences(frames):
	return [[coord for i, coord in enumerate(frame) if (i+1)%3 == 0] for frame in frames]

def get_centroid(frames, coord):
	if coord == 'x':
		coords = np.array([[coord for i, coord in enumerate(frame) if i%3 == 0] for frame in frames])
	else:
		coords = np.array([[coord for i, coord in enumerate(frame) if i%3 == 1] for frame in frames])
	# Return the mean coord
	return np.sum(np.sum(coords, axis=0))/len(X)

def get_first_hip(frames, coord):
	if coord == 'x':
		return frames[0][24]
	else:
		return frames[0][25]

def translate_video(frames, x_offset, y_offset):
	return [[(coord-x_offset) if i%2==0 else (coord-y_offset) for i, coord in enumerate(frame)] for frame in frames]

def compare_videos(X, Y):
	# Interpolate the shorter video to length of longer video
	if len(X) > len(Y):
		Y = interpolate_frames(Y, len(X))
	elif len(Y) > len(X):
		X = interpolate_frames(X, len(Y))
	X = normalize(np.array(X))
	Y = normalize(np.array(Y))
	dist, path = fastdtw(X, Y, dist=cosine)
	return dist

def json_to_np(directory):
	# Each entry contains a json array of pose coordinates
	videos = []
	video_dirs = os.listdir(directory)
	for video_dir in video_dirs:
		video = []
		for json_file in os.listdir('{}/{}'.format(directory, video_dir)):
			with open('{}/{}/{}'.format(directory, video_dir, json_file)) as f:
				frame = json.loads(f.read())
				try:
					video.append(frame['people'][0]['pose_keypoints_2d'])
				except IndexError:
					print(json_file)
		x_offset = get_first_hip(video, 'x')
		y_offset = get_first_hip(video, 'y')
		translated = translate_video(remove_confidences(video), x_offset, y_offset)
		#print(translated)
		videos.append(translated)
	return videos

POS_DIR = 'data/posed/floss'
NEG_DIR = 'data/posed/not-floss'

pos_videos = json_to_np(POS_DIR)
neg_videos = json_to_np(NEG_DIR)

print(compare_videos(np.array(pos_videos[2]), np.array(neg_videos[0])))
#dist = frame_cos_dis(videos[0][0]['pose_keypoints_2d'], videos[1][0]['pose_keypoints_2d'])
#print(dist)

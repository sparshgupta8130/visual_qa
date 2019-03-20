import pandas as pd
import os
from PIL import Image
import random
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def random_image(output_csv, image_dir, ans_file, idx=None):
	data_all = pd.read_csv(output_csv, sep=",", header=None)
	data_all.columns = ["img_id", "ques", "ans_id"]
	if idx is not None:
		ind = idx
	else:
		ind = random.randint(0, len(data_all))

	data_split = image_dir.split('/')[-1]
	image_filename = 'COCO_' + data_split + '_000000' + str(data_all["img_id"].ix[ind]).zfill(6) + '.jpg'
	image_path = os.path.join(image_dir, image_filename)
	image = Image.open(image_path).convert(mode="RGB")

	question = data_all["ques"].ix[ind]

	answer = ""
	aid_pred = data_all["ans_id"].ix[ind]
	f = open(ans_file, "rb")
	aid_dict = pickle.load(f)
	for ans, aid in aid_dict.items():
		if aid == aid_pred: 
			answer = ans

	return image_filename, image, question, answer

def result_to_jpg(image_filename, image, question, answer, ):
	fig = plt.figure()
	plt.imshow(np.asarray(image))
	plt.axis('off')
	txt = question + '? ' + answer
	fig.text(0.5, .05, txt, ha='center')
	plt.savefig(os.path.join('results/corr', image_filename))
	plt.close(fig)

if __name__ == "__main__":
	pics_list = [53016, 3960, 30548, 10435, 54290, 3974, 20, 1344, 2315, 2932, 778, 1746, 2340, 704, 8880, 47633]
	for i in range(len(pics_list)):
		image_filename, image, question, answer = random_image('evaluations/2019-03-20__21h34m07s_AttentionModel_1000__corr.csv', 'datasets/VQAimages/val2014', 'datasets/aid_ans_1000.pickle', pics_list[i])
		create_directory('./results/corr/')
		result_to_jpg(image_filename, image, question, answer)

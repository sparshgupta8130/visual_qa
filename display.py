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

def random_image(output_csv, image_dir, ans_file):
	data_all = pd.read_csv(output_csv, sep=",", header=None)
	data_all.columns = ["img_id", "ques", "ans_id"]
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
	plt.savefig(os.path.join('results/incorr', image_filename))
	plt.close(fig)

if __name__ == "__main__":
	for i in range(100):
		image_filename, image, question, answer = random_image('evaluations/2019-03-13__08h03m22s_JointEmbedModel_1000__incorr.csv', 'datasets/VQAimages/val2014', 'datasets/aid_ans_1000.pickle')
		create_directory('./results/incorr/')
		result_to_jpg(image_filename, image, question, answer)
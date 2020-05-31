from PIL import Image
import os, sys
import numpy as np

cat_dict = {}
candiate = []
IMAGE_WIDTH_NUM = 4
if __name__ == "__main__":
	base_dir = sys.argv[1]
	root, dirs, files = list(os.walk(base_dir))[0]
	for file in files:
		if file.endswith(".jpg"):
			cat = file.split("label_")[1].split("_pred_")[0]
			if cat in cat_dict:
				continue
			cat_dict[cat] = 1
			candiate.append(file)

print(len(candiate), candiate)
candiate = np.array(candiate)
candiate = candiate.reshape(-1, IMAGE_WIDTH_NUM)

all_imgs = []
for i in range(candiate.shape[0]):
	row = [np.asarray(Image.open(os.path.join(root, file))) for file in list(candiate[i, :])]
	all_imgs.append(np.concatenate(row, 1))

final_img = np.concatenate(all_imgs, 0)
img = Image.fromarray(final_img)
dump_path = os.path.join(os.path.dirname(os.path.dirname(root)), "large.png")
print(dump_path)
img.save(dump_path)





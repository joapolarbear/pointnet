import matplotlib.pyplot as plt
import sys, os

def read_data(_file):
	with open(_file, 'r') as fp:
		logs = fp.read().split("\n")


	batch_list = []
	loss_list = []
	accuracy_list = []

	epoch_list = []
	epoch_accuracy_l = []
	epoch_loss_l = []
	'''
		106.278540: **** EPOCH 003 ****
		110.533075: Batch 972  , mean loss: 1.06      , accuracy: 0.69      
		115.896235: Batch 1036 , mean loss: 1.12      , accuracy: 0.68      
		121.230702: Batch 1100 , mean loss: 1.17      , accuracy: 0.67      
		126.563334: Batch 1164 , mean loss: 1.09      , accuracy: 0.69      

		131.879054: Batch 1228 , mean loss: 1.07      , accuracy: 0.70      
		140.125749: Batch 1228 , eval mean loss: 1.00      , eval accuracy: 0.70      , eval avg class acc: 0.64
	''' 
	for log in logs:
		if "EPOCH" in log:
			epoch_list.append(int(log.split("EPOCH")[1].split("****")[0]))
		elif "eval" in log:
			epoch_loss_l.append(float(log.split("eval mean loss: ")[1].split(",")[0]))
			epoch_accuracy_l.append(float(log.split("eval accuracy: ")[1].split(",")[0]))
		elif "Batch" in log:
			log_split = log.split(",")
			batch_list.append(int(log_split[0].split("Batch")[1]))
			loss_list.append(float(log_split[1].split("mean loss:")[1]))
			accuracy_list.append(float(log_split[2].split("accuracy:")[1]))

	return {
		"batch_list": batch_list,
		"loss_list": loss_list,
		"accuracy_list": accuracy_list,
		"epoch_list": epoch_list,
		"epoch_loss_l": epoch_loss_l,
		"epoch_accuracy_l": epoch_accuracy_l
	}

if __name__ == "__main__":
	# file = sys.argv[1]
	# base_path = "/Users/hhp/0/git/pointnet"
	base_path = sys.argv[1]
	file_name = ["data/log/rotation_tnet/true_true.txt", 
				"data/log/rotation_tnet/true_false.txt",
				"data/log/rotation_tnet/false_false.txt",
				"data/log/rotation_tnet/false_true.txt"]
	name_list = ["w/ Rotation, w/ T-net", "w/ Rotation, w/o T-net",
				 "w/o Rotation, w/o T-net", "w/o Rotation, w T-net"]
	files = [os.path.join(base_path, n) for n in file_name]
	datas = [read_data(file) for file in files]
	
	plt.figure(num=4, figsize=(8, 6))

	### iteration to loss
	ax = plt.subplot(221)
	for i, data in enumerate(datas):
		ax.plot(data["batch_list"], data["loss_list"],  
			label=name_list[i])
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
	plt.legend(ncol=2, bbox_to_anchor=(1, 1.2), loc='center')
	plt.xlabel('a)\n# of Iteration')
	plt.ylabel('Training Loss')

	### iteration to accuracy
	ax = plt.subplot(222)
	for i, data in enumerate(datas):
		ax.plot(data["batch_list"], data["accuracy_list"],  
			label=name_list[i])
	plt.ylim(0, 1.2)
	plt.xlabel('b)\n# of Iteration')
	plt.ylabel('Training Accuracy')

	ax = plt.subplot(223)
	for i, data in enumerate(datas):
		ax.plot(data["epoch_list"], data["epoch_loss_l"],  
			label=name_list[i])
	plt.xlabel('c)\n# of Epoch')
	plt.ylabel('Evaluation Loss')

	ax = plt.subplot(224)
	for i, data in enumerate(datas):
		ax.plot(data["epoch_list"], data["epoch_accuracy_l"],  
			label=name_list[i])
	plt.ylim(0, 1.2)
	plt.xlabel('d)\n# of Epoch')
	plt.ylabel('Evaluation Accuracy')

	plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.4)
	plt.show()

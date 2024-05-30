from torch.utils.data import Dataset
import imageio
import os
import torchvision
import torch
import torchvision.transforms as v2

transform = v2.Compose([
    v2.ToTensor() # convert the image to a tensor with values between 0 and 1
])

def read_text_file(filename):
	lines = []
	with open(filename, 'r') as file:
		for line in file: 
			line = line.strip() #or some other preprocessing
			lines.append(line)
	return lines


class FixationDataset(Dataset):
	def __init__(self, root_dir, image_file, fixation_file, image_transform=None, fixation_transform=None):
		self.root_dir = root_dir
		self.image_files = read_text_file(image_file) # will return a list of strings
		self.fixation_files = read_text_file(fixation_file) # will return a list of strings
		self.image_transform = image_transform
		self.fixation_transform = fixation_transform
		assert len(self.image_files) == len(self.fixation_files), "lengths of image files and fixation files do not match!"

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.image_files[idx]) # access array location idx, which is a string, location of the image
		image = imageio.imread(img_name)

		fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
		fix = imageio.imread(fix_name)

		sample = {"image": image, "fixation": fix, "raw_image": torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)}

		if self.image_transform:
			sample["image"] = self.image_transform(sample["image"])
		if self.fixation_transform:
			sample["fixation"] = self.fixation_transform(sample["fixation"])

		return sample
	
if __name__ == "__main__":
	dataset = FixationDataset(root_dir="cv2_project_data", image_file="cv2_project_data/train_images.txt", 
						   fixation_file="cv2_project_data/train_fixations.txt", image_transform=transform, fixation_transform=transform)
	
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

	for img in dataloader:
		print(img["image"].shape, img["fixation"].shape, img["image"].dtype, img["fixation"].dtype)

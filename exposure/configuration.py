import os, random, shutil
import torch

from PIL import Image

base_dir = '/content/drive/MyDrive/fasterrccn_project/'

# Major architecture configurations

batch_size = 16
width, height = 224, 224
num_epochs = 25
num_workers = 2
learning_rate = 0.0001
gamma = 0.1
mean = [0.485, 0.456, 0.406]
sd = [0.229, 0.224, 0.225]

# Detection threshold representative of IoU

detection_threshold = 0.8
frame_count = 0
total_fps = 0

# Effective labels 0 designated as background, 1 designated as 'windTurbine'

classes = ["background", "windTurbine"]
num_classes = len(classes)
visualise_transformed_images = False

# Base directory comprising of entire dataset

def custom_path(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

drive_dir = '/content/drive/MyDrive/'

base_dir = os.path.join(drive_dir, 'frcnn_project')
custom_path(base_dir)

zip_file = os.path.join(drive_dir, 'windTurbineDataSet_xml_annotations.zip')
archive_format = "zip"
shutil.unpack_archive(zip_file, base_dir, archive_format)

img_path = os.path.join(base_dir, 'windTurbineDataSet/JPEGImages')
annot_path = os.path.join(base_dir, 'windTurbineDataSet/Annotations')

# Corrupt image file removal function

def corrupt_image_removal(img_path):
    for filename in os.listdir(img_path):
        if filename.endswith('.png'):
            try:
                img = Image.open(img_path + filename)
                img.verify()
            except(IOError, SyntaxError) as e:
                print(filename)
                os.remove(img_path + filename)


# corrupt_image_removal(img_path=train_dir)

custom_dir = os.path.join(base_dir, 'custom')
custom_path(custom_dir)

output_dir = os.path.join(base_dir, 'outputs')
custom_path(output_dir)

full_data = os.path.join(custom_dir, 'full_data')
train_dir = os.path.join(custom_dir, 'train')
val_dir = os.path.join(custom_dir, 'val')
test_dir = os.path.join(custom_dir, 'test')

custom_path(full_data)
custom_path(train_dir)
custom_path(val_dir)
custom_path(test_dir)

os.makedirs(full_data, exist_ok=True)

png_files = [f for f in os.listdir(img_path) if f.endswith('.png')]

for png_file in png_files:
    xml_file = os.path.splitext(png_file)[0] + '.xml'

    if os.path.exists(os.path.join(annot_path, xml_file)):
        shutil.move(os.path.join(img_path, png_file), os.path.join(full_data, png_file))
        shutil.move(os.path.join(annot_path, xml_file), os.path.join(full_data, xml_file))
        print(f"Moved {png_file} and {xml_file} to {full_data}")

# Specify and construct Train, Val and Test sub-diretories

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

png_files = [f for f in os.listdir(full_data) if f.endswith('.png')]

total_files = len(png_files)
train_size = int(0.7 * total_files)
val_size = test_size = int(0.15 * total_files)

random.shuffle(png_files)

train_files = png_files[:train_size]
val_files = png_files[train_size:train_size + val_size]
test_files = png_files[train_size + val_size:]

for files, dest_directory in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:

    for png_file in files:
        xml_file = os.path.splitext(png_file)[0] + '.xml'
        shutil.move(os.path.join(full_data, png_file), os.path.join(dest_directory, png_file))
        shutil.move(os.path.join(full_data, xml_file), os.path.join(dest_directory, xml_file))
        print(f"Moved {png_file} and {xml_file} to {dest_directory}")

print("Data split complete.")


# Refresh output for new training

def delete_previous_output(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


delete_previous_output(folder=output_dir)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.is_available()

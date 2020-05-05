import os
import shutil

train_dir = "/home/abi-osler/Documents/CV_final_project/DeepScoresClassification"
dest_dir = '/home/abi-osler/Documents/CV_final_project/final_project/template_matching_dataset'
counter = 0

for subdir, dirs, files in os.walk(train_dir):
    print(files)
    for file in files:
        full_path = os.path.join(subdir, file)
        shutil.copy(full_path, dest_dir)
        counter = counter + 1
print(counter)
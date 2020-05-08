# Music-Object-Classification

## Introduction

Music Object Classification is a part of Optical Music Recognition (OMR) where we task the computer to filter out and detect 
music symbols or objects in music score sheet. In this project, I have simplified the problem down to classifying individual 
musicial objects. I gathered a small subset of the DeepScores dataset, which had collection of annotated labels for 118 musical 
objects. I will outline the steps below to recreate this project on your machine.   

## Set up project folder 
Git clone the project or download and unzip in machine. This will be the local project folder. 

## Downloading the dataset

This project is based in the DeepScores dataset. [DeepScores](https://tuggeluk.github.io/downloads/) is a collection of datasets 
that musical scores in digitized format. It has datasets for both simple object classification and segmentation. We only need the 
dataset for the object classification for this problem. In the website for DeepScores, download the DeepScores-Classfication dataset
along with class-names.csv. First, move the class-names.csv in the dataset folder, next move dataset folder into your project folder.

## Setting up a virtual environment and dependencies

It is recommended to setup a virual environment. Follow steps below to do this.  

First install python3 if not install already. Then run the bash command on terminal below to install virtual-env.

```bash
sudo apt install python3-venv 

```
Next, switch to your project folder, You can now create the custom virtual env with the following command 

```bash
python3 -m venv my-project-env

```
Now, active the virtual environment with the following command. 

```bash
source my-project-env/bin/activate 

```
Next install pip if not installed. Then execute the following command to install all dependencies for the project 

```bash
pip install -r requirements.txt

```

## Setting up the scripts. 
Make sure to point the data_path variable to the path of the DeepScores-Classfication folder. The rest should work fine.  

## Testing 

There are three main scripts in this project. dataset_reader.py processes the dataset cleaning training and testing. The first
major script is detection.py. I have implemented a simple template matching using pearson correlation coefficients. I have set 
up the possible experiments that you can try with following commands. 

For one-to-one matching with randomized templates on a  on a batch_size of 1000 test samples execute the following: 

```bash 

 python detection.py --test_mode="one_to_one" --batch_size=1000 --randomized=False

```
For finding best match with randomized templates on a  on a batch_size of 1000 test samples execute the following: 

```bash 

 python detection.py --test_mode="best_match" --batch_size=1000 --randomized=False

```

Set the flag ```bash --randomized``` to ```bash False ``` if you want to test it out on averaged templates.  

Next, I have implemented a simple Convolutional Neural Network (CNN) using the Keras framework. The parameters 

for training can be tweaked with the following lines in model.py. 

```python  
# traning parameters 
batch_size = 10
num_classes = 118 
epochs = 100
```
Feel free to play around with them and train the network locally on your machine.  

## Reference 

I have heavily borrowed and modified code from the following references for this project 

[Keras Documentation example for CNN](https://keras.io/examples/cifar10_cnn/) 
[DeepScores test repository](https://github.com/tuggeluk/DeepScoresExamples) 
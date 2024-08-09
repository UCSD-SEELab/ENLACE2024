
# Human Activity Data Collection, Analysis and LLM-based Zero-Shot Reasoning
Hello reader! In this repository, you will find the codes necessary to decompress and save the information in a CSV file collected by a smartphone, the visualization and the statistical analysis of the data, and the Python code to call **ChatGPT 3.5 turbo**.

## Introduction
Our project aims to recognize human activities performed by multiple subjects for their subsequent prediction based on Zero-shot reasoning. The future goal is to obtain a system for health monitoring, aging care, and human behavior analysis.

## Method
Before processing sensor data, we strongly recommend taking into consideration that the project is missing several folders where data should be stored. So, for the scripts to work, three folders named *"uploaded_data"*, *"unpacked_data"* and *"CSVs"* should be added, each with folders for the subject, which should be named *"subject_(subject number)"* (e.g., x *"subject_1"*). This set of files will be used for data extraction and will be used throughout the project. The subject folders in "uploaded_data" should be filled with the data folder named after the UUID of each subject. For example, the directory should be named:

"C:\Users\John\Desktop\projects\ESS_test\uploaded_data\subject_1\6ec0bd7f-11c0-43da-975e-2a8ad9ebae0b”

The work published in ["Introduction to the ExtraSensory Dataset"](http://extrasensory.ucsd.edu/intro2extrasensory/intro2extrasensory.html#features) was taken as a reference to read the user’s data file, the analysis of the context labels, and the sensor features. 
The process for preparing the IMU raw dataset is described below:
-	Dataset collection
-	Data analysis
-	Feature extraction on sensor dataset
-	Zero-shot reasoning and prompt input

The process for preparing the IMU raw dataset is described below:
1. Dataset collection
To begin, the data recorded in the ExtraSensory app is collected and stored on its corresponding server. Once the data is obtained, the data is unzipped, processed, and saved in a folder with a Universally Unique Identifier (UUID).
2. Data analysis
The labels users reported are graphed to visualize activities over time and identify the main activities. [(link for data visualization examples)](https://docs.google.com/document/d/1bjD8qwGdPfZHh_1Ug0iqQT3fHHz1MVIR2ICpm1isey0/edit?usp=sharing) 
3. Feature extraction on sensor dataset
In this part, the features were calculated from the most complete raw measurements from the various sensors of the smartphone, especially the IMU device. Sensor features are named with a prefix to analyze them under a statistical analysis that shows the recurrence and distribution of information over time.
4. Zero-shot reasoning and prompt input
Finally, after obtaining the raw IMU data and identifying the activities with the most information collected from each subject, ChatGPT 3.5 Turbo is called from Python to predict the activities by varying the configuration settings, and the characteristics of the information provided in the prompt.

> [!NOTE] 
> Many of the functions used in the codes require downloading some functions and packages to run the code. Keep this in mind.

## How to run the code?
Once the repository has been downloaded and the correct naming of the subjects to be analyzed, the complete code requires the following running sequence:
* To get data collection: 
```
csv_zipper.py
```

* To label, process, and analyze the data:
```
label_visualizer.py
```

* To extract the raw sensor features:
```
feature_visualizer.py
```

* To make the ChatGPT prediction:
```
ai_predict.py
```

## Result
To obtain the best accuracy and F1-score, which were our main metrics, various tests were performed with different sensor features, considering both raw data and data on which statistical analysis was applied. Besides, the **statistical values** reduce input size for GPT and increase accuracy.

Sensor combination with the best results is Acc, Gyro, and Magnet (**56.5%** accuracy with raw data, **58.8%** accuracy with statistic data)

# 

Made by Audrick Bolaños, Daniel Ortiz, Diego Muñiz, and Marena Schiess in 2024.

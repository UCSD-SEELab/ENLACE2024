import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime  # Import the datetime module for manipulating dates and times
import pytz  # Import the pytz module for time zone calculations
# ------------------------------------------ Reading user data ----------------------------------------------------
import numpy as np  # Numerical operations on arrays
import gzip  # Which is used for reading and writing gzip-compressed files
from io import BytesIO  # Method that manipulate string and bytes data in memory
import sklearn.linear_model
from ess_utils import convert_extrasensory_dataset_label_to_standard_network_label
from ESS_test.chatGPT import *
import tqdm
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

def parse_header_of_csv(csv_str):  # Takes a string csv_str containing CSV data
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index(b'\n')].decode('utf-8')
    columns = headline.split(',') # Extracts the first line of the CSV string. Splits this line by commas into a list called columns.
    # The first column should be timestamp: (Asserts both columns have the corresponding name)
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    try:
        assert columns[-1] == 'label_source\r', 'Older'
        assert columns[-1] == 'label_source', 'Newer'
    except AssertionError as e:
        print (f'{str(e)} dataset active')
    
    # Search for the column of the first label:  (Iterates through the column names to find the index of the first column that starts with 'label:'
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci  # Stores the index in first_label_ind and breaks out of the loop once found.
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]  # Extracts the feature column names (those between 'timestamp' and the first label)
    # Then come the labels, till the one-before-last column:
    label_names= columns[first_label_ind:-1]# Extracts the label column names (those from the first label to the one-before-last column
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:','')  # Iterates through label_names to remove the 'label:' prefix from each label name
        pass
    
    return (feature_names,label_names)  # Tuple

def parse_body_of_csv(csv_str,n_features):  # Defines a function named parse_body_of_csv that takes a CSV string
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(BytesIO(csv_str),delimiter=',',skiprows=1)  # Reads the CSV data (excluding the header) into a NumPy array.
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int)  # Extracts the first column (timestamps) and converts it to integers
    
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)]  # Extracts the feature columns into the matrix X
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # Extracts the label columns into trinary_labels_mat (values of either 0., 1. or NaN)
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix. Creates a boolean matrix M indicating where labels are missing (NaN).
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix. Converts the label matrix to a binary matrix Y where NaN values are treated as 0.
    
    return (X,Y,M,timestamps)  # Returns the feature matrix X, label matrix Y, missing label matrix M, and timestamps

def read_user_data(uuid,subject_num):  # Defines a function named read_user_data that takes a user ID uuid
    user_data_file = 'csvs/%s/%s.features_labels.csv.gz' % (subject_num,uuid)  # Constructs the filename for the user's data file based on the uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:  # Opens the gzip-compressed CSV file and reads its contents into csv_str
        csv_str = fid.read()
        pass

    (feature_names,label_names) = parse_header_of_csv(csv_str)  # Parses the header of the CSV to get the feature and label names.
    n_features = len(feature_names)  # 
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features)  # Parses the body of the CSV to get the feature matrix X, label matrix Y, missing label matrix M, and timestamps

    return (X,Y,M,timestamps,feature_names,label_names)  # Returns the extracted data and metadata.

# ----------------------------------------- Read data for a particular user ----------------------------------------
#(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid,subject_num)

#print("User %s has %d examples (Folders with collected data)" % (uuid,len(timestamps),))

# Count non-zero sensor features
#non_zero_sensors = np.sum(np.any(X != 0, axis=0))
#print("The primary data files have %d different sensor-features, of which %d have information." % (len(feature_names), non_zero_sensors))

# Count non-zero context labels
#non_zero_labels = np.sum(np.any(Y != 0, axis=0))
#print("The primary data files have %d context-labels, of which %d have information." % (len(label_names), non_zero_labels))

def get_timestamps_with_label(Y, timestamps, label_names, target_label):
    label_index = label_names.index(target_label)
    relevant_timestamps = timestamps[Y[:, label_index] == 1] 
    return relevant_timestamps

def combine_timestamps(target_timestamps, mistake_timestamps):
    combined_timestamps = []
    mistake_index = 0
    
    for i in range(len(target_timestamps)):
        combined_timestamps.append(target_timestamps[i])
        if (i + 1) % 2 == 0 and mistake_index < len(mistake_timestamps):
            combined_timestamps.append(mistake_timestamps[mistake_index])
            mistake_index += 1
    
    return combined_timestamps

def calculate_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    return mean, std_dev, variance

def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    return accuracy, precision, recall, f1

def load_mfcc_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        data = []
        for line in lines:
            try:
                row = list(map(float, line.strip().rstrip(',').split(',')))
                data.append(row)
            except ValueError as e:
                print(f"Error processing line: {line.strip()}\n{e}")
        
        data = np.array(data)
        
        if data.size == 0:
            raise ValueError("No valid data found in the file.")
        
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def reshape_mfcc_data(data):
    if data is None or data.size == 0:
        print("No data to reshape.")
        return None

    try:
        num_rows = data.shape[0]
        reshaped_data = data.reshape(num_rows, 13)
        return reshaped_data
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return None

def plot_spectrogram(data, Hz=430/20):
    if data is None or data.size == 0:
        print("No data to plot.")
        return
    
    try:
        num_samples = data.shape[0]
        duration = num_samples / Hz
        
        plt.figure(figsize=(12, 8))
        
        plt.imshow(data.T, aspect='auto', origin='lower', cmap='jet', norm=mcolors.Normalize(vmin=-4, vmax=10))
        
        plt.colorbar(label='MFCC Value')
        plt.xlabel('Time (seconds)')
        plt.xticks(ticks=np.linspace(0, num_samples, num=5), labels=np.round(np.linspace(0, duration, num=5), 2))
        plt.ylabel('MFCC Channels')
        plt.title('audio')
        
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the spectrogram: {e}")

def read_dat_file_for_plot(timestamp, dat_names_, subject_num, uuid):
    file_path = os.path.join('unpacked_data', subject_num, uuid, str(timestamp), dat_names_)
    data = np.loadtxt(file_path)
    if data.size <= 20 or np.isnan(data).all():
        print(f"No valid data in file: {file_path}")
        return None, None, None, None, None, None, None, None, None
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    
    Hz_full = 40
    timestamps_full = np.arange(len(x)) / Hz_full

    total_seconds = len(x) / Hz_full
    print(f"Total second available in the file {dat_names_}: {total_seconds:.2f}s")

    Hz_options = ['40 Hz', '20 Hz', '10 Hz', '5 Hz', '2 Hz']

    Hz_int = [40, 20, 10, 5, 2]

    time_options = ['20s', '15 s', '10 s', '5 s', 'all']

    time_int = [20, 15, 10, 5]

    while True:
        print("List of Frequencies:")
        for i, Hz_val in enumerate(Hz_options, start=1):
            print(f"{i}. {Hz_val}")

        Hz_selected = input(f"Select a Frequencie (1-{len(Hz_options)}): ")

        if Hz_selected.isdigit() and 1 <= int(Hz_selected) <= len(Hz_options):
            Hz = Hz_int[int(Hz_selected) - 1]
            print(f"Selected Frecuencie: {Hz} Hz")
            break
        print("Invalid frequency. Please try again.")
    
    while True:
        print(f"Total second available in the file {dat_names_}: {total_seconds:.2f}s")
        print("List of Times:")
        for i, time_val in enumerate(time_options, start=1):
            print(f"{i}. {time_val}")

        time_selected = input(f"Select a Time (1-{len(time_options)}): ")
        if time_selected.isdigit() and 1 <= int(time_selected) <= len(Hz_options):
            if int(time_selected) == len(time_options):
                time_duration = int(total_seconds)
            else:
                time_duration = time_int[int(time_selected) - 1]
            if time_duration <= total_seconds:
                print(f"Selected Time: {time_duration} s")
                break
        print("Invalid time duration. Please try again.")

    step = Hz_full // Hz
    
    x_reduced = np.mean(x[:Hz*time_duration*step].reshape(-1, step), axis=1)
    y_reduced = np.mean(y[:Hz*time_duration*step].reshape(-1, step), axis=1)
    z_reduced = np.mean(z[:Hz*time_duration*step].reshape(-1, step), axis=1)

    Hz_reduced = Hz
    timestamp_reduced = np.arange(len(x_reduced)) / Hz_reduced

    return timestamps_full, x, y, z, timestamp_reduced, x_reduced, y_reduced, z_reduced, Hz_reduced

def plot_dat_file(timestamps_list, target_label, uuid, subject_num):
    
    flower = 0
    
    for timestamp in timestamps_list:
        
        if flower==5:
            break
        
        print('Timestamp')
        print(timestamp)
        
        acc_data = read_dat_file_for_plot(timestamp, 'm_raw_acc.dat', subject_num, uuid)
        gyro_data = read_dat_file_for_plot(timestamp, 'm_raw_gyro.dat', subject_num, uuid)  
        magnet_data = read_dat_file_for_plot(timestamp, 'm_raw_magnet.dat', subject_num, uuid)
        
        if any(data is None for data in acc_data) or any(data is None for data in gyro_data) or any(data is None for data in magnet_data):
            print(f"Skipping timestamp {timestamp} due to no valid data.")
            continue
        
        mfcc_file = os.path.join('unpacked_data', subject_num, uuid, str(timestamp), 'sound.mfcc')
        
        mfcc_data = load_mfcc_file(mfcc_file)
        audio_data = reshape_mfcc_data(mfcc_data)
        
        if audio_data is None or audio_data.size == 0:
            print("No data to plot.")
            audio_data = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        axs[0, 0].plot(acc_data[0], acc_data[1], label='x')
        axs[0, 0].plot(acc_data[0], acc_data[2], label='y')
        axs[0, 0].plot(acc_data[0], acc_data[3], label='z')
        axs[0, 0].set_title('Acceleration (40 Hz)')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Acceleration')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(acc_data[4], acc_data[5], label='x')
        axs[0, 1].plot(acc_data[4], acc_data[6], label='y')
        axs[0, 1].plot(acc_data[4], acc_data[7], label='z')
        axs[0, 1].set_title(f'Acceleration ({acc_data[8]} Hz)')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Acceleration')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        axs[1, 0].plot(gyro_data[0], gyro_data[1], label='x')
        axs[1, 0].plot(gyro_data[0], gyro_data[2], label='y')
        axs[1, 0].plot(gyro_data[0], gyro_data[3], label='z')
        axs[1, 0].set_title('Gyroscope (40 Hz)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Angular Velocity')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(gyro_data[4], gyro_data[5], label='x')
        axs[1, 1].plot(gyro_data[4], gyro_data[6], label='y')
        axs[1, 1].plot(gyro_data[4], gyro_data[7], label='z')
        axs[1, 1].set_title(f'Gyroscope ({gyro_data[8]} Hz)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Angular Velocity')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.suptitle(f'{target_label} - Timestamp: {timestamp} - {subject_num}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.4)
        plt.show()
        
        fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))
        
        axs2[0].plot(magnet_data[0], magnet_data[1], label='x')
        axs2[0].plot(magnet_data[0], magnet_data[2], label='y')
        axs2[0].plot(magnet_data[0], magnet_data[3], label='z')
        axs2[0].set_title('Magnetometer (40 Hz)')
        axs2[0].set_xlabel('Time (s)')
        axs2[0].set_ylabel('Magnetic Field Strength')
        axs2[0].legend()
        axs2[0].grid(True)
        
        axs2[1].plot(magnet_data[4], magnet_data[5], label='x')
        axs2[1].plot(magnet_data[4], magnet_data[6], label='y')
        axs2[1].plot(magnet_data[4], magnet_data[7], label='z')
        axs2[1].set_title(f'Magnetometer ({magnet_data[8]} Hz)')
        axs2[1].set_xlabel('Time (s)')
        axs2[1].set_ylabel('Magnetic Field Strength')
        axs2[1].legend()
        axs2[1].grid(True)
        
        plt.suptitle(f'Magnetometer Data - Timestamp: {timestamp}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.4)
        plt.show()
        
        plot_spectrogram(audio_data)
        
        flower += 1
    
    return

def read_dat_file_for_AI(timestamp, dat_names_, subject_num, uuid):
    file_path = os.path.join('unpacked_data', subject_num, uuid, str(timestamp), dat_names_)
    data = np.loadtxt(file_path)
    if data.size <= 40 or np.isnan(data).all():
        print(f"No valid data in file: {file_path}")
        return None, None, None, None, None, None, None, None, None
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    
    Hz_full = 40
    timestamps_full = np.arange(len(x)) / Hz_full

    total_seconds = len(x) / Hz_full
    
    Hz = 10
    
    time_duration = 10
    
    print(f"Total second available in the file {dat_names_}: {total_seconds:.2f}s")
    
    step = max(1, Hz_full // Hz)
    
    num_samples = len(x)
    remainder = num_samples % step
    if remainder > 0:
        x = x[:-remainder]
        y = y[:-remainder]
        z = z[:-remainder]
    
    try:
        x_reduced = np.mean(x[:Hz*time_duration*step].reshape(-1, step), axis=1)
        y_reduced = np.mean(y[:Hz*time_duration*step].reshape(-1, step), axis=1)
        z_reduced = np.mean(z[:Hz*time_duration*step].reshape(-1, step), axis=1)
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return timestamps_full, x, y, z, None, None, None, None, Hz_full, None, None, None, None, None, None, None, None, None

    Hz_reduced = Hz
    timestamp_reduced = np.arange(len(x_reduced)) / Hz_reduced
    
    x_mean, x_std, x_var = calculate_statistics(x)
    y_mean, y_std, y_var = calculate_statistics(y)
    z_mean, z_std, z_var = calculate_statistics(z)

    return timestamps_full, x, y, z, timestamp_reduced, x_reduced, y_reduced, z_reduced, Hz_reduced, x_mean, x_std, x_var, y_mean, y_std, y_var, z_mean, z_std, z_var

def extract_action_from_response(response):
    match = re.search(r'\{(.*?)\}', response)
    if match:
        action = match.group(1)
        return action
    else:
        return None

def AI_prediction(timestamps_list, num_predictions, target_label, mistake_label, uuid, subject_num, relevant_timestamps):
    true_labels_all = []
    predictions_all = []
    all_name_true = []
    all_name_pred = []
    flower = 0
    for timestamp in timestamps_list:
        if flower==1:
            break
        print('Timestamp')
        print(timestamp)
        
        acc_data = read_dat_file_for_AI(timestamp, 'm_raw_acc.dat', subject_num, uuid)
        gyro_data = read_dat_file_for_AI(timestamp, 'm_raw_gyro.dat', subject_num, uuid)  
        magnet_data = read_dat_file_for_AI(timestamp, 'm_raw_magnet.dat', subject_num, uuid)
        
        if any(data is None for data in acc_data) or any(data is None for data in gyro_data) or any(data is None for data in magnet_data):
            print(f"Skipping timestamp {timestamp} due to no valid data.")
            continue
        
        mfcc_file = os.path.join('unpacked_data', subject_num, uuid, str(timestamp), 'sound.mfcc')
        
        mfcc_data = load_mfcc_file(mfcc_file)
        audio_data = reshape_mfcc_data(mfcc_data)
        mfcc_statistics = []
        
        if audio_data is None or audio_data.size == 0:
            print("No data to plot.")
            audio_data = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            for i in range(audio_data.shape[1]):
                mfcc_mean, mfcc_std, mfcc_var = 0, 0, 0
                mfcc_statistics.extend([mfcc_mean, mfcc_std, mfcc_var])
        else:
            for i in range(audio_data.shape[1]):
                mfcc_mean, mfcc_std, mfcc_var = calculate_statistics(audio_data[:, i])
                mfcc_statistics.extend([mfcc_mean, mfcc_std, mfcc_var])
        
        statistics_list = [
            acc_data[9], acc_data[10], acc_data[11], acc_data[12], acc_data[13], acc_data[14], acc_data[15], acc_data[16], acc_data[17],
            gyro_data[9], gyro_data[10], gyro_data[11], gyro_data[12], gyro_data[13], gyro_data[14], gyro_data[15], gyro_data[16], gyro_data[17],
            magnet_data[9], magnet_data[10], magnet_data[11], magnet_data[12], magnet_data[13], magnet_data[14], magnet_data[15], magnet_data[16], magnet_data[17]
        ] + mfcc_statistics
        
        Hz_audio = 10
        seconds_audio = 5
        
        num_samples = int(seconds_audio * Hz_audio)
        step = int(20 / Hz_audio)
        
        if audio_data is None or audio_data.size == 0:
            audio_data = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            audio_data = np.array([
                np.mean(audio_data[:num_samples * step, i].reshape(-1, step), axis=1)
                for i in range(audio_data.shape[1])
                ]).T
        
        if 'LYING_DOWN' in target_label:
            target_label = 'LYING DOWN'
            print(f'----------------------------------{target_label}----------------------------------')
        
        prompt = generate_prompt_reduced_acc(acc_data[8], acc_data[5], acc_data[6], acc_data [7])
        
        print(f"Target label: {target_label}")
        
        with tqdm.tqdm(total=1) as pbar:
            analysis_result = call_openai(prompt)
            pbar.update(1)
        
        print(f"Analysis for timestamp {timestamp}: {analysis_result}")
        
        analysis_result = extract_action_from_response(analysis_result)
        
        print(f'Action: {analysis_result}')
        
        if analysis_result == None:
            print(f'--------------------------------------------------------------------')
            continue
        
        if 'WALKING' in analysis_result:
            label_resp = 'WALKING'
        elif 'SITTING' in analysis_result:
            label_resp = 'SITTING'
        elif 'LYING DOWN' in analysis_result or 'LIYING DOWN' in analysis_result:
            label_resp = 'LYING DOWN'
        elif 'STANDING' in analysis_result:
            label_resp = 'STANDING'
        else:
            print('-------------------------------------------------------------------------------------------')
            continue
        
        print(f'label_resp: {label_resp}')
        
        if timestamp in relevant_timestamps:
            predicted_label = target_label in analysis_result
            true_label = True
            
            predictions_all.append(int(predicted_label))
            true_labels_all.append(int(true_label))
            
            all_name_true.append(target_label)
            all_name_pred.append(label_resp)
            
        else:
            predicted_label = target_label in analysis_result
            true_label = False
            
            predictions_all.append(int(predicted_label))
            true_labels_all.append(int(true_label))
            
            all_name_true.append(mistake_label)
            all_name_pred.append(label_resp)
        
        print('y_true:')
        print(true_labels_all)
        print('y_pred')
        print(predictions_all)
        
        accuracy, precision, recall, f1 = calculate_metrics(true_labels_all, predictions_all)
        
        num_predictions += 1
        print(f'NUM predictions: {num_predictions}')
        
        flower += 1
        
        if 'LYING DOWN' in target_label:
            target_label = 'LYING_DOWN'
            print(f'----------------------------------{target_label}----------------------------------')
    
    print(f"Accuracy for label '{target_label}': {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    
    return true_labels_all, predictions_all, all_name_true, all_name_pred, num_predictions

#//////////////////////////////////////////////////// LABELS ///////////////////////////////////////////////

def get_label_pretty_name(label): 
    if label.endswith('_'):
        label = label[:-1] + ')'
        pass
    
    label = label.replace('__',' (').replace('_',' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m','I\'m')
    return label

def calculate_total_time(Y, timestamps):
    unique_timestamps = set()
    for i in range(Y.shape[0]):
        if np.any(Y[i, :]):
            unique_timestamps.add(timestamps[i])
    total_time = len(unique_timestamps)
    return total_time

# -------------- For mutually-exclusive context-labels (time the user recorded and labeled contex) -------------------

'''
Y: The binary label matrix, where rows represent examples and columns represent labels.
label_names: A list of label names corresponding to the columns of Y.
labels_to_display: A list of specific labels that should be displayed in the pie chart.
title_str: A string representing the title of the pie chart.
ax: The matplotlib axis object where the pie chart will be plotted.
'''

def figure__pie_chart(Y,label_names,labels_to_display,title_str,ax):
    
    portion_of_time = np.mean(Y,axis=0) # Calculates the mean of each column in Y. This gives the average value of each label across all examples.
    
    portions_to_display = []  
    pretty_labels_to_display = []  
    portion_addition = 0
    
    for label in labels_to_display:
        label_ind = label_names.index(label)
        if portion_of_time[label_ind] > 0:
            portions_to_display.append(portion_of_time[label_ind])
            pretty_labels_to_display.append(get_label_pretty_name(label))
            pass
        pass
    
    for i in range(0, len(portions_to_display)):
        portion_addition += portions_to_display[i]
        pass
    
    if portion_addition > 0:
        ax.pie(portions_to_display,labels=pretty_labels_to_display,autopct='%.2f%%')  # Plots a pie chart on the given axis ax using the proportions in portions_to_display and labels in pretty_labels_to_display.
        # The autopct='%.2f%%' argument formats the percentages shown on the pie chart to two decimal places.
        ax.axis('equal')  # Sets the aspect ratio of the pie chart to be equal, ensuring that the pie is drawn as a circle.
        plt.title(title_str)
        pass
    else:
        ax.remove()
        pass
    return

# --------------- Label combinations that describe specific situations. ---------------

def get_actual_date_labels(tick_seconds):  # Converts a list of timestamps into human-readable date and time labels, assuming the data is in the US/Pacific time zone.
    
    time_zone = pytz.timezone('US/Pacific') # Assuming the data comes from PST time zone
    weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    datetime_labels = []  # Stores the formatted date-time labels
    for timestamp in tick_seconds:  #  # Loop over each timestamp in the input list
        tick_datetime = datetime.datetime.fromtimestamp(timestamp,tz=time_zone)  # Convert the timestamp to a datetime object in the specified time zone
        weekday_str = weekday_names[tick_datetime.weekday()]  # Get the day of the week as a string.
        time_of_day = tick_datetime.strftime('%I:%M%p')  # Get the time of day in the format 'HH:MM AM/PM'
        datetime_labels.append('%s\n%s' % (weekday_str,time_of_day))  # Format the date and time as 'Weekday\nTime' and append it to the list
        pass
    
    return datetime_labels
    
def figure__context_over_participation_time(timestamps, Y, label_names, label_colors, use_actual_dates=True):
    
    n_examples_per_label = np.sum(Y, axis=0)
    
    sorted_labels_and_counts = sorted(zip(label_names, n_examples_per_label), reverse=True, key=lambda pair: pair[1])
    
    if sum(count >= 50 for _, count in sorted_labels_and_counts) < 3:
        
        if sum(count >= 10 for _, count in sorted_labels_and_counts) < 3:
            limit = 0
        else:
            limit = 10
    else:
        limit = 50
    
    labels_to_display = [label for label, count in sorted_labels_and_counts [:9]]
    
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = plt.subplot(1, 1, 1)
    
    seconds_in_day = (60 * 60 * 24)

    ylabels = []
    ax.plot(timestamps, len(ylabels) * np.ones(len(timestamps)), '|', color='0.5', label='(Collected data)')
    ylabels.append('(Collected data)')

    for li, label in enumerate(labels_to_display):
        lind = label_names.index(label)
        is_label_on = Y[:, lind]
        label_times = timestamps[is_label_on]

        label_str = get_label_pretty_name(label)
        ax.plot(label_times, len(ylabels) * np.ones(len(label_times)), '|', color=label_colors[li], label=label_str)
        ylabels.append(label_str)
    
    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day)
    
    if use_actual_dates:
        tick_labels = get_actual_date_labels(tick_seconds)
        plt.xlabel('Time in San Diego', fontsize=14)
    else:
        tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)
        plt.xlabel('Days of participation', fontsize=14)
    
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels, fontsize=14)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=14)

    ax.set_ylim([-1, len(ylabels)])
    ax.set_xlim([min(timestamps), max(timestamps)])
    
    plt.show()
    return

# -------------Analyze the label matrix Y to see overall similarity/co-occurrence relations between labels ------------

def jaccard_similarity_for_label_pairs(Y):
    (n_examples,n_labels) = Y.shape
    Y = Y.astype(int)
    # For each label pair, count cases of:
    # Intersection (co-occurrences) - cases when both labels apply:
    both_labels_counts = np.dot(Y.T,Y)
    # Cases where neither of the two labels applies:
    neither_label_counts = np.dot((1-Y).T,(1-Y))
    # Union - cases where either of the two labels (or both) applies (this is complement of the 'neither' cases):
    either_label_counts = n_examples - neither_label_counts
    # To avoid division by zero, we set denominators to 1 where the counts are zero.
    either_label_counts_safe = np.where(either_label_counts == 0, 1, either_label_counts)
    # Jaccard similarity - intersection over union:
    J = both_labels_counts.astype(float) / either_label_counts_safe
    # Ensure J is zero where either_label_counts was originally zero
    J[either_label_counts == 0] = 0.
    return J

#//////////////////////////////////////////////////// SENSOR FEATURES  ///////////////////////////////////////////////

'''
Acc (phone-accelerometer), Gyro (phone-gyroscope), WAcc (watch-accelerometer), Loc (location), Aud (audio), and PS (phone-state).
Plus, the other sensors provided here that were not analyzed in the original paper: Magnet (phone-magnetometer), Compass (watch-compass), 
AP (audio properties, about the overall power of the audio), and LF (various low-frequency sensors).
'''

def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass

    return feat_sensor_names

# ------------------------------------- Examine the values of these features ----------------------------------------

def figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds):
    seconds_in_day = (60*60*24)
    days_since_participation = (timestamps - timestamps[0]) / float(seconds_in_day)
    
    for ind in feature_inds:
        feature = feature_names[ind]
        feat_values = X[:,ind]
        
        fig = plt.figure(figsize=(10,3),facecolor='white')
        
        ax1 = plt.subplot(1,2,1)
        ax1.plot(days_since_participation,feat_values,'.-',markersize=3,linewidth=0.1)
        plt.xlabel('days of participation')
        plt.ylabel('feature value')
        plt.title('%d) %s\nfunction of time' % (ind,feature))
        
        ax1 = plt.subplot(1,2,2)
        existing_feature = np.logical_not(np.isnan(feat_values))
        ax1.hist(feat_values[existing_feature],bins=30)
        plt.xlabel('feature value')
        plt.ylabel('count')
        plt.title('%d) %s\nhistogram' % (ind,feature))
        plt.show()
        pass
    
    return

# ///////////////////////////// Relation between sensor-features and context-label ///////////////////////////////////

def figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map):
    feat_ind1 = feature_names.index(feature1)
    feat_ind2 = feature_names.index(feature2)
    example_has_feature1 = np.logical_not(np.isnan(X[:,feat_ind1]))
    example_has_feature2 = np.logical_not(np.isnan(X[:,feat_ind2]))
    example_has_features12 = np.logical_and(example_has_feature1,example_has_feature2)
    
    fig = plt.figure(figsize=(12,5),facecolor='white')
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,4)
    
    for label in label2color_map.keys():
        label_ind = label_names.index(label)
        pretty_name = get_label_pretty_name(label)
        color = label2color_map[label]
        style = '.%s' % color
        
        is_relevant_example = np.logical_and(example_has_features12,Y[:,label_ind])
        count = sum(is_relevant_example)
        feat1_vals = X[is_relevant_example,feat_ind1]
        feat2_vals = X[is_relevant_example,feat_ind2]
        ax1.plot(feat1_vals,feat2_vals,style,markersize=5,label=pretty_name)
        
        ax2.hist(X[is_relevant_example,feat_ind1],bins=20,density=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count))
        ax3.hist(X[is_relevant_example,feat_ind2],bins=20,density=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count))
        pass
    
    ax1.set_xlabel(feature1)
    ax1.set_ylabel(feature2)
    
    ax2.set_title(feature1)
    ax3.set_title(feature2)
    
    ax2.legend(loc='best')
    plt.show()
    
    return

#/////////// The task of context recognition: predicting the context-labels based on the sensor-features //////////////
'''
We'll use linear models, specifically a logistic-regression classifier, to predict a single binary label.
We can choose which sensors to use.
'''
# ------------------------------------ Training the model -------------------------------------

def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature,is_from_sensor)
        pass
    X = X[:,use_feature]
    return X

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0)
    std_vec = np.nanstd(X_train,axis=0)
    return (mean_vec,std_vec)

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1))
    X_standard = X_centralized / normalizers
    return X_standard

def train_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use)
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)))

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train,mean_vec,std_vec)
    
    # The single target label:
    label_ind = label_names.index(target_label)
    y = Y_train[:,label_ind]
    missing_label = M_train[:,label_ind]
    existing_label = np.logical_not(missing_label)
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:]
    y = y[existing_label]
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.
    
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
        (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))))
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    lr_model.fit(X_train,y)
    
    # Assemble all the parts of the model:
    model = {\
            'sensors_to_use':sensors_to_use,\
            'target_label':target_label,\
            'mean_vec':mean_vec,\
            'std_vec':std_vec,\
            'lr_model':lr_model}
    
    return model

# --------------------------------------------------- Testing -----------------------------------------------------

def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model, target_label):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use'])
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])))

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec'])
    
    # The single target label:
    label_ind = label_names.index(model['target_label'])
    y = Y_test[:,label_ind]
    missing_label = M_test[:,label_ind]
    existing_label = np.logical_not(missing_label)
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:]
    y = y[existing_label]
    timestamps = timestamps[existing_label]

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.
    
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
        (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) )
    
    # Preform the prediction:
    y_pred = model['lr_model'].predict(X_test)
    
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y)
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y))
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)))
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y))
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp+fn)
    specificity = float(tn) / (tn+fp)
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp+fp)
    
    print("-"*10)
    print('Accuracy*:         %.2f' % accuracy)
    print('Sensitivity (TPR): %.2f' % sensitivity)
    print('Specificity (TNR): %.2f' % specificity)
    print('Balanced accuracy: %.2f' % balanced_accuracy)
    print('Precision**:       %.2f' % precision)
    print("-"*10)
    
    print('* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print('** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')
    
    fig = plt.figure(figsize=(10,4),facecolor='white')
    ax = plt.subplot(1,1,1)
    ax.plot(timestamps[y],1.4*np.ones(sum(y)),'|g',markersize=10,label='ground truth')
    ax.plot(timestamps[y_pred],np.ones(sum(y_pred)),'|b',markersize=10,label='prediction')
    
    seconds_in_day = (60*60*24)
    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day)
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)
    
    ax.set_ylim([0.5,5])
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels)
    plt.xlabel('days of participation',fontsize=14)
    ax.legend(loc='best')
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']))
    plt.show()
    
    return

# ------------------------------------- The same model on data of another user -------------------------------------

def validate_column_names_are_consistent(old_column_names,new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")
        
    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci,old_column_names[ci],new_column_names[ci]))
        pass
    return
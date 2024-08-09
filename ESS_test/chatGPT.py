import os
import json
import time
from openai import OpenAI

def load_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('api_key='):
                return line.strip().split('=', 1)[1]
    return None

os.environ["OPENAI_API_KEY"] = load_api_key()

client = OpenAI(
api_key=os.environ["OPENAI_API_KEY"],
)

# ////////////////////////////////////////////////////////////// Acc ///////////////////////////////////////////////////////////////////

def generate_prompt_reduced_acc(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced):
    print('Sensors:')
    print('Accelerometer')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc). The IMU data is given in the IMU coordinate frame. The three-axis accelerations are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity. The phone is in the pants pocket.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

The person's activity belongs to one of the following categories: "Lying down", "Sitting", "Standing", or "Walking". Please analyze the IMU readings step by step and determine the most likely activity. Your final answer should be a single activity, written in capital letters and enclosed in curly braces.
### Response: {{answer}}"""

# ////////////////////////////////////////////////////////// Acc and Gyro //////////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_gyro(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_gyro_reduced, x_gyro_reduced, y_gyro_reduced, z_gyro_reduced):
    print('Sensors:')
    print('Accelerometer and Gyroscope')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc) and {Hz_gyro_reduced} Hz (Gyro). The IMU data is given in the IMU coordinate frame. The three-axis accelerations and gyroscopes are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Gyroscopes (rad/s):
x-axis: {x_gyro_reduced}
y-axis: {y_gyro_reduced}
z-axis: {z_gyro_reduced}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

# ///////////////////////////////////////////////////////// Acc and Magnet ///////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_magnet(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_magnet_reduced, x_magnet_reduced, y_magnet_reduced, z_magnet_reduced):
    print('Sensors:')
    print('Accelerometer and Magnetometer')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc) and {Hz_magnet_reduced} Hz (Magnet). The IMU data is given in the IMU coordinate frame. The three-axis accelerations and magnetometer are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Magnetometer (µT):
x-axis: {x_magnet_reduced}
y-axis: {y_magnet_reduced}
z-axis: {z_magnet_reduced}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

# /////////////////////////////////////////////////////// Acc and Audio //////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_audio(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_audio, audio_data_reduced):
    print('Sensors:')
    print('Accelerometer and Audio')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc) and {Hz_audio} Hz (Audio). The IMU data is given in the IMU coordinate frame. The three-axis accelerations and magnetometers are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Audio Channels:
Channel 0: {audio_data_reduced[0]}
Channel 1: {audio_data_reduced[1]}
Channel 2: {audio_data_reduced[2]}
Channel 3: {audio_data_reduced[3]}
Channel 4: {audio_data_reduced[4]}
Channel 5: {audio_data_reduced[5]}
Channel 6: {audio_data_reduced[6]}
Channel 7: {audio_data_reduced[7]}
Channel 8: {audio_data_reduced[8]}
Channel 9: {audio_data_reduced[9]}
Channel 10: {audio_data_reduced[10]}
Channel 11: {audio_data_reduced[11]}
Channel 12: {audio_data_reduced[12]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

# /////////////////////////////////////////////////////// Acc, Gyro and Magnet /////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_gyro_magnet(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_gyro_reduced, x_gyro_reduced, y_gyro_reduced, z_gyro_reduced, Hz_magnet_reduced, x_magnet_reduced, y_magnet_reduced, z_magnet_reduced):
    print('Sensors:')
    print('Accelerometer, Gyroscope and Magnetometer')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc), {Hz_gyro_reduced} Hz (Gyro) and {Hz_magnet_reduced} Hz (Magnet). The IMU data is given in the IMU coordinate frame. The three-axis accelerations, gyroscopes and magnetometer are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Gyroscopes (rad/s):
x-axis: {x_gyro_reduced}
y-axis: {y_gyro_reduced}
z-axis: {z_gyro_reduced}

Magnetometer (µT):
x-axis: {x_magnet_reduced}
y-axis: {y_magnet_reduced}
z-axis: {z_magnet_reduced}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

# /////////////////////////////////////////////////////// Acc, Gyro and Audio //////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_gyro_audio(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_gyro_reduced, x_gyro_reduced, y_gyro_reduced, z_gyro_reduced, Hz_audio, audio_data_reduced):
    print('Sensors:')
    print('Accelerometer, Gyroscope and Audio')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc), {Hz_gyro_reduced} Hz (Gyro) and {Hz_audio} Hz (Audio). The IMU data is given in the IMU coordinate frame. The three-axis accelerations, gyroscopes and audio are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Gyroscopes (rad/s):
x-axis: {x_gyro_reduced}
y-axis: {y_gyro_reduced}
z-axis: {z_gyro_reduced}

Audio Channels:
Channel 0: {audio_data_reduced[0]}
Channel 1: {audio_data_reduced[1]}
Channel 2: {audio_data_reduced[2]}
Channel 3: {audio_data_reduced[3]}
Channel 4: {audio_data_reduced[4]}
Channel 5: {audio_data_reduced[5]}
Channel 6: {audio_data_reduced[6]}
Channel 7: {audio_data_reduced[7]}
Channel 8: {audio_data_reduced[8]}
Channel 9: {audio_data_reduced[9]}
Channel 10: {audio_data_reduced[10]}
Channel 11: {audio_data_reduced[11]}
Channel 12: {audio_data_reduced[12]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give the label that you think it is in capital letters.
###Response: {{answer}}"""

# ///////////////////////////////////////////////////// Acc, Magnet and Audio ////////////////////////////////////////////////////////

def generate_prompt_reduced_acc_magnet_audio(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_magnet_reduced, x_magnet_reduced, y_magnet_reduced, z_magnet_reduced, Hz_audio, audio_data_reduced):
    print('Sensors:')
    print('Accelerometer, Magnetometer and Audio')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc), {Hz_magnet_reduced} Hz (Magnet) and {Hz_audio} Hz (Audio). The IMU data is given in the IMU coordinate frame. The three-axis accelerations and gyroscopes are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Magnetometer (µT):
x-axis: {x_magnet_reduced}
y-axis: {y_magnet_reduced}
z-axis: {z_magnet_reduced}

Audio Channels:
Channel 0: {audio_data_reduced[0]}
Channel 1: {audio_data_reduced[1]}
Channel 2: {audio_data_reduced[2]}
Channel 3: {audio_data_reduced[3]}
Channel 4: {audio_data_reduced[4]}
Channel 5: {audio_data_reduced[5]}
Channel 6: {audio_data_reduced[6]}
Channel 7: {audio_data_reduced[7]}
Channel 8: {audio_data_reduced[8]}
Channel 9: {audio_data_reduced[9]}
Channel 10: {audio_data_reduced[10]}
Channel 11: {audio_data_reduced[11]}
Channel 12: {audio_data_reduced[12]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me the label that you think it is in capital letters, be careful that only one label its going to be in capital letters in your response.
###Response: {{answer}}"""

# /////////////////////////////////////////////////// Acc, Gyro, Magnet and Audio //////////////////////////////////////////////////////

def generate_prompt_reduced(Hz_acc_reduced, x_acc_reduced, y_acc_reduced, z_acc_reduced, Hz_gyro_reduced, x_gyro_reduced, y_gyro_reduced, z_gyro_reduced, Hz_magnet_reduced, x_magnet_reduced, y_magnet_reduced, z_magnet_reduced, Hz_audio, audio_data_reduced):
    print('Sensors:')
    print('Accelerometer, Gyroscope, Magnetometer and Audio')
    return f"""###Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
###Question:The IMU data is collected from sensors attached to the user's phone with a sampling rate of {Hz_acc_reduced} Hz (Acc), {Hz_gyro_reduced} Hz (Gyro), {Hz_magnet_reduced} Hz (Magnet) and {Hz_audio} Hz (Audio). The IMU data is given in the IMU coordinate frame. The three-axis of the sensors are given below. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerations (m/s^2):
x-axis: {x_acc_reduced}
y-axis: {y_acc_reduced}
z-axis: {z_acc_reduced}

Gyroscopes (rad/s):
x-axis: {x_gyro_reduced}
y-axis: {y_gyro_reduced}
z-axis: {z_gyro_reduced}

Magnetometer (µT):
x-axis: {x_magnet_reduced}
y-axis: {y_magnet_reduced}
z-axis: {z_magnet_reduced}

Audio Channels:
Channel 0: {audio_data_reduced[0]}
Channel 1: {audio_data_reduced[1]}
Channel 2: {audio_data_reduced[2]}
Channel 3: {audio_data_reduced[3]}
Channel 4: {audio_data_reduced[4]}
Channel 5: {audio_data_reduced[5]}
Channel 6: {audio_data_reduced[6]}
Channel 7: {audio_data_reduced[7]}
Channel 8: {audio_data_reduced[8]}
Channel 9: {audio_data_reduced[9]}
Channel 10: {audio_data_reduced[10]}
Channel 11: {audio_data_reduced[11]}
Channel 12: {audio_data_reduced[12]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me the label that you think it is in capital letters, be careful that only one label its going to be in capital letters in your response.
###Response: {{answer}}"""

# ///////////////////////////////////////////////////////// Statistics Acc /////////////////////////////////////////////////////////////

def generate_prompt_statistics_acc(list_statistics):
    print('Sensors Statistics:')
    print('Accelerometer')
    return f"""### Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses. Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.

### Question: The IMU data is collected from sensors attached to the user's phone with the following statistics:

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerometer Statistics (m/s²):
x-axis: mean = {list_statistics[0]}, std = {list_statistics[1]}, var = {list_statistics[2]}
y-axis: mean = {list_statistics[3]}, std = {list_statistics[4]}, var = {list_statistics[5]}
z-axis: mean = {list_statistics[6]}, std = {list_statistics[7]}, var = {list_statistics[8]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me the label that you think it is in capital letters, be careful that only one label its going to be in capital letters in your response.
###Response: {{answer}}"""

# ////////////////////////////////////////////////////// Statistics Acc and Gyro ///////////////////////////////////////////////////////

def generate_prompt_statistics_acc_gyro(list_statistics):
    return f"""### Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.
### Question: The IMU data is collected from sensors attached to the user's phone with the following statistics:

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerometer Statistics (m/s²):
x-axis: mean = {list_statistics[0]}, std = {list_statistics[1]}, var = {list_statistics[2]}
y-axis: mean = {list_statistics[3]}, std = {list_statistics[4]}, var = {list_statistics[5]}
z-axis: mean = {list_statistics[6]}, std = {list_statistics[7]}, var = {list_statistics[8]}

Gyroscope Statistics (rad/s):
x-axis: mean = {list_statistics[9]}, std = {list_statistics[10]}, var = {list_statistics[11]}
y-axis: mean = {list_statistics[12]}, std = {list_statistics[13]}, var = {list_statistics[14]}
z-axis: mean = {list_statistics[15]}, std = {list_statistics[16]}, var = {list_statistics[17]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me the label that you think it is in capital letters, be careful that only one label its going to be in capital letters in your response.
###Response: {{answer}}"""

# //////////////////////////////////////////////////// Statistics Acc, Gyro and Magnet /////////////////////////////////////////////////

def generate_prompt_statistics_acc_gyro_magnet(list_statistics):
    return f"""### Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.Note that the accelerations include gravitational acceleration, where 1 g ≈ 9.81 m/s². A constant acceleration value of around ±9.81 m/s² in one axis indicates the orientation of the device with respect to gravity.
### Question: The IMU data is collected from sensors attached to the user's phone with the following statistics:

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerometer Statistics (m/s²):
x-axis: mean = {list_statistics[0]}, std = {list_statistics[1]}, var = {list_statistics[2]}
y-axis: mean = {list_statistics[3]}, std = {list_statistics[4]}, var = {list_statistics[5]}
z-axis: mean = {list_statistics[6]}, std = {list_statistics[7]}, var = {list_statistics[8]}

Gyroscope Statistics (rad/s):
x-axis: mean = {list_statistics[9]}, std = {list_statistics[10]}, var = {list_statistics[11]}
y-axis: mean = {list_statistics[12]}, std = {list_statistics[13]}, var = {list_statistics[14]}
z-axis: mean = {list_statistics[15]}, std = {list_statistics[16]}, var = {list_statistics[17]}

Magnetometer Statistics (µT):
x-axis: mean = {list_statistics[18]}, std = {list_statistics[19]}, var = {list_statistics[20]}
y-axis: mean = {list_statistics[21]}, std = {list_statistics[22]}, var = {list_statistics[23]}
z-axis: mean = {list_statistics[24]}, std = {list_statistics[25]}, var = {list_statistics[26]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me the label that you think it is in capital letters, be careful that only one label its going to be in capital letters in your response.
###Response: {{answer}}"""

# ///////////////////////////////////////////////////// Statistics Acc, Gyro and Audio /////////////////////////////////////////////////

def generate_prompt_statistics_acc_gyro_audio(list_statistics):
    return f"""### Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
### Question: The IMU data is collected from sensors attached to the user's phone with the following statistics:

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerometer Statistics (m/s²):
x-axis: mean = {list_statistics[0]}, std = {list_statistics[1]}, var = {list_statistics[2]}
y-axis: mean = {list_statistics[3]}, std = {list_statistics[4]}, var = {list_statistics[5]}
z-axis: mean = {list_statistics[6]}, std = {list_statistics[7]}, var = {list_statistics[8]}

Gyroscope Statistics (rad/s):
x-axis: mean = {list_statistics[9]}, std = {list_statistics[10]}, var = {list_statistics[11]}
y-axis: mean = {list_statistics[12]}, std = {list_statistics[13]}, var = {list_statistics[14]}
z-axis: mean = {list_statistics[15]}, std = {list_statistics[16]}, var = {list_statistics[17]}

Audio MFCC Statistics:
Channel 1: mean = {list_statistics[27]}, std = {list_statistics[28]}, var = {list_statistics[29]}
Channel 2: mean = {list_statistics[30]}, std = {list_statistics[31]}, var = {list_statistics[32]}
Channel 3: mean = {list_statistics[33]}, std = {list_statistics[34]}, var = {list_statistics[35]}
Channel 4: mean = {list_statistics[36]}, std = {list_statistics[37]}, var = {list_statistics[38]}
Channel 5: mean = {list_statistics[39]}, std = {list_statistics[40]}, var = {list_statistics[41]}
Channel 6: mean = {list_statistics[42]}, std = {list_statistics[43]}, var = {list_statistics[44]}
Channel 7: mean = {list_statistics[45]}, std = {list_statistics[46]}, var = {list_statistics[47]}
Channel 8: mean = {list_statistics[48]}, std = {list_statistics[49]}, var = {list_statistics[50]}
Channel 9: mean = {list_statistics[51]}, std = {list_statistics[52]}, var = {list_statistics[53]}
Channel 10: mean = {list_statistics[54]}, std = {list_statistics[55]}, var = {list_statistics[56]}
Channel 11: mean = {list_statistics[57]}, std = {list_statistics[58]}, var = {list_statistics[59]}
Channel 12: mean = {list_statistics[60]}, std = {list_statistics[61]}, var = {list_statistics[62]}
Channel 13: mean = {list_statistics[63]}, std = {list_statistics[64]}, var = {list_statistics[65]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

# //////////////////////////////////////////////// Statistics Acc, Gyro, Magnet and Audio //////////////////////////////////////////////

def generate_prompt_statistics(list_statistics):
    return f"""### Instruction: You are an expert in IMU-based human activity analysis. Analyze the provided data without considering any previous analyses.
### Question: The IMU data is collected from sensors attached to the user's phone with the following statistics:

The accelerometer is oriented such that:
- The z-axis is normal to the screen.
- The x-axis is normal to the top of the phone.
- The y-axis is normal to the side of the phone.

Accelerometer Statistics (m/s²):
x-axis: mean = {list_statistics[0]}, std = {list_statistics[1]}, var = {list_statistics[2]}
y-axis: mean = {list_statistics[3]}, std = {list_statistics[4]}, var = {list_statistics[5]}
z-axis: mean = {list_statistics[6]}, std = {list_statistics[7]}, var = {list_statistics[8]}

Gyroscope Statistics (rad/s):
x-axis: mean = {list_statistics[9]}, std = {list_statistics[10]}, var = {list_statistics[11]}
y-axis: mean = {list_statistics[12]}, std = {list_statistics[13]}, var = {list_statistics[14]}
z-axis: mean = {list_statistics[15]}, std = {list_statistics[16]}, var = {list_statistics[17]}

Magnetometer Statistics (µT):
x-axis: mean = {list_statistics[18]}, std = {list_statistics[19]}, var = {list_statistics[20]}
y-axis: mean = {list_statistics[21]}, std = {list_statistics[22]}, var = {list_statistics[23]}
z-axis: mean = {list_statistics[24]}, std = {list_statistics[25]}, var = {list_statistics[26]}

Audio MFCC Statistics:
Channel 1: mean = {list_statistics[27]}, std = {list_statistics[28]}, var = {list_statistics[29]}
Channel 2: mean = {list_statistics[30]}, std = {list_statistics[31]}, var = {list_statistics[32]}
Channel 3: mean = {list_statistics[33]}, std = {list_statistics[34]}, var = {list_statistics[35]}
Channel 4: mean = {list_statistics[36]}, std = {list_statistics[37]}, var = {list_statistics[38]}
Channel 5: mean = {list_statistics[39]}, std = {list_statistics[40]}, var = {list_statistics[41]}
Channel 6: mean = {list_statistics[42]}, std = {list_statistics[43]}, var = {list_statistics[44]}
Channel 7: mean = {list_statistics[45]}, std = {list_statistics[46]}, var = {list_statistics[47]}
Channel 8: mean = {list_statistics[48]}, std = {list_statistics[49]}, var = {list_statistics[50]}
Channel 9: mean = {list_statistics[51]}, std = {list_statistics[52]}, var = {list_statistics[53]}
Channel 10: mean = {list_statistics[54]}, std = {list_statistics[55]}, var = {list_statistics[56]}
Channel 11: mean = {list_statistics[57]}, std = {list_statistics[58]}, var = {list_statistics[59]}
Channel 12: mean = {list_statistics[60]}, std = {list_statistics[61]}, var = {list_statistics[62]}
Channel 13: mean = {list_statistics[63]}, std = {list_statistics[64]}, var = {list_statistics[65]}

The person's action belongs to one of the following categories: "Lying down", "Sitting", "Standing" or "Walking". Could you please analyze and determine the person's action based on the given IMU readings? Please make an analysis step by step. I want you to give me only the label that you think it is in capital letters, but put me only one action in capital letters.
###Response: {{answer}}"""

def call_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.8,
            top_p=1
        )
    except Exception as e:
        time.sleep(20)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.8,
            top_p=1
        )
    response = json.loads(response.model_dump_json())
    return response["choices"][0]["message"]["content"]
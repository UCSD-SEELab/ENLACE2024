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


'F4665B94-4B4A-4FE0-BA33-FB567BE2E139' #1,#5
'50FB4388-2DD2-4246-A508-93A4D2894361' #2
'E4712828-EF23-4B4F-AA26-5A92D0BD4239' #3,#4

subject = int(input("Select subject number  to visualize (1/2/3/4/5):"))

match subject:
    case 1:
        uuid = 'F4665B94-4B4A-4FE0-BA33-FB567BE2E139'
        subject_num = 'subject_1'
    case 2:
        uuid = '50FB4388-2DD2-4246-A508-93A4D2894361'
        subject_num = 'subject_2'
    case 3:
        uuid = 'E4712828-EF23-4B4F-AA26-5A92D0BD4239'
        subject_num = 'subject_3'
    case 4:
        uuid = 'E4712828-EF23-4B4F-AA26-5A92D0BD4239'
        subject_num = 'subject_4'
    case 5:
        uuid = 'F4665B94-4B4A-4FE0-BA33-FB567BE2E139'
        subject_num = 'subject_5'


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

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
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
(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid,subject_num)

print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
timestamps.shape

print("The primary data files have %d different sensor-features" % len(feature_names))
X.shape
Y.shape
M.shape

print("The primary data files have %s context-labels" % len(label_names))


#//////////////////////////////////////////////////// LABELS ///////////////////////////////////////////////


# array n_examples_per_label where each element represents the number of examples for a corresponding label
n_examples_per_label = np.sum(Y,axis=0)  # Computes the sum of the binary label matrix Y along the columns
labels_and_counts_verif = zip(label_names,n_examples_per_label)# Creates pairs (tuples) of label names and their corresponding counts. 


i = 0

for (label,count) in labels_and_counts_verif:
    label_names.insert(i,convert_extrasensory_dataset_label_to_standard_network_label(label))
    i += 1
    label_names.pop(i)
    pass

labels_and_counts = zip(label_names, n_examples_per_label) #Creates an updated tuple
sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1])  # Sorts the pairs in descending order based on the counts.
# reverse=True ensures the sorting is in descending order.
# key=lambda pair: pair[1] specifies that sorting should be done based on the second element of each pair (the count).

# To prettify the standardized label names.

def get_label_pretty_name(label): 
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label;
print ("How many examples does this user have for each contex-label:")
print ("-"*20)
for (label,count) in sorted_labels_and_counts:
    print (" %s - %d minutes" % (get_label_pretty_name(label),count))
    pass

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

fig = plt.figure(figsize=(15,5),facecolor='white')

labels_to_display = ['LYING_DOWN','SITTING','STANDING','WALKING','RUNNING']
ax1 = plt.subplot(1,2,1)
figure__pie_chart(Y,label_names,labels_to_display,'Body state',ax1)


labels_to_display = ['PHONE_IN_HAND','PHONE_IN_BAG','PHONE_IN_POCKET','PHONE_ON_TABLE']
ax2 = plt.subplot(1,2,2)
figure__pie_chart(Y,label_names,labels_to_display,'Phone position',ax2)
plt.show()

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
    
def figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=True):

    fig = plt.figure(figsize=(10,7),facecolor='white')
    ax = plt.subplot(1,1,1)
    
    seconds_in_day = (60*60*24)

    ylabels = []
    ax.plot(timestamps,len(ylabels)*np.ones(len(timestamps)),'|',color='0.5',label='(Collected data)')
    ylabels.append('(Collected data)')

    for (li,label) in enumerate(labels_to_display):
        lind = label_names.index(label)
        is_label_on = Y[:,lind]
        label_times = timestamps[is_label_on]

        if len(label_times) > 0:
            label_str = get_label_pretty_name(label)
            ax.plot(label_times,len(ylabels)*np.ones(len(label_times)),'|',color=label_colors[li],label=label_str)
            ylabels.append(label_str)
        pass

    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day)
    if use_actual_dates:
        tick_labels = get_actual_date_labels(tick_seconds)
        plt.xlabel('Time in San Diego',fontsize=14)
        pass
    else:
        tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)
        plt.xlabel('Days of participation',fontsize=14)
        pass
    
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels,fontsize=14)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels,fontsize=14)

    ax.set_ylim([-1,len(ylabels)])
    ax.set_xlim([min(timestamps),max(timestamps)])
    plt.show()
    return

# Calling the function above

print("Here is a track of when the user was engaged in different contexts.")
print("The bottom row (gray) states when sensors were recorded (the data-collection app was not running all the time).")
print("The context-labels annotations were self-reported by ther user (and then cleaned by the researchers).")

labels_to_display = ['LYING_DOWN','RUNNING','BICYCLING','SITTING','STANDING','WALKING',
                    'IN_A_CAR','AT_HOME','AT_WORK']
label_colors = ['g','y','b','c','m','b','r','k','purple']
figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors)

# See the day of week and time of day
# figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=True)

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

J = jaccard_similarity_for_label_pairs(Y)

print("Label-pairs with higher color value tend to occur together more.")

fig = plt.figure(figsize=(10,10),facecolor='white')
ax = plt.subplot(1,1,1)
plt.imshow(J,interpolation='none');plt.colorbar()

pretty_label_names = [get_label_pretty_name(label) for label in label_names]
n_labels = len(label_names)
ax.set_xticks(range(n_labels))
ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7)
ax.set_yticks(range(n_labels))
ax.set_yticklabels(pretty_label_names,fontsize=7)
plt.show()

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

feat_sensor_names = get_sensor_names_from_features(feature_names)

for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature))
    pass

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

feature_inds = [0,102,133,148,157,158]
figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds)

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

feature1 = 'proc_gyro:magnitude_stats:time_entropy';#raw_acc:magnitude_autocorrelation:period';
feature2 = 'raw_acc:3d:mean_y'
label2color_map = {'PHONE_IN_HAND':'b','PHONE_ON_TABLE':'g'}
figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)

feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1'
feature2 = 'watch_acceleration:3d:mean_z'
label2color_map = {'WALKING':'b','WATCHING_TV':'g'}
figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)

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

sensors_to_use = ['Acc','WAcc']
target_label = 'WALKING'
model = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)

# --------------------------------------------------- Testing -----------------------------------------------------
def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
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
    
    return

test_model(X,Y,M,timestamps,feat_sensor_names,label_names,model)

# ------------------------------------- The same model on data of another user -------------------------------------
'''def validate_column_names_are_consistent(old_column_names,new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")
        
    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci,old_column_names[ci],new_column_names[ci]))
        pass
    return

uuid = '50FB4388-2DD2-4246-A508-93A4D2894361'
(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid)

# All the user data files should have the exact same columns. We can validate it:
validate_column_names_are_consistent(feature_names,feature_names2)
validate_column_names_are_consistent(label_names,label_names2)

test_model(X2,Y2,M2,timestamps2,feat_sensor_names,label_names,model)'''

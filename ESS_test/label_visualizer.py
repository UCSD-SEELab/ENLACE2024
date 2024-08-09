from visualizer import *

while True: 
    subject = input("Select subject to visualize (1/2/3/4/5...): ")
    if subject.isdigit():
        subject = int(subject)
        match subject:
            case 1:
                uuid = ''
                subject_num = 'subject_1'
                break
            case 2:
                uuid = ''
                subject_num = 'subject_2'
                break
            case 3:
                uuid = ''
                subject_num = 'subject_3'
                break
            case 4:
                uuid = ''
                subject_num = 'subject_4'
                break
            case 5:
                uuid = ''
                subject_num = 'subject_5'
                break
            case _:
                print("Input a valid number")
                break
    else:
        print("Input a number")


print(f"Selected Subject: {subject_num}")
print(f"UUID: {uuid}")

(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid,subject_num)

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

print ("How many examples does this user have for each context-label:")
print ("-"*20)
for (label,count) in sorted_labels_and_counts:
    print (" %s - %d minutes" % (get_label_pretty_name(label),count))
    pass

# Counts the total number of unique timestamps where any label is active
print ("-"*20)

total_time = calculate_total_time(Y, timestamps)
print(f'The total time collected for context-labels is {total_time} minutes')


# -------------- For mutually-exclusive context-labels (time the user recorded and labeled contex) -------------------
'''
Y: The binary label matrix, where rows represent examples and columns represent labels.
label_names: A list of label names corresponding to the columns of Y.
labels_to_display: A list of specific labels that should be displayed in the pie chart.
title_str: A string representing the title of the pie chart.
ax: The matplotlib axis object where the pie chart will be plotted.
'''

fig = plt.figure(figsize=(15,5),facecolor='white')

labels_to_display = ['LYING_DOWN','SITTING','STANDING','WALKING','RUNNING']
ax1 = plt.subplot(1,2,1)
figure__pie_chart(Y,label_names,labels_to_display,'Body state',ax1)


labels_to_display = ['PHONE_IN_HAND','PHONE_IN_BAG','PHONE_IN_POCKET','PHONE_ON_TABLE']
ax2 = plt.subplot(1,2,2)
figure__pie_chart(Y,label_names,labels_to_display,'Phone position',ax2)

plt.tight_layout()
plt.show()

# --------------- Label combinations that describe specific situations. ---------------

# Calling the function above

print("Here is a track of when the user was engaged in different contexts.")
print("The bottom row (gray) states when sensors were recorded (the data-collection app was not running all the time).")
print("The context-labels annotations were self-reported by ther user (and then cleaned by the researchers).")

labels_to_display = ['LYING_DOWN','RUNNING','BICYCLING','SITTING','STANDING','WALKING',
                    'IN_A_CAR','AT_HOME','AT_WORK']
label_colors = ['g','y','b','c','m','b','r','k','purple']
figure__context_over_participation_time(timestamps,Y,label_names,label_colors)

# See the day of week and time of day
# figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=True)

# -------------Analyze the label matrix Y to see overall similarity/co-occurrence relations between labels ------------

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

feat_sensor_names = get_sensor_names_from_features(feature_names)

for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature))
    pass

# ------------------------------------- Examine the values of these features ----------------------------------------

feature_inds = [0,102,133,148,157,158]
figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds)

# ///////////////////////////// Relation between sensor-features and context-label ///////////////////////////////////


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

sensors_to_use = ['Acc','WAcc']
target_label = 'WALKING'
model = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)

# --------------------------------------------------- Testing -----------------------------------------------------

test_model(X,Y,M,timestamps,feat_sensor_names,label_names,model)

# ------------------------------------- The same model on data of another user -------------------------------------

uuid = '50FB4388-2DD2-4246-A508-93A4D2894361'
(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid)

# All the user data files should have the exact same columns. We can validate it:
validate_column_names_are_consistent(feature_names,feature_names2)
validate_column_names_are_consistent(label_names,label_names2)

test_model(X2,Y2,M2,timestamps2,feat_sensor_names,label_names,model,target_label)
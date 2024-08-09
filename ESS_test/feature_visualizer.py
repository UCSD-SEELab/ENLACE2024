from visualizer import read_user_data, get_timestamps_with_label, plot_dat_file

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

Hz_options = ['40 Hz', '20 Hz', '10 Hz', '5 Hz', '2 Hz']

Hz_int = [40, 20, 10, 5, 2]

time_options = ['20s', '15 s', '10 s', '5 s', 'all']

time_int = [20, 15, 10, 5]

main_labels = [
    'WALKING',
    'RUNNING',
    'SITTING',
    'LYING_DOWN',
    'KNEELING',
    'STANDING',
    'BICYCLING'
]

print("List of Labels: ")

for i, main_label in enumerate(main_labels, start=1):
    print(f"{i}. {main_label}")

while True:
    selection_label = input(f"Select a Label (1-{len(main_labels)}): ")
    if selection_label.isdigit() and 1 <= int(selection_label) <= len(main_labels):
        target_label = main_labels[int(selection_label) - 1]
        print(f"Selected Label: {target_label}")
        break
    else:
        print("Input a valid selection")

X, Y, M, timestamps, feature_names, label_names = read_user_data(uuid, subject_num)

label_timestamps = get_timestamps_with_label(Y,timestamps,label_names, target_label)
relevant_timestamps = get_timestamps_with_label(Y, timestamps, label_names, target_label)

print(f"Timestamps with the label '{target_label}':")
print(label_timestamps)

plot_dat_file(label_timestamps, target_label, uuid, subject_num)
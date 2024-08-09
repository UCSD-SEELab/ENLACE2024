from visualizer import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

main_labels = [
    'WALKING',
    'SITTING',
    'LYING_DOWN',
    'STANDING'
]

subjects = [
    ('', 'subject_1'),
    ('', 'subject_2'),
    ('', 'subject_3'),
    ('', 'subject_4'),
    ('', 'subject_5')
]

total_correct_predictions = 0
total_predictions = 0
all_pred = []
all_true = []
all_name_pred = []
all_name_true = []
num_predictions = 0

mistake_labels = [
    'WALKING',
    'SITTING',
    'LYING_DOWN',
    'STANDING'
]

for target_label in main_labels:
    print('Case: ', target_label)
    for i, mistake_label_selection in enumerate(mistake_labels, start=1):
        print(f"{i}. {mistake_label_selection}")
    while True:
        selection_label_mistake = input(f"Select a Mistake Label (1-{len(mistake_labels)}): ")
        if selection_label_mistake.isdigit() and 1 <= int(selection_label_mistake) <= len(mistake_labels):
            mistake_label = mistake_labels[int(selection_label_mistake) - 1]
            
            print(f"Selected Label for mistake: {mistake_label}")
            break
        else:
            print("Input a valid selection")
    
    for uuid, subject_num in subjects:
        print(f"Reading data for {subject_num}")
        all_true_user = []
        all_pred_user = []
        try:
            X, Y, M, timestamps, feature_names, label_names = read_user_data(uuid, subject_num)
            
            relevant_timestamps = get_timestamps_with_label(Y, timestamps, label_names, target_label)
            mistakes_timestamps = get_timestamps_with_label(Y, timestamps, label_names, mistake_label)
            
            combined_timestamps = combine_timestamps(relevant_timestamps, mistakes_timestamps)
            
            print(f"Subject: {subject_num}")    
            print(f"Total timestamps with {target_label}: {len(relevant_timestamps)}")
            print(f"Total timestamps with {mistake_label}: {len(mistakes_timestamps)}")
            
            true_label, pred_label, true_name, pred_name, num_predictions = AI_prediction(combined_timestamps, num_predictions, target_label, mistake_label, uuid, subject_num, relevant_timestamps)
            
            all_true += true_label
            all_pred += pred_label
            
            all_name_true += true_name
            all_name_pred += pred_name
    
        except Exception as e:
            print(f"An error occurred for {subject_num}: {e}")

print(f'all_y_true size:', len(all_true))
print(f'all_y_pred size:', len(all_pred))

print(f'true_name: {all_name_true}')
print(f'pred_name: {all_name_pred}')

all_accuracy, all_precision, all_recall, all_f1 = calculate_metrics(all_true, all_pred)

abbreviations = {
    'WALKING': 'W',
    'SITTING': 'S',
    'LYING DOWN': 'LD',
    'STANDING': 'ST'
}

all_name_true = [abbreviations[name] for name in all_name_true]
all_name_pred = [abbreviations[name] for name in all_name_pred]

main_labels = ['W', 'S', 'LD', 'ST']

cm = confusion_matrix(all_name_true, all_name_pred, labels=main_labels)

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=main_labels)
disp.plot(ax=ax, cmap='Blues', colorbar=True)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for text in disp.text_.ravel():
    text.set_fontsize(30)

cbar = disp.im_.colorbar
cbar.ax.tick_params(labelsize=20)

ax.set_xlabel(ax.get_xlabel(), fontsize=20)
ax.set_ylabel(ax.get_ylabel(), fontsize=20)

plt.show()

print(f"All Accuracy for 'Prompt 1': {all_accuracy * 100:.2f}%")
print(f"All Precision: {all_precision * 100:.2f}%")
print(f"All Recall: {all_recall * 100:.2f}%")
print(f"All F1-score: {all_f1 * 100:.2f}%")
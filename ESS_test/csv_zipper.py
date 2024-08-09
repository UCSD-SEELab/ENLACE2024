from processing_uploaded_data import *

while True: 
    subject = input("Select subject number to extract (1/2/3/4/5...): ")
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


unpack_user_uploaded_data(uuid, 'uploaded_data',subject_num, 'unpacked_data')

collect_user_data_and_save_csv_file(uuid, 'unpacked_data',subject_num, 'csvs',subject_num, get_default_list_of_labels())
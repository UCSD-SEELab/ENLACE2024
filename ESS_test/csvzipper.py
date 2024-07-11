from processing_uploaded_data import *

'F4665B94-4B4A-4FE0-BA33-FB567BE2E139' #1
'50FB4388-2DD2-4246-A508-93A4D2894361' #2
'E4712828-EF23-4B4F-AA26-5A92D0BD4239' #3

subject = int(input("Select subject number to extract (1/2/3/4/5):"))

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


unpack_user_uploaded_data(uuid, 'uploaded_data',subject_num, 'unpacked_data')

collect_user_data_and_save_csv_file(uuid, 'unpacked_data',subject_num, 'csvs',subject_num, get_default_list_of_labels())
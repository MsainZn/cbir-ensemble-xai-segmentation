
# ImportsMore actions
import os
import string
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import euclidean
from PIL import Image

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch



# Function: Evaluates the validity of dates
def is_valid_date(date_str):
    try:
        if isinstance(date_str, str):
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        else:
            return False
    except ValueError:
        return False



# Function: Converts given date to age
def convert_birthday_to_age(birthday_date):
    birthday_date = datetime.strptime(birthday_date, "%Y-%m-%d").date()
    today = datetime.today()
    age = today.year - birthday_date.year - ((today.month, today.day) < (birthday_date.month, birthday_date.day))
    return age



# Function: Convert bra band size to under-bust (interval midpoint)
def convert_brasize_to_underbust(bra_band):

    """
    International / European / Japan / South Korea
    Measurements in centimeters (cm)
    """

    # Build the dict
    bra_band_dict = dict()
    for b in range(55, 155, 5):
        bra_band_dict[f'{b}'] = pd.Interval(left=b, right=b+5, closed='left')

    # And compute the under_bust
    try:
        under_bust = bra_band_dict[f'{bra_band}'].mid
    except:
        under_bust = None

    return under_bust



# Function: Convert bra cup to difference between over and under-bust (interval midpoint)
def convert_bracup_to_oubustdiff(bra_cup):

    """
    International / Europe / France / Spain / Belgium / Italy
    Measurements in centimetres (cm)
    """

    # Get the possible bra cups
    iefebi_bra_cups = ['AA']
    iefebi_bra_cups += list(string.ascii_uppercase)[0:20]
    iefebi_bra_cups.remove('Q')

    # Create a dictionary of bra cups and intervals of differences
    iefebi_bra_cups_dict = dict()
    for c, b in zip(iefebi_bra_cups, range(10, 50, 2)):
        iefebi_bra_cups_dict[c] = pd.Interval(left=b, right=b+2, closed='left')

    # And compute the bust_ou_diff
    try:
        bust_ou_diff_out  = iefebi_bra_cups_dict[bra_cup.upper()].mid
    except:
        bust_ou_diff_out = None

    return bust_ou_diff_out



# Function: Get tabular (clinical) data from ID
def get_tabular_data_from_id(patient_info, id):

    # If "patient_info" is a str, then it should be a path to the CSV
    if isinstance(patient_info, str):
        patient_info = pd.read_csv(patient_info)

    # Get data
    tab_data = patient_info[patient_info['Patient ID'] == id]

    return tab_data



# Function: Get tabular (clinical) data in NumPy format
def get_tabular_data_from_id_numpy(patient_info, id):

    # Path to the uploaded CSV file
    data_as_strings = np.genfromtxt(patient_info, delimiter=',', dtype=str, encoding='utf-8', skip_header=1)

    # Ensuring data is 2D
    if data_as_strings.ndim == 1:
        data_as_strings = np.array([data_as_strings])

    # Convert id_to_match to string for comparison
    id_to_match_str = str(id)

    # Selecting rows where the value in the specified column matches the given ID (as string)
    selected_rows_as_strings = [row for row in data_as_strings if row[2] == id_to_match_str]

    # Display the selected rows

    return selected_rows_as_strings



# Function: Get tabular (clinical) features, filtered, in NumPy
def get_tabular_features_filtered_numpy(patient_info_csv, id, selected_features=[4, 5, 1, 6, 7]):

    # Get full info from the patient
    full_info = get_tabular_data_from_id_numpy(patient_info_csv, id)
    selected_info = np.array([full_info[0][index] for index in selected_features])
    if '' in selected_info:
        return None

    # Divide the info into clinical features
    height = np.float64(selected_info[0])
    weight = np.float64(selected_info[1])
    age = convert_birthday_to_age(selected_info[2])
    bs = convert_brasize_to_underbust(selected_info[3])
    bc = convert_bracup_to_oubustdiff(selected_info[4])
    if height is None or weight is None or age is None or bs is None or bc is None:
        return None

    # Reorganise the vectors
    selected_info[0] = height    
    selected_info[1] = weight    
    selected_info[2] = np.float64(age)
    selected_info[3] = np.float64(bs)
    selected_info[4] = np.float64(bc)
    selected_info = selected_info.astype(np.float64) 

    # If there is still a None, return None
    if np.any(selected_info == None):
        return None

    return selected_info



# Function: Get tabular (clinical) features, filtered
def get_tabular_features_filtered(patient_info_csv, id, selected_features=['Patient Height', 'Patient Weight', 'Patient Birthday', 'Bra Size', 'Bra Cup']):

    # Get full information from the patients
    full_info = get_tabular_data_from_id(patient_info_csv, id)
    selected_info = full_info[selected_features]

    # Check if any of the selected information is a NaN
    if selected_info.isna().any().any():
        return None

    # Select the tabular (clinical) features
    if 'Patient Birthday' in selected_features:
        selected_info.loc[:, 'Patient Birthday'] = selected_info['Patient Birthday'].apply(convert_birthday_to_age)

    if 'Bra Size' in selected_features:
        selected_info.loc[:, 'Bra Size']  = selected_info['Bra Size'].apply(convert_brasize_to_underbust)

    if 'Bra Cup' in selected_features:
        selected_info.loc[:, 'Bra Cup']  = selected_info['Bra Cup'].apply(convert_bracup_to_oubustdiff)

    # Check if there is no NaN
    if selected_info.isna().any().any():
        return None

    return selected_info.to_numpy()[0].astype(np.float64)



# Function: Get catalogues IDs per doctor (i.e., ground-truth)
def get_catalogues_ids_per_doctor(catalogue_csv, doctor_csv, type, doctor_code):

    # If catalogue is a string then it should be a filepath to a CSV
    if isinstance(catalogue_csv, str):
        catalogue_csv = pd.read_csv(catalogue_csv)
    if isinstance(doctor_csv, str):
        doctor_csv = pd.read_csv(doctor_csv)

    # Filter DataFrame for the specific doctor
    doctor_objects = doctor_csv[doctor_csv['User'] == doctor_code]
    catalogues = []

    # Iterate through every row in the doctor_objects DataFrame
    for index, row in doctor_objects.iterrows():
        catalogues.append(row['Catalogue ID'])

    # Initialize lists to hold query and retrieval data
    query = []
    retr = []

    # Determine the type of catalogue
    if type in ['E', 'G']:
        type_str = 'Ordered Exc Good Catalogue'
    elif type in ['P', 'F']:
        type_str = 'Ordered Fair Poor Catalogue'
    else:
        raise ValueError("Invalid type. Use 'E', 'G' for Excellent/Good and 'P', 'F' for Poor/Fair catalogues.")

    # Iterate through catalogues
    for i in range(len(catalogues)):
        row = catalogue_csv.loc[catalogue_csv['Catalogue ID'] == catalogues[i]]
        if pd.notna(row['Query Patient']).any() and pd.notna(row[type_str]).any():
            result_array = np.fromstring(row[type_str].values[0], sep=',')
            result_array = result_array.astype(int)
            if len(result_array) >= 2:
                query.append(row['Query Patient'].values[0])  # Adding the query patient value
                retr.append(result_array)  # Appending the result array as is

    # Return query and retr without converting retr to a NumPy array
    return query, retr



# Function: Get pre-operative image from patient ID
def get_pre_image_from_id(patient_info, favorite_images, patient_images, id, images_resized_path):
    print(f"Processing Patient ID: {id}")

    # If these are strings, then they should be CSV filepaths
    if isinstance(patient_info, str):
        patient_info = pd.read_csv(patient_info, low_memory=False)
    if isinstance(favorite_images, str):
        favorite_images = pd.read_csv(favorite_images, low_memory=False)
    if isinstance(patient_images, str):
        patient_images = pd.read_csv(patient_images, low_memory=False)

    # Filter DataFrame for the specific doctor
    this_patient_info = patient_info[patient_info['Patient ID'] == id]
    this_favorite_images = favorite_images[favorite_images['Patient ID'] == id]
    pre_img = ' '

    # Debug: Print the patient info and favorite images
    print(f"Patient Info: {this_patient_info}")
    print(f"Favorite Images: {this_favorite_images}")

    for _, row in this_patient_info.iterrows():
        surg_date = pd.to_datetime(row['Surgery Date'])

    for _, row in this_favorite_images.iterrows():
        date_difference = pd.to_datetime(row['Date']) - surg_date
        if (date_difference) <= pd.Timedelta(0):
            pre_img = row['Image ID']

    this_patient_image = patient_images[patient_images['Image ID'] == pre_img]
    if this_patient_image.empty:
        return None
    else:
        image_filename = this_patient_image['Image Filename'].item()
        
        # Clean any existing _resized suffix before processing
        base_name = os.path.splitext(image_filename.replace('_resized', ''))[0]
        
        if images_resized_path:
            # Return the correct resized path (will be created later)
            return os.path.join(images_resized_path, f"{os.path.basename(base_name)}_resized.jpg")
        else:
            # Return original filename
            return f"{base_name}.jpg"

# Function: Get catalogues IDs from CSV
def get_catalogues_ids_from_csv(catalogue_csv, type):

    # If this is a string, then it should be a CSV filepath
    if isinstance(catalogue_csv, str):
        catalogue_csv = pd.read_csv(catalogue_csv)

    # Get the values of the 'Age' column
    Q_IDs = catalogue_csv['Query Patient']
    if(type == 'E' or type == 'G'):
        R_IDs = catalogue_csv['Ordered Exc Good Catalogue']
    if(type == 'P' or type == 'F'):
        R_IDs = catalogue_csv['Ordered Fair Poor Catalogue']
    query = []
    retr = []

    for i in range(len(Q_IDs)):
        # Check for NaN values before using np.fromstring
        if pd.notna(R_IDs.iloc[i]) and pd.notna(Q_IDs.iloc[i]):
            result_array = np.fromstring(R_IDs[i], sep=',')
            result_array = result_array.astype(int)
            if(len(result_array) >= 2):
                retr.append(result_array)
                query.append(Q_IDs[i].item())

    return query, retr



# Class: QNS Structures
class QNS_structure:

    # Method: __init__
    def __init__(self, query_vector=None, neighbor_vectors=[], scores=None, cmpfunc=euclidean, query_vector_id=-1, neighbor_vectors_ids=[]):    

        self.set_query_vector(query_vector, query_vector_id)
        self.set_neighbor_vectors(neighbor_vectors, neighbor_vectors_ids)
        self.set_score_vector(scores)
        self.dist_func = cmpfunc

        return


    # Method: Set query vector
    def set_query_vector(self, query_vector,query_vector_id=-1):
        self.query_vector = query_vector #+ 1e-15 * np.ones(query_vector.Shape, np.float64)
        self.query_vector_id = query_vector_id

        return


    # Method: Append information to vector
    def append_to_vector(self, vec1, vec2):

        vec1 = np.append(vec1, vec2)

        return vec1


    # Method: Set score vector
    def set_score_vector(self, score_vector):

        self.score_vector = score_vector

        return


    # Method: Set neighbour vectors
    def set_neighbor_vectors(self, neighbor_vectors, neighbor_vectors_ids):
        if neighbor_vectors == []:
            self.neighbor_count = 0
            self.neighbor_vectors = []
            self.neighbor_vectors_id = []
        else:
            self.neighbor_count = len(neighbor_vectors)
            self.neighbor_vectors = neighbor_vectors
            self.set_neighbor_vectors_ids = neighbor_vectors_ids

        return


    # Method: Calculate expert score
    def calculate_expert_score(self):

        if self.neighbor_count > 0:
            self.score_vector = np.arange(self.neighbor_count, dtype=np.float64)

        return


    # Method: Show neighbour vectors
    def show_neighbor_vectors(self):

        for idx, neighbor in enumerate(self.neighbor_vectors):
            print(f'Neighbor PID: {self.neighbor_vectors_id[idx]} NID: {idx} Vec: {neighbor}')

        return


    # Method: Show query vector
    def show_query_vector(self):

        print(f'Query => PID: {self.query_vector_id} | Vec: {self.query_vector}')

        return


    # Method: Show score vector
    def show_score_vector(self):

        print('Score-Vector: ', self.score_vector)

        return


    # Method: Show summary
    def show_summary(self, num_flag=False, str=True):
        if num_flag:
            init_dist = self.calculate_initial_distance()
            init_score = self.calculate_initial_score ()

        self.show_query_vector()
        for idx, neighbor in enumerate(self.neighbor_vectors):
            print(f'Neighbor NID:{idx:02} => Value: [ ', end='')
            if str == False:
                for n in neighbor:
                    print(f' {n:02}', end='')
            else:
                for n in neighbor:
                    print(f'{n:}', end='')
            print(' ]', end='')
            if num_flag:
                print(f'Init Dist: {init_dist[idx]:06.3f}  | Dist-Based Score: {init_score[idx]:02d}')
            print(f' | Expert Score: {self.score_vector[idx]:02.1f}')

        if num_flag:
            print("Initial Sort: ", self.calculate_initial_sort())
        print("Neighbor PID: ", self.neighbor_vectors_id)
        print("Expert  Sort: ", self.calculate_expert_sort())
        print("----------------------------------")

        return


    # Method: Calculate initial distance between neighbours
    def calculate_initial_distance(self):
        initial_distance = [self.dist_func(self.query_vector, neighbor) for neighbor in self.neighbor_vectors]
        if self.dist_func is not euclidean:
            initial_distance = 1.0 - initial_distance

        return initial_distance  


    # Method: Calculate initial score
    def calculate_initial_score(self):
        sorted_indices = np.argsort(np.array(self.calculate_initial_distance())) # Use argsort to get the indices that would sort the array
        initial_score  = np.zeros_like(sorted_indices)

        for score, index in enumerate(sorted_indices):
            initial_score[index] = score

        return initial_score


    # Method: Calculate initial sort of the distances
    def calculate_initial_sort(self):
        initial_sort  = np.argsort(self.calculate_initial_distance())
        return initial_sort


    # Method: Calculate expert sort
    def calculate_expert_sort(self):
        if self.score_vector is not None:
            expert_sort  = np.argsort(self.score_vector)
        return expert_sort


    # Method: Add neighbour vector
    def add_neighbor_vector(self, neighbor_vector, neighbor_vector_id=-1):
        self.neighbor_vectors.append(neighbor_vector)
        self.neighbor_vectors_id.append(neighbor_vector_id)
        self.neighbor_count = self.neighbor_count + 1

        return


    # Method: Delete neighbour vector
    def delete_neighbor_vector(self, pid):
        for i in range(self.neighbor_count):
            if pid == self.neighbor_vectors_id[i]:
                del self.neighbor_vectors[i]
                del self.neighbor_vectors_id[i]
                self.score_vector = np.delete(self.score_vector, i)
                self.neighbor_count = self.neighbor_count - 1
                print('Deletion...')
                break

        return


    # Method: Utility function to convert query_vector to torch tensor
    def convert_query_to_torch(self):
        if self.query_vector is not None:
            self.query_vector = torch.tensor(self.query_vector, dtype=torch.float64, requires_grad=False)

        return


    # Method: Utility function to convert score_vector to torch tensor
    def convert_score_to_torch(self):
        if self.score_vector is not None:
            self.score_vector = torch.tensor(self.score_vector, dtype=torch.float64, requires_grad=False)

        return


    # Method: Utility function to convert neighbor_vectors to torch tensors
    def convert_neighbors_to_torch(self):

        self.neighbor_vectors = [
            torch.tensor(neighbor, dtype=torch.float64, requires_grad=False) for neighbor in self.neighbor_vectors
        ]

        return


    # Method: Utility function to convert query_vector back to NumPy array
    def convert_query_to_numpy(self):

        if isinstance(self.query_vector, torch.Tensor):
            self.query_vector = self.query_vector.numpy()

        return


    # Method: Utility function to convert score_vector back to NumPy array
    def convert_score_to_numpy(self):

        if isinstance(self.score_vector, torch.Tensor):
            self.score_vector = self.score_vector.numpy()

        return


    # Method: Utility function to convert neighbor_vectors back to numpy arrays
    def convert_neighbors_to_numpy(self):

        if all(isinstance(neighbor, torch.Tensor) for neighbor in self.neighbor_vectors):
            self.neighbor_vectors = [neighbor.numpy() for neighbor in self.neighbor_vectors]

        return



# Function: Get query neighbour elements
def get_query_neighbor_elements(catalogue_info_csv, catalogue_user_info_csv, patient_info_csv, 
                              catalogue_type='E', doctor_code=-1, patient_ids=None):
    """Get tabular data elements, filtered by patient_ids if provided"""
    if doctor_code == -1:
        queries_id, neighbours_id = get_catalogues_ids_from_csv(catalogue_info_csv, catalogue_type)
    else:
        queries_id, neighbours_id = get_catalogues_ids_per_doctor(catalogue_info_csv, catalogue_user_info_csv, catalogue_type, doctor_code)

    # Filter by patient_ids if provided
    if patient_ids is not None:
        # Create filtered lists
        filtered_queries = []
        filtered_neighbours = []
        
        for q, n_list in zip(queries_id, neighbours_id):
            if q in patient_ids:
                # Filter neighbors to only include valid patients
                filtered_n_list = [n for n in n_list if n in patient_ids]
                if filtered_n_list:  # Only add if there are valid neighbors
                    filtered_queries.append(q)
                    filtered_neighbours.append(filtered_n_list)
        
        queries_id = filtered_queries
        neighbours_id = filtered_neighbours

    # Build QNS list
    QNS_list = []
    for idx in range(len(queries_id)): 
        qns_element = QNS_structure()
        itm = get_tabular_features_filtered_numpy(patient_info_csv, queries_id[idx]) 
        if itm is None:
            continue
        qns_element.set_query_vector(itm, queries_id[idx])

        for jdx in range(len(neighbours_id[idx])): 
            itm = get_tabular_features_filtered_numpy(patient_info_csv, neighbours_id[idx][jdx]) 
            if itm is None:
                continue
            qns_element.add_neighbor_vector(itm, neighbours_id[idx][jdx])
        qns_element.calculate_expert_score()
        QNS_list.append(qns_element)

    return QNS_list, len(QNS_list)

# Function: Get query neighbour elements from both image and tabular datasets
def get_query_neighbor_elements_image_tabular(catalogue_info_csv, catalogue_user_info_csv, 
                                            patient_info_csv, favorite_image_info_csv, 
                                            patient_images_info_csv, images_original_path, 
                                            catalogue_type='E', doctor_code=-1,
                                            images_resized_path=None):
    print("\n=== Starting Data Processing ===")
    
    # Load catalogue data
    if doctor_code == -1:
        queries_id, neighbours_id = get_catalogues_ids_from_csv(catalogue_info_csv, catalogue_type)
    else:
        queries_id, neighbours_id = get_catalogues_ids_per_doctor(catalogue_info_csv, catalogue_user_info_csv, catalogue_type, doctor_code)

    print(f"Found {len(queries_id)} initial queries")
    
    # Remove repetitive indexes
    for idx, q in enumerate(queries_id):
        neighbours_id[idx] = [n for n in neighbours_id[idx] if n != q]

    QNS_img_list = []
    QNS_tab_list = []
    skipped_counts = {'no_tabular': 0, 'no_image': 0, 'no_file': 0}

    for idx, qid in enumerate(queries_id):
        # Debug print every 10 samples
        if idx % 10 == 0:
            print(f"Processing query {idx}/{len(queries_id)} - QID: {qid}")
        
        # 1. Get tabular data
        itm_tab_q = get_tabular_features_filtered_numpy(patient_info_csv, qid)
        if itm_tab_q is None:
            skipped_counts['no_tabular'] += 1
            continue
            
        # 2. Get image filename
        img_filename = get_pre_image_from_id(patient_info_csv, favorite_image_info_csv, 
                                          patient_images_info_csv, qid,
                                          images_resized_path)
        if img_filename is None:
            skipped_counts['no_image'] += 1
            continue
            
        # 3. Verify image files exist
        #original_img_path = os.path.join(images_original_path, os.path.basename(img_filename.replace('_resized.jpg', '.jpg')))
        original_img_path = os.path.join(images_original_path, os.path.basename(img_filename))
        if not os.path.exists(original_img_path):
            print(f"Missing original image: {original_img_path}")
            skipped_counts['no_file'] += 1
            continue
            
        # Initialize elements
        qns_img_element = QNS_structure()
        qns_tab_element = QNS_structure()
        
        # Process resized image path
        resized_img_path = os.path.join(images_resized_path, os.path.basename(img_filename))
        if not os.path.exists(resized_img_path):
            try:
                resize_image(original_img_path, resized_img_path, (224, 224))
                print(f"Created resized image: {resized_img_path}")
            except Exception as e:
                print(f"Failed to resize {original_img_path}: {str(e)}")
                continue
                
        # Set query vectors
        qns_img_element.set_query_vector(resized_img_path, qid)
        qns_tab_element.set_query_vector(itm_tab_q, qid)

        # Process neighbors
        valid_neighbors = 0
        for nid in neighbours_id[idx]:
            itm_tab_n = get_tabular_features_filtered_numpy(patient_info_csv, nid)
            if itm_tab_n is None:
                continue
                
            neighbor_img_filename = get_pre_image_from_id(patient_info_csv, favorite_image_info_csv, 
                                                       patient_images_info_csv, nid,
                                                       images_resized_path)
            if neighbor_img_filename is None:
                continue
                
            neighbor_original = os.path.join(images_original_path, os.path.basename(neighbor_img_filename))
#            neighbor_original = os.path.join(images_original_path, os.path.basename(neighbor_img_filename.replace('_resized.jpg', '.jpg')))
            neighbor_resized = os.path.join(images_resized_path, os.path.basename(neighbor_img_filename))
            
            if not os.path.exists(neighbor_original):
                continue
                
            if not os.path.exists(neighbor_resized):
                try:
                    resize_image(neighbor_original, neighbor_resized, (224, 224))
                except:
                    continue
                    
            qns_img_element.add_neighbor_vector(neighbor_resized, nid)
            qns_tab_element.add_neighbor_vector(itm_tab_n, nid)
            valid_neighbors += 1

        if valid_neighbors > 0:
            qns_img_element.calculate_expert_score()
            QNS_img_list.append(qns_img_element)
            
            qns_tab_element.calculate_expert_score()
            QNS_tab_list.append(qns_tab_element)

    print("\n=== Processing Summary ===")
    print(f"Total queries processed: {len(queries_id)}")
    print(f"Skipped - no tabular data: {skipped_counts['no_tabular']}")
    print(f"Skipped - no image data: {skipped_counts['no_image']}")
    print(f"Skipped - missing files: {skipped_counts['no_file']}")
    print(f"Final valid samples: {len(QNS_img_list)}")
    
    return QNS_img_list, len(QNS_img_list), QNS_tab_list, len(QNS_tab_list)

# Function: Create a new name for the resized samples
def edit_name_incase_using_resized(path, filename):
    basename = os.path.basename(filename)
    resized_filename = os.path.splitext(basename)[0] + '_resized.jpg'
    image_path = os.path.join(path, resized_filename)

    return image_path



# Function: Collaborative tabular normalize
def collaborative_tabular_normalize(qns_list, min_max_values=None):
    if min_max_values is not None:
        vec_len = len(min_max_values)
    else:
        vec_len = len(qns_list[0].query_vector) 
        min_max_values = []

    all_elements = [[] for _ in range(vec_len)]

    # Collecting all elements for each position from both query and neighbor vectors
    for qns in qns_list:
        for i in range(vec_len):
            all_elements[i].append(qns.query_vector[i])
            for neighbor_vector in qns.neighbor_vectors:
                all_elements[i].append(neighbor_vector[i])

    # If min_max_values is provided, use it for normalization
    if min_max_values:
        for i in range(vec_len):
            min_val, max_val = min_max_values[i]
            all_elements[i] = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
    else:
        # Normalizing each position across all instances and storing min-max values
        for i in range(vec_len):
            min_val = np.min(all_elements[i])
            max_val = np.max(all_elements[i])
            all_elements[i]  = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
            min_max_values.append((min_val, max_val))
        print("Min_Max Values are: ", min_max_values)

    # Updating the vectors in QNS_structure instances
    for qns in qns_list:
        for i in range(vec_len):
            qns.query_vector[i] = all_elements[i].pop(0)
            for neighbor_vector in qns.neighbor_vectors:
                neighbor_vector[i] = all_elements[i].pop(0)

    return min_max_values



# Function: Sample manager
def sample_manager(pickles_path):



    # Open files
    with open(os.path.join(pickles_path, 'image_train.pkl'), 'rb') as file:
        QNS_image_list_train = pickle.load(file)
    with open(os.path.join(pickles_path, 'image_test.pkl'), 'rb') as file:
        QNS_image_list_test = pickle.load(file)

    with open(os.path.join(pickles_path, 'tabular_train.pkl'), 'rb') as file:
        QNS_tabular_list_train = pickle.load(file)
    with open(os.path.join(pickles_path, 'tabular_test.pkl'), 'rb') as file:
        QNS_tabular_list_test = pickle.load(file)


    return QNS_image_list_train, QNS_image_list_test, QNS_tabular_list_train, QNS_tabular_list_test


# Function: Data preprocessing
def data_preprocessing(images_resized_path, images_original_path, pickles_path, catalogue_info, 
                      catalogue_user_info, patient_info, favorite_image_info, patient_images_info, 
                      catalogue_type='E', doctor_code=-1, split_ratio=0.8, seed=10):

    # Get images and tabular data
    # QNS_image_list, QNS_image_count = get_query_neighbor_elements_path(catalogue_info, catalogue_user_info, patient_info, favorite_image_info, patient_images_info,catalogue_type=catalogue_type, doctor_code=doctor_code) # 39 57 36 -1
    # QNS_tabular_list, QNS_tabular_count = get_query_neighbor_elements(catalogue_info, catalogue_user_info, patient_info, doctor_code=doctor_code)
    QNS_image_list, QNS_image_count, QNS_tabular_list, QNS_tabular_count = get_query_neighbor_elements_image_tabular(
        catalogue_info_csv=catalogue_info, 
        catalogue_user_info_csv=catalogue_user_info, 
        patient_info_csv=patient_info, 
        favorite_image_info_csv=favorite_image_info, 
        patient_images_info_csv=patient_images_info,
        images_original_path=images_original_path,
        images_resized_path=images_resized_path,  
        catalogue_type=catalogue_type, 
        doctor_code=doctor_code
    )

    # print(QNS_image_count, QNS_tabular_count)
    assert QNS_image_count == QNS_tabular_count, f'len(QNS_image_count):{QNS_image_count} != len(QNS_tabular_count):{QNS_tabular_count}'

    # print('Modifying File Addressing')

    for QNS_element in QNS_image_list:
        # Get original filename (without _resized)
        original_basename = os.path.basename(QNS_element.query_vector).replace('_resized.jpg', '_resized.jpg')#change to adapt to suffix image names
        original_path = os.path.join(images_original_path, original_basename)
        resized_path = QNS_element.query_vector  # This already has correct path
        
        # Only resize if needed
        if not os.path.exists(resized_path):
            try:
                resize_image(original_path, resized_path, (224, 224))
            except Exception as e:
                print(f"Failed to resize {original_path}: {str(e)}")
                continue
        
        # Process neighbors
        for j in range(QNS_element.neighbor_count):
#            neighbor_basename = os.path.basename(QNS_element.neighbor_vectors[j]).replace('_resized.jpg', '.jpg')
            neighbor_basename = os.path.basename(QNS_element.neighbor_vectors[j]).replace('_resized.jpg', '_resized.jpg') #change to adapt to suffix image names
            neighbor_original = os.path.join(images_original_path, neighbor_basename)
            neighbor_resized = QNS_element.neighbor_vectors[j]  # Already has correct path
            
            if not os.path.exists(neighbor_resized):
                try:
                    resize_image(neighbor_original, neighbor_resized, (224, 224))
                except Exception as e:
                    print(f"Failed to resize neighbor {neighbor_original}: {str(e)}")
                    continue
    # Perform train-test split
    QNS_image_list_train, QNS_image_list_test, QNS_tabular_list_train, QNS_tabular_list_test = train_test_split(QNS_image_list, QNS_tabular_list, train_size=split_ratio, random_state=seed)

    # QNS_image_list_train = QNS_image_list[:int(np.floor(split_ratio * QNS_image_count))]
    # QNS_image_list_test  = QNS_image_list[int(np.floor(split_ratio * QNS_image_count)):]
    # QNS_tabular_list_train = QNS_tabular_list[:int(np.floor(split_ratio * QNS_tabular_count))]
    # QNS_tabular_list_test  = QNS_tabular_list[int(np.floor(split_ratio * QNS_tabular_count)):]

    # Save these into pickle files
    with open(os.path.join(pickles_path, 'image_train.pkl'), 'wb') as file:
        pickle.dump(QNS_image_list_train, file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(os.path.join(pickles_path, 'image_test.pkl'), 'wb') as file:
        pickle.dump(QNS_image_list_test, file, protocol=pickle.DEFAULT_PROTOCOL)

    with open(os.path.join(pickles_path, 'tabular_train.pkl'), 'wb') as file:
        pickle.dump(QNS_tabular_list_train, file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(os.path.join(pickles_path, 'tabular_test.pkl'), 'wb') as file:
        pickle.dump(QNS_tabular_list_test, file, protocol=pickle.DEFAULT_PROTOCOL)

    return

# Function: Resize a single image
def resize_image(input_path, output_path, size):
    with Image.open(input_path) as img:
        img_resized = img.resize(size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
        img_resized.save(output_path)

    return

# Function: Resize a list of images
def resize_images(input_dir, output_dir, size=(224, 224)):

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):  # Add more extensions if needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_image(input_path, output_path, size)

    print("All images have been resized and saved.")
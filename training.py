import h5py

# Load the MATLAB file
mat_data = ['./2017-05-12_batchdata_updated_struct_errorcorrect.mat',
            './2017-06-30_batchdata_updated_struct_errorcorrect.mat',
            './2018-04-12_batchdata_updated_struct_errorcorrect.mat']

# Load each file and print data
for file_path in mat_data:
    try:
        with h5py.File(file_path, 'r') as f:
            # List all groups in the file
            print(f"Groups in {file_path}: {list(f.keys())}")

            # Access a specific dataset if you know its name
            # For example, assuming the dataset is named 'data':
            # data = f['data'][()]
            # print(f"Data from {file_path}: {data}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")


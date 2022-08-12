import pickle 
def load_data(f_path):
    with open(f_path, 'rb') as f:
        mynewlist = pickle.load(f)
        return mynewlist
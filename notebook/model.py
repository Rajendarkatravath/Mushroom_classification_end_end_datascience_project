import pickle

# Open the pickle file in 'rb' mode (read binary)
with open('E:/Mushrooms_classfication_project/artifacts/label_encoder.pkl', 'rb') as file:
    # Load the content from the file
    loaded_object = pickle.load(file)

# Now you can use the loaded object
print(loaded_object)
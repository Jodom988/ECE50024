
transfer_train_cropped_single:
	python3 transfer_learn.py ../data/HW5/train_train_cropped_single_80_20.csv ../data/HW5/train_test_cropped_single_80_20.csv ../data/HW5/train_cropped_single/ ../data/HW5/train_cropped_single/ -b 32 -n -1

transfer_train_cropped_single_mutated:
	python3 transfer_learn.py ../data/HW5/train_train_cropped_single_mutated_80_20.csv ../data/HW5/train_test_cropped_single_80_20.csv ../data/HW5/train_cropped_single_mutated/ ../data/HW5/train_cropped_single/ -b 32 -n -1


# Crop the faces from the TRAINING data set with LARGEST face policy
crop_train_largest:
	# python3 crop_faces.py ../data/HW5/train_train_80_20.csv ../data/HW5/train ../data/HW5/train_cropped_largest/ ../data/HW5/train_train_cropped_largest_80_20.csv largest

# Crop the faces from the TESTING data set with LARGEST face policy
crop_test_largest:
	# python3 crop_faces.py ../data/HW5/train_test_80_20.csv ../data/HW5/train ../data/HW5/train_cropped_largest/ ../data/HW5/train_test_cropped_largest_80_20.csv largest

# Crop the faces from the TRAINING data set with SINGLE face policy
crop_train_single:
	# python3 crop_faces.py ../data/HW5/train_train_80_20.csv ../data/HW5/train ../data/HW5/train_cropped_single_mutated/ ../data/HW5/train_train_cropped_single_mutated_80_20.csv single

# Crop the faces from the TRAINING data set with SINGLE face policy
crop_test_single:
	# python3 crop_faces.py ../data/HW5/train_test_80_20.csv ../data/HW5/train ../data/HW5/train_cropped_single/ ../data/HW5/train_test_cropped_single_80_20.csv single
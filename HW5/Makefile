
small_naive_train:
	python3 naive_method.py ../data/HW5/train_train_80_20.csv ../data/HW5/train_test_80_20.csv ../data/HW5/train/ -b 32 -n 100 -p 1

naive_train:
	python3 naive_method.py ../data/HW5/train_train_80_20.csv ../data/HW5/train_test_80_20.csv ../data/HW5/train/ -b 32 -n -1

naive_train_cropped:
	python3 naive_method.py ../data/HW5/train_train_cropped_80_20.csv ../data/HW5/train_test_cropped_80_20.csv ../data/HW5/train_cropped/ -b 32 -n -1

naive_train_cropped_ex:
	python3 naive_method.py ../data/HW5/train_train_cropped_80_20.csv ../data/HW5/train_test_cropped_80_20.csv ../data/HW5/train_cropped_ex/ -b 32 -n -1

transfer_train:
	python3 transfer_learn.py ../data/HW5/train_train_80_20.csv ../data/HW5/train_test_80_20.csv ../data/HW5/train/ -b 32 -n -1

transfer_train_cropped_ex:
	python3 transfer_learn.py ../data/HW5/train_train_cropped_80_20.csv ../data/HW5/train_test_cropped_80_20.csv ../data/HW5/train_cropped_ex/ -b 32 -n -1

crop_train:
	python3 crop_faces.py ../data/HW5/train_train_80_20.csv ../data/HW5/train/ ../data/HW5/train_cropped_ex/ ../data/HW5/train_train_cropped_ex_80_20.csv

crop_test:
	python3 crop_faces.py ../data/HW5/train_test_80_20.csv ../data/HW5/train/ ../data/HW5/train_cropped_ex/ ../data/HW5/train_test_cropped_ex_80_20.csv
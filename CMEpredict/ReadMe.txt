In this zip, there are csv files named normalized_training_x, which contain training data samples (after normalization) used for predicting CMEs within the next x hours (x = 12, 24, 36, 48 or 60). In addition, there are csv files named normalized_testing_x, which contain testing data samples (after normalization) within the next x hours (x = 12, 24, 36, 48 or 60) of active region #12497 and #12529.

Each file has 22 columns. 
The first column is titled Label. This column has 3 values: padding, N, and P. Padding means this is an auxiliary data sample used to construct time series for prediction. N means there is a >=M class flare within the next x hours but the flare is not associated with a CME. P means there is a >=M class flare within the next x hours and this flare is associated with a CME.
The second column is titled Timestamp. The third column and fourth column are titled NOAA active region number and HARP number, respectively. Starting from the fifth column, you can see physical parameters of data samples, which include 18 SHARP parameters: TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, MEANGAM, MEANJZH, MEANGBT, MEANGBZ, MEANJZD, MEANGBH, MEANSHR, MEANALP. 

This zip also contains the source code of our program, called CMEpredict.py, which is used to predict labels of testing data samples. 

The usage is given as follows:
	python3 CMEpredict.py gru 12 0

The first argument "CMEpredict.py" is the Python program file name.
The second argument "gru" denotes that the program will make predictions using GRU. 

An alternative option is "lstm" which uses LSTM. The usage is given as follows:
	python3 CMEpredict.py lstm 12 0

The third argument "12" denotes that the program will predict CMEs within the next 12 hours. Other options are 24, 36, 48 or 60 hours.
The fourth argument "0" denotes that the program will load and use the pre-trained model, named gru-x-model.h5 or lstm-x-model.h5. If one would like to re-train the model, change "0" to "1".

The output obtained by executing the above command is stored in the file named gru-x-output.csv or lstm-x-output.csv in the zip. This output file is the same as the normalized_testing_x file except that it has one additional column (the first column) titled "Predicted Label," which contains labels predicted by our program. The value "padding" is removed from the output file.

Our program is run on Python 3.6.8, Keras 2.2.4, and TensorFlow 1.12.0.


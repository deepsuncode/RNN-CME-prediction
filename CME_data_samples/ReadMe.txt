This zip contains one folder, named CME_data_samples, containing data samples shown in Table 1 of the paper. In the folder, there are two types of csv files named normalized_training_x and normalized_testing_x respectively, which contain training and testing data samples (after normalization) used for predicting CMEs within the next x hours (x = 12, 24, 36, 48 or 60).

Each file has 22 columns.
The first column is titled Label. This column has 2 values: N, and P. N means there is a >=M class flare within the next x hours but this flare is not associated with a CME. P means there is a >=M class flare within the next x hours and this flare is associated with a CME.
The second column is titled Timestamp. The third column and fourth column are titled NOAA active region number and HARP number, respectively. Starting from the fifth column, you can see physical parameters of data samples, which include 18 SHARP parameters: TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, MEANGAM, MEANJZH, MEANGBT, MEANGBZ, MEANJZD, MEANGBH, MEANSHR, MEANALP.


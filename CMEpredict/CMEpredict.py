# =========================================================================
#   (c) Copyright 2020
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import pandas as pd
from sklearn.utils import class_weight
from keras.models import *
from keras.layers import *
import numpy as np
import sys
import csv
import os
import warnings


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turn off loggins is not supported')


def load_data(datafile, series_len, start_feature, n_features, mask_value, type, time_window):
    df = pd.read_csv(datafile, header=None)
    df_values0 = df.values
    if type == 'gru':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 7, 8, 15, 18, 21, 6, 9, 10, 17, 5, 16, 4, 12, 19, 20, 14]]  # 12   GRU
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 15, 5, 20, 9, 21, 7, 8, 6, 17, 18, 10, 14, 4, 12, 16, 19]]  # 24   GRU
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 21, 15, 8, 7, 4, 6, 14, 12, 17, 10, 18, 16, 19]]  # 36   GRU
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 14, 8, 7, 21, 6, 4, 15, 12, 17, 16, 10, 18, 19]]  # 48   GRU
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   GRU
    elif type == 'lstm':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 20, 7, 15, 8, 21, 6, 18, 5, 10, 9, 17, 16, 19, 12, 14, 4]]  # 12   LSTM
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 20, 11, 13, 9, 15, 14, 8, 7, 5, 21, 6, 17, 18, 10, 12, 16, 4, 19]]  # 24   LSTM
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 20, 13, 5, 14, 8, 15, 7, 9, 21, 6, 4, 12, 17, 18, 10, 16, 19]]  # 36   LSTM
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 20, 13, 9, 14, 7, 15, 8, 6, 4, 21, 12, 17, 18, 16, 10, 19]]  # 48   LSTM
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   LSTM
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    n_neg = 0
    n_pos = 0
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        label = row[0]
        if label == 'padding':
            continue
        has_zero_record = False
        # if one of the physical feature values is missing, then discard it.
        for k in range(start_feature, start_feature + n_features):
            if float(row[k]) == 0.0:
                has_zero_record = True
                break

        if has_zero_record is False:
            cur_harp_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_harp_num = int(prev_row[3])
                if prev_harp_num != cur_harp_num:
                    break
                has_zero_record_tmp = False
                for k in range(start_feature, start_feature + n_features):
                    if float(prev_row[k]) == 0.0:
                        has_zero_record_tmp = True
                        break
                if float(prev_row[-5]) >= 3500 or float(prev_row[-4]) >= 65536 or \
                        abs(float(prev_row[-1]) - float(prev_row[-2])) > 70:
                    has_zero_record_tmp = True

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if (label == 'N' or label == 'P') and len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                if label == 'N':
                    y.append(0)
                    n_neg += 1
                elif label == 'P':
                    y.append(1)
                    n_pos += 1
    X_arr = np.array(X)
    y_arr = np.array(y)
    nb = n_neg + n_pos
    return X_arr, y_arr, nb


def attention_3d_block(hidden_states, series_len):
    hidden_size = int(hidden_states.shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
    hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
    score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


def lstm(n_features, series_len):
    inputs = Input(shape=(series_len, n_features,))
    lstm_out = LSTM(10, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(inputs)
    attention_mul = attention_3d_block(lstm_out, series_len)
    layer1 = Dense(100, activation='relu')(attention_mul)
    layer1 = Dropout(0.25)(layer1)
    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.0001))(layer1)
    model = Model(inputs=[inputs], outputs=output)
    return model


def gru(n_features, series_len):
    inputs = Input(shape=(series_len, n_features,))
    lstm_out = GRU(10, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(inputs)
    attention_mul = attention_3d_block(lstm_out, series_len)
    layer1 = Dense(100, activation='relu')(attention_mul)
    layer1 = Dropout(0.25)(layer1)
    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.0001))(layer1)
    model = Model(inputs=[inputs], outputs=output)
    return model


def output_result(test_data_file, result_file, type, time_window, start_feature, n_features, thresh):
    df = pd.read_csv(test_data_file, header=None)
    df_values0 = df.values
    if type == 'gru':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 7, 8, 15, 18, 21, 6, 9, 10, 17, 5, 16, 4, 12, 19, 20, 14]]  # 12   GRU
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 15, 5, 20, 9, 21, 7, 8, 6, 17, 18, 10, 14, 4, 12, 16, 19]]  # 24   GRU
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 21, 15, 8, 7, 4, 6, 14, 12, 17, 10, 18, 16, 19]]  # 36   GRU
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 14, 8, 7, 21, 6, 4, 15, 12, 17, 16, 10, 18, 19]]  # 48   GRU
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   GRU
    elif type == 'lstm':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 20, 7, 15, 8, 21, 6, 18, 5, 10, 9, 17, 16, 19, 12, 14, 4]]  # 12   LSTM
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 20, 11, 13, 9, 15, 14, 8, 7, 5, 21, 6, 17, 18, 10, 12, 16, 4, 19]]  # 24   LSTM
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 20, 13, 5, 14, 8, 15, 7, 9, 21, 6, 4, 12, 17, 18, 10, 16, 19]]  # 36   LSTM
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 20, 13, 9, 14, 7, 15, 8, 6, 4, 21, 12, 17, 18, 16, 10, 19]]  # 48   LSTM
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   LSTM
    with open(result_file, 'w', encoding='UTF-8') as result_csv:
        w = csv.writer(result_csv)
        w.writerow(['Predicted Label', 'Label', 'Timestamp', 'NOAA AR NUM', 'HARP NUM',
                      'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR',
                      'MEANPOT', 'R_VALUE', 'SHRGT45', 'MEANGAM', 'MEANJZH', 'MEANGBT', 'MEANGBZ',
                      'MEANJZD', 'MEANGBH', 'MEANSHR', 'MEANALP'])
        idx = 0
        for i in range(len(df_values)):
            line = df_values[i].tolist()
            if line[0] == 'padding' or float(line[-5]) >= 3500 or float(line[-4]) >= 65536 \
                    or abs(float(line[-1]) - float(line[-2])) > 70:
                continue
            has_zero_record = False
            # if one of the physical feature values is missing, then discard it.
            for k in range(start_feature, start_feature + n_features):
                if float(line[k]) == 0.0:
                    has_zero_record = True
                    break
            if has_zero_record:
                continue
            if prob[idx] >= thresh:
                line.insert(0, 'P')
            else:
                line.insert(0, 'N')
            idx += 1
            w.writerow(line)


def get_n_features_thresh(type, time_window):
    n_features = 0
    thresh = 0
    if type == 'gru':
        if time_window == 12:
            n_features = 16
            thresh = 0.45
        elif time_window == 24:
            n_features = 12
            thresh = 0.4
        elif time_window == 36:
            n_features = 9
            thresh = 0.45
        elif time_window == 48:
            n_features = 14
            thresh = 0.45
        elif time_window == 60:
            n_features = 5
            thresh = 0.5
    elif type == 'lstm':
        if time_window == 12:
            n_features = 15
            thresh = 0.4
        elif time_window == 24:
            n_features = 12
            thresh = 0.45
        elif time_window == 36:
            n_features = 8
            thresh = 0.45
        elif time_window == 48:
            n_features = 15
            thresh = 0.45
        elif time_window == 60:
            n_features = 6
            thresh = 0.5
    return n_features, thresh


if __name__ == '__main__':
    type = sys.argv[1]
    time_window = int(sys.argv[2])
    train_again = int(sys.argv[3])
    train_data_file = './normalized_training_' + str(time_window) + '.csv'
    test_data_file = './normalized_testing_' + str(time_window) + '.csv'
    result_file = './' + type + '-' + str(time_window) + '-output.csv'
    model_file = './' + type + '-' + str(time_window) + '-model.h5'
    start_feature = 4
    n_features, thresh = get_n_features_thresh(type, time_window)
    mask_value = 0
    series_len = 20
    epochs = 20
    batch_size = 256
    nclass = 2

    if train_again == 1:
        # Train
        print('loading training data...')
        X_train, y_train, nb_train = load_data(datafile=train_data_file,
                                               series_len=series_len,
                                               start_feature=start_feature,
                                               n_features=n_features,
                                               mask_value=mask_value,
                                               type=type,
                                               time_window=time_window)

        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weight_ = {0: class_weights[0], 1: class_weights[1]}
        print('done loading training data...')

        if type == 'gru':
            model = gru(n_features, series_len)
        elif type == 'lstm':
            model = lstm(n_features, series_len)
        print('training the model, wait until it is finished...')
        model.compile(loss='binary_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])

        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=False,
                            shuffle=True,
                            class_weight=class_weight_)
        print('finished...')
        model.save(model_file)
    else:
        print('loading model...')
        model = load_model(model_file)
        print('done loading...')

    # Test
    print('loading testing data')
    X_test, y_test, nb_test = load_data(datafile=test_data_file,
                                        series_len=series_len,
                                        start_feature=start_feature,
                                        n_features=n_features,
                                        mask_value=mask_value,
                                        type=type,
                                        time_window=time_window)
    print('done loading testing data...')
    print('predicting testing data...')
    prob = model.predict(X_test,
                         batch_size=batch_size,
                         verbose=False,
                         steps=None)
    print('done predicting...')
    print('writing prediction results into file...')
    output_result(test_data_file=test_data_file,
                  result_file=result_file,
                  type=type,
                  time_window=time_window,
                  start_feature=start_feature,
                  n_features=n_features,
                  thresh=thresh)
    print('done...')

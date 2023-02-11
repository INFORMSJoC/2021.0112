import json
import numpy as np
import tensorflow.compat.v1 as tf
from myutil import get_split_idx, generate_time_slices, part_data_time_slice

from gcn.inits import glorot, zeros

tf.disable_eager_execution()


def normalize_adj(adj, symmetric=True):
    '''
    Input Arguments:
    @adj: adjacency matrix with self connection (value 1 for diaganal elements)
    @symmetric: whether do symmetric normalization
    Return:
    Normalized adjacency matrix, if symmetric is True, return D^{-1/2}AD^{-1/2}, else return D^{-1}A
    '''
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm


def preprocess_adj(adj, symmetric=True):
    '''
    Input Arguments:
    @adj: raw adjacency matrix
    @symmetric: whether do symmetric normalization
    Return:
    Normalized adjacency matrix 
    '''
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def get_data(ratios, interested_clocks, prior_days, prior_hours, usesample=True):
    '''
    Get a dict of training data with following keys:
    'T': total number of observations along time
    'N': number of nodes (road segments)
    'D': dimension of time series features
    'DW': dimension of weather data
    'train_x_day': training data of prior days
    'train_x_hour': training data of prior hours
    'train_x_weather_day': training weather data of prior days
    'train_x_weather_hour': training weather data of prior hours
    'train_y': label of training data
    'val_x_day': validation data of prior days
    'val_x_hour': validation data of prior hours
    'val_x_weather_day': validation weather data of prior days
    'val_x_weather_hour': validation weather data of prior hours
    'val_y': label of validation data
    'test_x_day': test data of prior days
    'test_x_hour': test data of prior hours
    'test_x_weather_day': test weather data of prior days
    'test_x_weather_hour': test weather data of prior hours
    'test_y': label of test data
    'adj': adjacency matrix of road segments.
    Input Arguments:
    @ratios list of 3 float numbers representation portions for training, validation and test.
    @interested_clocks: a list of number in range(0,24) to indicate which clock observations for prior days are used as input features.
    @prior_days: number of prior days data used as input
    @prior_hours: number of most recent hours used as input
    @usesample: whether downsample the road network, if True, loading file '../data/sample_input/sample_roads_5000.txt' which is a np array of indexes selected.
    '''
    import pandas as pd
    import scipy.sparse as sparse

    whole_data = np.load('../data/sample_input/sample_input.npy')

    # sample_segments_filter = np.loadtxt('../output/sample_sanhuan_data/sample_segment_idx_in_sihuan_file.txt', dtype=int)

    if usesample:
        sample_segments_filter = np.loadtxt('../data/sample_input/sample_roads_5000.txt', dtype=int)
    else:
        sample_segments_filter = np.arange(whole_data.shape[1])

    # label idx should be 0 or 2, 0 is demand, 2 is pickup prob

    whole_data = whole_data[:, sample_segments_filter]

    x = whole_data[:-1]

    y_demand = whole_data[1:, :, 0]
    y_supply = whole_data[1:, :, 1]
    y_pickup_prob = whole_data[1:, :, 2]
    y = y_pickup_prob

    T, N, D = x.shape

    weather_data_columns = ['datetime', 'apparentTemperature', 'dewPoint',
                            'humidity', 'summary', 'temperature', 'visibility', 'windSpeed']
    weather_data = pd.read_csv('../data/weather.csv',
                               parse_dates=['datetime'],
                               usecols=weather_data_columns)
    weather_data = weather_data[weather_data.datetime < '2013-12-18']
    weather_data = pd.get_dummies(weather_data, columns=['summary'])
    weather_data = weather_data[[c for c in weather_data.columns if c != 'datetime']].values
    x_weather = weather_data[:-1]
    date_indicator = np.loadtxt('../data/sample_input/date_indicator.txt')[:-1]
    x_weather = np.concatenate([x_weather, date_indicator], axis=1)

    adjs = sparse.load_npz('../data/sample_input/sanhuan_network_adjacent_matrix_symmetric_sparse.npz')
    adjs = adjs[sample_segments_filter][:, sample_segments_filter]
    adjs = preprocess_adj(adjs)

    # split time based on day
    train_idx, val_idx, test_idx = get_split_idx(round(T / 24), ratios)
    # print('T', T, 'T/24', T/24)
    train_day_slices, train_hour_slices = generate_time_slices(T, prior_days, prior_hours, interested_clocks,
                                                               train_idx[0], train_idx[1])
    val_day_slices, val_hour_slices = generate_time_slices(T, prior_days, prior_hours, interested_clocks, val_idx[0],
                                                           val_idx[1])
    test_day_slices, test_hour_slices = generate_time_slices(T, prior_days, prior_hours, interested_clocks, test_idx[0],
                                                             test_idx[1])

    train_x_day, train_x_weather_day, _ = part_data_time_slice(train_day_slices, x, x_weather, y)
    train_x_hour, train_x_weather_hour, train_y = part_data_time_slice(train_hour_slices, x, x_weather, y)

    val_x_day, val_x_weather_day, _ = part_data_time_slice(val_day_slices, x, x_weather, y)
    val_x_hour, val_x_weather_hour, val_y = part_data_time_slice(val_hour_slices, x, x_weather, y)

    test_x_day, test_x_weather_day, _ = part_data_time_slice(test_day_slices, x, x_weather, y)
    test_x_hour, test_x_weather_hour, test_y = part_data_time_slice(test_hour_slices, x, x_weather, y)

    return {
        'T': T,
        'N': N,
        'D': D,
        'DW': x_weather.shape[1],
        'train_x_day': train_x_day,
        'train_x_hour': train_x_hour,
        'train_x_weather_day': train_x_weather_day,
        'train_x_weather_hour': train_x_weather_hour,
        'train_y': train_y,
        'val_x_day': val_x_day,
        'val_x_hour': val_x_hour,
        'val_x_weather_day': val_x_weather_day,
        'val_x_weather_hour': val_x_weather_hour,
        'val_y': val_y,
        'test_x_day': test_x_day,
        'test_x_hour': test_x_hour,
        'test_x_weather_day': test_x_weather_day,
        'test_x_weather_hour': test_x_weather_hour,
        'test_y': test_y,
        'adj': adjs
    }

class GCNLSTM_SPLIT(object):
    '''
    Graph Convolution LSTM with separate Graph Convolution layer and LSTM layer for prior days and prior hours.
    '''
    def __init__(self, N, n_days, n_hours, input_dim, weather_dim, days_gc_dims, hours_gc_dims, days_lstm_dims,
                 hours_lstm_dims, dense_dims, learning_rate=0.01, dropout=.2, act=tf.nn.relu):
        self.N = N
        self.n_days = n_days
        self.n_hours = n_hours
        self.input_dim = input_dim
        self.weather_dim = weather_dim
        self.days_gc_dims = days_gc_dims
        self.hours_gc_dims = hours_gc_dims
        self.days_lstm_dims = days_lstm_dims
        self.hours_lstm_dims = hours_lstm_dims
        self.dense_dims = dense_dims
        self.learning_rate = learning_rate

        self.var = dict()
        self.placeholders = dict()

        # B/T/N/D
        self.input_days = tf.placeholder(tf.float32, shape=(None, n_days, N, input_dim), name='input_days')
        self.input_hours = tf.placeholder(tf.float32, shape=(None, n_hours, N, input_dim), name='input_hours')

        # N/N
        self.support = tf.placeholder(tf.float32,
                                      shape=(N, N),
                                      name='support')
        # self.support = tf.sparse_placeholder(tf.float32)
        # B/T/DW
        self.weather_input_days = tf.placeholder(tf.float32, shape=(None, n_days, weather_dim))
        self.weather_input_hours = tf.placeholder(tf.float32, shape=(None, n_hours, weather_dim))

        # N/
        self.label = tf.placeholder(tf.float32, shape=(None, N), name='label')

        last_dim = input_dim
        days_gc_weights, days_gc_biases = [], []
        for i, gc_dim in enumerate(days_gc_dims):
            days_gc_weights.append(glorot((last_dim, gc_dim), name='days_gc_weight_%d' % i))
            days_gc_biases.append(zeros([gc_dim], name='days_gc_bias_%d' % i))
            last_dim = gc_dim
        self.var['days_gc_weights'] = days_gc_weights
        self.var['days_gc_biases'] = days_gc_biases

        last_dim = input_dim
        hours_gc_weights, hours_gc_biases = [], []
        for i, gc_dim in enumerate(hours_gc_dims):
            hours_gc_weights.append(glorot((last_dim, gc_dim), name='hours_gc_weight_%d' % i))
            hours_gc_biases.append(zeros([gc_dim], name='hours_gc_bias_%d' % i))
            last_dim = gc_dim
        self.var['hours_gc_weights'] = hours_gc_weights
        self.var['hours_gc_biases'] = hours_gc_biases

        last_dim = days_lstm_dims[-1] + hours_lstm_dims[-1]
        dense_weights = []
        dense_biases = []
        for i, dense_dim in enumerate(dense_dims):
            dense_weights.append(glorot((last_dim, dense_dim), name='dense_weight_%d' % i))
            dense_biases.append(zeros([dense_dim], name='dense_bias_%d' % i))
            last_dim = dense_dim
        self.var['dense_weights'] = dense_weights
        self.var['dense_biases'] = dense_biases

        self.dropout = dropout

        # self.train_op = None
        # self.output = None
        # self.loss = None
        # self.optimizer = None
        self.act = act

        self._build()

    def graph_conv(self, weights, biases, gc_input, prefix):
        '''
        Generate graph convolution layer
        '''

        for i, (gc_weight, gc_bias) in enumerate(zip(weights, biases)):
            gc_input = tf.nn.dropout(gc_input, rate=self.dropout)

            # B/T/N/DI dot DI/GD -> B/T/N/GD
            pre_sup = tf.tensordot(gc_input, gc_weight, axes=1, name='%s_pre_sup_%d' % (prefix, i))

            # N/N dot B/T/N/GD -> N/B/T/GD
            supports = tf.tensordot(self.support, pre_sup, axes=[[1], [2]], name='%s_supports_%d' % (prefix, i))

            # N/B/T/GD -> B/T/N/GD
            gc_output = tf.transpose(supports, [1, 2, 0, 3], name='%s_gc_output_%d' % (prefix, i))

            gc_output += gc_bias

            gc_input = gc_output

        return gc_output

    def weather_reshape(self, weather_input, N):
        '''
        Reshape weather data from B/T/DW to B/T/N/DW
        '''
        # B/T/DW -> B/T/1/DW
        weather_input = tf.expand_dims(weather_input, 2)
        # B/T/1/DW -> B/T/N/DW
        weather_input = tf.tile(weather_input, (1, 1, self.N, 1))
        return weather_input

    def _build(self):
        '''
        Building GCN LSTM split model.
        '''
        from tensorflow.keras.layers import LSTMCell

        # x = tf.nn.dropout(x, 1 - self.dropout)
        self.mode = tf.placeholder(tf.string, name='mode')

        # B/T/N/GD
        gc_output_days = self.graph_conv(self.var['days_gc_weights'], self.var['days_gc_biases'], self.input_days,
                                         'days')
        gc_output_hours = self.graph_conv(self.var['hours_gc_weights'], self.var['hours_gc_biases'], self.input_hours,
                                          'hours')

        # B/T/N/DW
        weather_input_days = self.weather_reshape(self.weather_input_days, self.N)
        weather_input_hours = self.weather_reshape(self.weather_input_hours, self.N)

        lstm_input_days = tf.concat([gc_output_days, weather_input_days], axis=3)
        # B/T/N/GD -> B/N/T/GD
        lstm_input_days = tf.transpose(lstm_input_days, (0, 2, 1, 3))
        lstm_input_days = tf.reshape(lstm_input_days, (-1, self.n_days, self.days_gc_dims[-1] + self.weather_dim))

        lstm_input_hours = tf.concat([gc_output_hours, weather_input_hours], axis=3)
        # B/T/N/GD -> B/N/T/GD
        lstm_input_hours = tf.transpose(lstm_input_hours, (0, 2, 1, 3))
        lstm_input_hours = tf.reshape(lstm_input_hours, (-1, self.n_hours, self.hours_gc_dims[-1] + self.weather_dim))

        for i, unit in enumerate(self.days_lstm_dims):
            lstm_output_days, state = tf.nn.dynamic_rnn(
                LSTMCell(unit, name='days_lstm_cell_%d' % i), lstm_input_days, dtype=tf.float32,
                scope='days_rnn_%d' % i)
            lstm_input_days = lstm_output_days
        lstm_out_days = lstm_output_days[:, -1]

        for i, unit in enumerate(self.hours_lstm_dims):
            lstm_output_hours, state = tf.nn.dynamic_rnn(
                LSTMCell(unit, name='hours_lstm_cell_%d' % i), lstm_input_hours, dtype=tf.float32,
                scope='hours_rnn_%d' % i)
            lstm_input_hours = lstm_output_hours
        lstm_out_hours = lstm_output_hours[:, -1]

        lstm_out = tf.concat([lstm_out_days, lstm_out_hours], axis=-1)

        # BN/U -> B/N/U
        dense_input = tf.reshape(lstm_out, (-1, self.N, self.days_lstm_dims[-1] + self.hours_lstm_dims[-1]))

        for i, (dense_weight, dense_bias) in enumerate(
                zip(self.var['dense_weights'][:-1], self.var['dense_biases'][:-1])):
            dense_output = tf.tensordot(dense_input, dense_weight, axes=1)
            dense_output = self.act(dense_output + dense_bias)
            dense_input = dense_output

        dense_output = tf.tensordot(dense_input, self.var['dense_weights'][-1], axes=1)
        dense_output += self.var['dense_biases'][-1]

        self.output = tf.reshape(dense_output, (-1, self.N))

        self.loss = tf.losses.mean_squared_error(self.label, self.output)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)

        self.mae = tf.metrics.mean_absolute_error(self.label, self.output)


def construct_feed_dict(model, X_input_days, X_input_hours, X_weather_days, X_weather_hours, y, adj, mode):
    d = {
        model.input_days: X_input_days,
        model.input_hours: X_input_hours,
        model.label: y,
        model.support: adj,
        model.mode: mode,
        model.weather_input_days: X_weather_days,
        model.weather_input_hours: X_weather_hours
    }
    return d


def train_test(model, data, flags):
    '''
    train GCNLSTMSplit model, and test on test data.
    '''
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        val_dict = construct_feed_dict(model, data['val_x_day'], data['val_x_hour'], data['val_x_weather_day'],
                                       data['val_x_weather_hour'], data['val_y'], data['adj'], 'val')

        test_dict = construct_feed_dict(model, data['test_x_day'], data['test_x_hour'], data['test_x_weather_day'],
                                        data['test_x_weather_hour'], data['test_y'], data['adj'], 'test')

        train_dict = construct_feed_dict(model, data['train_x_day'], data['train_x_hour'], data['train_x_weather_day'],
                                         data['train_x_weather_hour'], data['train_y'], data['adj'], 'train')

        val_losses = []

        train_losses = []

        for step in range(flags.epochs):

            train_loss, _ = sess.run([model.loss, model.train_op], train_dict)

            train_losses.append(train_loss)

            val_loss = sess.run(model.loss, val_dict)

            val_losses.append(val_loss)

            print('training step', step, 'training loss', train_loss, 'validation loss', val_loss)

            if len(val_losses) > flags.early_stopping and np.mean(val_losses[-flags.early_stopping:]) < val_loss:
                print('optimization complete')
                break
            elif len(val_losses) > flags.early_stopping:
                print('early stop mean loss {:.5f}'.format(np.mean(val_losses[-flags.early_stopping:])))

        output = sess.run(model.output, test_dict)

        pred = np.clip(output, 0, 1)

        from sklearn.metrics import mean_absolute_error
        label = test_dict[model.label]
        test_mae = mean_absolute_error(label, pred)

        return pred, label, test_mae, train_losses, val_losses, (label.min(), label.max()), (pred.min(), pred.max())


# for sparse tensor dot refer to [this page](https://github.com/tensorflow/tensorflow/issues/9210)



if __name__ == '__main__':
    # Settings
    from myutil import dump_summary
    from myutil import file_base_name, file_time_stamp
    from argparse import ArgumentParser

    def numeric_list_parser(s):
        return [int(i) for i in s.split('/')]

    parser = ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-p', '--dropout', type=float, default=0.1)
    parser.add_argument('-e', '--epochs', type=int, default=400)
    parser.add_argument('-s', '--early_stopping', type=int, default=10)
    parser.add_argument('-g', '--gc_dims', type=numeric_list_parser, default=[16, 16])
    parser.add_argument('-u', '--lstm_dims', type=numeric_list_parser, default=[8, 8])
    parser.add_argument('-d', '--dense_dims', type=numeric_list_parser, default=[4, 1])
    parser.add_argument('-a', '--prior_days', type=int, default=3)
    parser.add_argument('-o', '--prior_hours', type=int, default=4)
    parser.add_argument('-c', '--interested_clocks', type=numeric_list_parser, default=list(range(24)))
    parser.add_argument('-m', '--usesample', type=int, choices=[0, 1], default=1)

    args = parser.parse_args()

    prior_days = args.prior_days
    prior_hours = args.prior_hours
    data = get_data((.65, .1, .25), args.interested_clocks, prior_days, prior_hours, args.usesample==1)

    N, D = data['N'], data['D']

    weather_dim = data['DW']

    gc_dims = args.gc_dims
    #
    lstm_dims = args.lstm_dims
    #
    dense_dims = args.dense_dims

    regressor = GCNLSTM_SPLIT(N, prior_days, prior_hours, D, weather_dim, gc_dims, gc_dims, lstm_dims, lstm_dims,
                              dense_dims)

    predicted, label, mae, train_losses, val_losses, label_range, pred_range = train_test(regressor, data, args)

    timpstamp, model_name, model_time_str = file_time_stamp(__file__)

    columns = ['date', 'net_size', 'interested_clocks', 'prior_days', 'prior_hours', 'learning_rate', 'dropout', 'gc_dims', 'lstm_dims', 'dense_dims',
               'label_range', 'range', 'mae']

    result = [timpstamp, N, args.interested_clocks, prior_days, prior_hours, args.learning_rate, args.dropout, gc_dims, lstm_dims, dense_dims, label_range,
              pred_range, mae]

    dump_summary('../results/summary_result_of_{}.csv'.format(model_name), result, cols=columns, add_date=False)

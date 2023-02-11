def file_time_stamp(filename):
    from datetime import datetime
    import os.path as op
    timpstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = op.basename(filename)[:-3]
    model_time_str = timpstamp + '_' + op.basename(filename)[:-3]
    return timpstamp, model_name, model_time_str


def dump_summary(path, content, cols=None, add_date=True):
    from datetime import datetime
    import os.path as op

    def quote_filter(s):
        if ',' in s:
            return '"{}"'.format(s)
        else:
            return s

    if add_date:
        content.insert(0, datetime.now().strftime('%Y-%m-%d'))

    content = [quote_filter(str(c)) for c in content]

    file_existed = op.exists(path)

    with open(path, 'a+') as f:
        if not file_existed:
            f.write(','.join(cols) + '\n')

        f.write(','.join(content) + '\n')


def file_base_name(name):
    import os
    s = os.path.basename(name)
    return s[:s.rindex('.')]


def get_split_idx(T, ratios):
    import numpy as np
    lens = np.round(T * np.array(ratios))
    lens[-1] = T - lens[:-1].sum()
    lens = lens.astype(int)

    idxes = np.cumsum(lens).tolist()
    idxes.insert(0, 0)
    return list(zip(idxes, idxes[1:]))


def part_data_time_slice(time_slices, x, x_weather, y):
    import numpy as np
    ox = []
    oxweather = []
    oy = []
    for time_slice in time_slices:
        ox.append(x[time_slice])
        oxweather.append(x_weather[time_slice])
        oy.append(y[time_slice[-1]])

    ox = np.stack(ox)
    oxweather = np.stack(oxweather)
    oy = np.stack(oy)

    return ox, oxweather, oy


def generate_time_slices(T, prior_days, prior_hours, interested_clocks, begin, end):
    import os.path as op
    import pickle
    import numpy as np

    day_slice_file_path = '../data/day_T{}_d{}_h{}_i{}_b{}_e{}'.format(T, prior_days, prior_hours,
                                                                       '-'.join([str(i) for i in interested_clocks]),
                                                                       begin,
                                                                       end)
    hour_slice_file_path = '../data/hour_T{}_d{}_h{}_i{}_b{}_e{}'.format(T, prior_days, prior_hours,
                                                                         '-'.join([str(i) for i in interested_clocks]),
                                                                         begin,
                                                                         end)

    if op.exists(day_slice_file_path) and op.exists(hour_slice_file_path):
        return pickle.load(open(day_slice_file_path, 'rb')), pickle.load(open(hour_slice_file_path, 'rb'))

    interested_hours_list = [np.arange(interested_clock, T, 24) for interested_clock in interested_clocks]

    day_time_slices, hour_slices = [], []

    for i in range(begin, end):
        if i - prior_days < 0:
            continue
        for interested_hours in interested_hours_list:
            if len(interested_hours) > i:
                day_time_slices.append([interested_hours[j] for j in range(i - prior_days, i)])
                hour_slices.append(list(
                    range(interested_hours[i] - prior_hours, interested_hours[i])))

    pickle.dump(day_time_slices, open(day_slice_file_path, 'wb'))
    pickle.dump(hour_slices, open(hour_slice_file_path, 'wb'))

    return day_time_slices, hour_slices


if __name__ == '__main__':
    # print(auc([1, 1, 2, 2], [0.1, 0.4, 0.35, 0.8],2))

    print(__file__)

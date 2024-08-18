# -*- coding = utf-8 -*-
# seismic data preprocess
# 2024.07.01,edit by Lin

import os
import datetime
from obspy import read
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel
from math import radians, cos, sin, asin, sqrt
from core.train_data import round_up


def check_data(paths):
    e_dict = {}
    for path in paths:
        files = os.listdir(path)
        if not files:
            continue
        e_type_name = os.path.basename(path)
        e_dict[e_type_name] = {}
        e_dict[e_type_name]['seed'] = []
        e_dict[e_type_name]['phase'] = []
        for i in range(0, len(files), 2):
            pha_name = files[i][:-5]
            seed_name = files[i + 1][:-4]
            if pha_name != seed_name:
                raise ValueError('请检查seed和phase是否成对存在({}, {})。'.format(files[i], files[i + 1]))
            e_dict[e_type_name]['phase'].append(os.path.join(path, files[i]))
            e_dict[e_type_name]['seed'].append(os.path.join(path, files[i + 1]))
        e_dict[e_type_name]['number'] = len(e_dict[e_type_name]['seed'])

    if not e_dict:
        raise ValueError('请检查data事件文件夹下是否载入数据。')

    print('数据已载入。')

    return e_dict


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def get_p_and_s_arrival_times(source_depth_in_km, distance):
    distance_in_degree = distance / 111

    model = TauPyModel(model="iasp91")

    try:
        arrival_p = model.get_travel_times(source_depth_in_km=source_depth_in_km,
                                           distance_in_degree=distance_in_degree,
                                           phase_list=["p"])
        travel_time_p = round_up(arrival_p[0].time, 2)
    except IndexError:
        arrival_p = model.get_travel_times(source_depth_in_km=source_depth_in_km,
                                           distance_in_degree=distance_in_degree,
                                           phase_list=["P"])
        travel_time_p = round_up(arrival_p[0].time, 2)

    try:
        arrival_s = model.get_travel_times(source_depth_in_km=source_depth_in_km,
                                           distance_in_degree=distance_in_degree,
                                           phase_list=["s"])
        travel_time_s = round(arrival_s[0].time, 2)
    except IndexError:
        arrival_s = model.get_travel_times(source_depth_in_km=source_depth_in_km,
                                           distance_in_degree=distance_in_degree,
                                           phase_list=["S"])
        travel_time_s = round(arrival_s[0].time, 2)

    return travel_time_p, travel_time_s


def sta_info(path):
    with open(path, 'r', encoding='utf-8') as file_sta:
        sta = file_sta.readlines()
        sta_dict = {}
        for sta_line in sta:
            sta_line = sta_line.strip("\n").split()
            sta_dict[sta_line[0] + sta_line[1]] = [float(sta_line[2]), float(sta_line[3])]

    return sta_dict


def get_pha(pha, stations, dis_list):
    with open(pha, 'r', encoding='utf-8') as file:
        data = file.readlines()
        t_dict = {}
        count = 0
        for data_line in data:
            data_line = data_line.strip("\n").split()
            if count == 0:
                e_lat = data_line[6]
                e_lon = data_line[8]
                e_depth = float(data_line[10])
                ori_time = data_line[0] + '-' + data_line[1] + '-' + data_line[2] + ' ' + data_line[3] + ':' + \
                           data_line[4] + ':' + data_line[5]
                ot = datetime.datetime.strptime(ori_time, "%Y-%m-%d %H:%M:%S.%f")
                ot_strf = ot.strftime("%Y-%m-%dT%H:%M:%S.%f")
            elif count >= 3:
                if data_line[0].startswith('#Station'):
                    break
                pha_sta = data_line[0].split('.')[0] + data_line[0].split('.')[1]
                if pha_sta.startswith('HK') or pha_sta == 'GDDYXC' or pha_sta.startswith('GDL4') or pha_sta.startswith(
                        'GXL4') or pha_sta == 'GDFSTN':
                    continue

                date_time = data_line[4] + ' ' + data_line[5]
                dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f")
                delta = (dt - ot).total_seconds()
                if ('Pg' in data_line[3] or 'Pb' in data_line[3] or data_line[3] == 'P') and data_line[7] == '1.0':
                    t_dict[pha_sta] = [delta]
                elif ('Sg' in data_line[3] or 'Pb' in data_line[3] or data_line[3] == 'S') and data_line[7] == '1.0':
                    if pha_sta in t_dict:
                        t_dict[pha_sta].append(delta)
                    else:
                        distance = geodistance(stations[pha_sta][1], stations[pha_sta][0], e_lon, e_lat)
                        p_delta, _ = get_p_and_s_arrival_times(e_depth, distance)
                        t_dict[pha_sta] = [p_delta, delta]
                else:
                    continue
            count += 1

    t_dict2 = {}
    for key, value in t_dict.items():
        if len(value) == 1:
            distance = geodistance(stations[key][1], stations[key][0], e_lon, e_lat)
            _, s_delta = get_p_and_s_arrival_times(e_depth, distance)
            t_dict2[key] = [value[0], round_up(s_delta, 2)]
        else:
            t_dict2[key] = value

    t_dict3 = {}
    for key, value in t_dict2.items():
        distance = geodistance(stations[key][1], stations[key][0], e_lon, e_lat)
        t3 = None
        t4 = None
        for dis in dis_list:
            if dis[0] <= distance < dis[1]:
                if dis[2] == -1:
                    t3 = value[1]
                else:
                    t3 = round_up(value[0] + dis[2], 2)
                    if t3 > value[1]:
                        t3 = value[1]
                t4 = round_up(value[1] + dis[3], 2)
                break
        if t3 is not None and t4 is not None and t3 <= value[1]:
            value.append(t3)
            value.append(t4)
            t_dict3[key] = value

    t_dict3['ot'] = ot_strf

    return t_dict3


def get_phases(phase_path, stations, cf):
    t_dicts = {}
    dis_limit = []
    for key, value in cf.items():
        dis = key.split('-')
        dis_min = int(dis[0])
        dis_max = int(dis[1])
        dis_limit.append([dis_min, dis_max, value[0], value[1]])
    for root, dirs, files in os.walk(phase_path):
        for file in files:
            if not file.endswith('phase'):
                continue
            t_dict = get_pha(os.path.join(root, file), stations, dis_limit)
            t_dicts[file] = t_dict
    return t_dicts


def get_sacs(seed_dir, out_dir, t_arr, sac_len):
    for key, values in t_arr.items():
        out_dir_name = os.path.splitext(key)[0]
        seed_name = out_dir_name + '.seed'
        seed_path = os.path.join(seed_dir, seed_name)
        if not os.path.exists(seed_path):
            continue

        ot = values['ot']
        ot = UTCDateTime(ot + '+08')
        et = ot + datetime.timedelta(seconds=sac_len)
        if not os.path.exists(os.path.join(out_dir, out_dir_name)):
            os.mkdir(os.path.join(out_dir, out_dir_name))
        stream = read(seed_path)
        for st in stream:
            head = st.stats
            need_sta = head.network + head.station
            if need_sta not in values.keys():
                continue
            sac_name = out_dir_name + '.' + head.network + '.' + head.station + '.' + head.channel + '.SAC'
            st_cut = st.trim(starttime=ot, endtime=et, fill_value=0, pad=True)
            st_cut.write(os.path.join(out_dir, out_dir_name, sac_name))
            st_sac = read(os.path.join(out_dir, out_dir_name, sac_name))
            head_sac = st_sac[0].stats.sac
            head_sac['nzyear'] = ot.year
            head_sac['nzjday'] = ot.julday
            head_sac['nzhour'] = ot.hour
            head_sac['nzmin'] = ot.minute
            head_sac['nzsec'] = ot.second
            head_sac['nzmsec'] = ot.microsecond
            head_sac['o'] = 0
            head_sac['kevnm'] = sac_name.split('.')[1]

            t1 = values[need_sta][0]
            t2 = values[need_sta][1]
            t3 = values[need_sta][2]
            t4 = values[need_sta][3]
            if t1 != 0:
                head_sac['t1'] = t1
            if t2 != 0:
                head_sac['t2'] = t2
            if t3 != 0:
                head_sac['t3'] = t3
            if t4 != 0:
                head_sac['t4'] = t4
            st_sac.write(os.path.join(out_dir, out_dir_name, sac_name))

    return

# all the imports ---------

from dask_jobqueue import SGECluster, SLURMCluster
from dask.distributed import Client
import dask
import dask.dataframe as dd
import math 
import pdb
import os
import glob

# These are for python basic
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
import math 
from scipy import fftpack
from math import sqrt
from csv import reader
import datetime as dt
from datetime import datetime
import pdb
import seaborn as sn

# these are for ML 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# dask imports
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from dask_ml.compose import ColumnTransformer
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.compose import ColumnTransformer
from dask_ml.compose import make_column_transformer
from dask_ml.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import GridSearchCV, RandomizedSearchCV


class DataGeneration():
    # cluster ON
    def start_cluster(self, jobs):
        cluster = SLURMCluster(
        header_skip=['--mem', 'another-string', '-A'],
        queue='generic',
        project="myproj",
        cores=24,
        memory='400GB',
        walltime="60:00:00",
        n_workers=120,
        )
        cluster.scale(jobs)
        client = Client(cluster)


    # ** (internal function call)**
    def time_generate(self, i, reconstructed_df, recon_row):
        #----------reding the csv files-----------------#
        name = 'New_splitted_csv_' + str(int(i)) + '.csv'
        tmp = pd.read_csv(name) 
        name_of_generated_csv = 'New_time_generated_csv_' + str(int(i)) + '.csv'
        
        #----------final compute .......................#

        for index, row in tmp.iterrows():
        
            if index == len(tmp)-1:
                break
            reconstructed_df.loc[recon_row, 'SNR'] = tmp.loc[index, 'SNR']
            reconstructed_df.loc[recon_row, 'RSRP'] = tmp.loc[index, 'RSRP']
            reconstructed_df.loc[recon_row, 'RSSI'] = tmp.loc[index, 'RSSI']
            reconstructed_df.loc[recon_row, 'NOISE POWER'] = tmp.loc[index, 'NOISE POWER']
            reconstructed_df.loc[recon_row, 'TIME(S)'] = tmp.loc[index, 'TIME(S)']
            reconstructed_df.loc[recon_row, 'TIME(FrS)'] = tmp.loc[index, 'TIME(FrS)']
            reconstructed_df.loc[recon_row, 'RX_GAIN'] = tmp.loc[index, 'RX_GAIN']
            reconstructed_df.loc[recon_row, 'Channel_Start'] = 0
            reconstructed_df.loc[recon_row, 'Channel_Length'] = 10
            reconstructed_df.loc[recon_row, 'Rx_Power'] = tmp.loc[index, 'Rx_power']
            reconstructed_df.loc[recon_row, 'MCS'] = tmp.loc[index, 'MCS']
            reconstructed_df.loc[recon_row, 'Successfully_Decoded'] = tmp.loc[index, 'Successfully_Decoded']
            reconstructed_df.loc[recon_row, 'new_time_epoch'] = float(float(tmp.loc[index, 'TIME(S)']) + float(tmp.loc[index, 'TIME(FrS)']))
            next_index = index + 1
            temp = tmp.loc[index, 'TIME(FrS)']
            next_temp = tmp.loc[next_index, 'TIME(FrS)']
            temp_sec = tmp.loc[index, 'TIME(S)']
            next_temp_sec = tmp.loc[next_index, 'TIME(S)']
            if math.isnan(temp) or math.isnan(next_temp) or math.isnan(temp_sec) or math.isnan(next_temp_sec):
                continue
            diff_sec = abs(next_temp_sec - temp_sec)

            if diff_sec > 1:
                recon_row = recon_row + 1
                continue
            else:
                if next_temp < temp:
                    next_t1 = next_temp * 1000
                    t1 = temp * 1000
                    next_t1 = int(next_t1)
                    t1 = int(t1)
                    next_t1 = next_t1 + 1000
                    diff_millisec = abs(next_t1 - t1)
                elif next_temp == temp:
                    if temp_sec!= next_temp_sec:
                        diff_millisec = 1000
                    else:
                        diff_millisec = 0
                else:
                    next_t1 = next_temp * 1000
                    t1 = temp * 1000
                    next_t1 = int(next_t1)
                    t1 = int(t1)
                    diff_millisec = abs(next_t1 - t1)
                
            
                if diff_millisec == 1:
                    recon_row = recon_row + 1
                    continue
            
                elif diff_millisec == 2 and int(next_temp*1000)%5==1:
                    recon_row = recon_row + 1
                    continue
                else:
                    temp = temp * 1000
                    temp = int(temp)
                    next_temp = next_temp * 1000
                    next_temp = int(next_temp)
                    count = 0
                    for i in range(diff_millisec):
                        flag = 0
                        tmp_millisec = temp + 1 + i
                        if tmp_millisec%5==0:
                            count = count + 1
                            continue
                        x1 = i + recon_row + 1 - count
                        if tmp_millisec >= 1000:
                            flag = 1
                            tmp_millisec = tmp_millisec % 1000
                            reconstructed_df.loc[x1, 'TIME(S)'] = next_temp_sec
                        else:
                            reconstructed_df.loc[x1, 'TIME(S)'] = temp_sec
                        reconstructed_df.loc[x1, 'TIME(FrS)'] = (float(tmp_millisec)/1000.00)
                        reconstructed_df.loc[x1, 'SNR'] = np.nan 
                        reconstructed_df.loc[x1, 'RSRP'] = np.nan 
                        reconstructed_df.loc[x1, 'RSSI'] = np.nan 
                        reconstructed_df.loc[x1, 'NOISE POWER'] = np.nan 
                        reconstructed_df.loc[x1, 'RX_GAIN'] = np.nan 
                        reconstructed_df.loc[x1, 'Channel_Start'] = 0
                        reconstructed_df.loc[x1, 'Channel_Length'] = 10
                        reconstructed_df.loc[x1, 'Rx_Power'] = np.nan 
                        tem_mcs = (tmp.loc[index, 'MCS'] + i + 1 - count)%20
                        reconstructed_df.loc[x1, 'MCS'] = tem_mcs
                        reconstructed_df.loc[x1, 'Successfully_Decoded'] = 0
                        if flag == 0:
                            reconstructed_df.loc[x1, 'new_time_epoch'] = float(temp_sec) + reconstructed_df.loc[x1, 'TIME(FrS)']
                        else:
                            reconstructed_df.loc[x1, 'new_time_epoch'] = float(next_temp_sec) + reconstructed_df.loc[x1, 'TIME(FrS)']             
                    recon_row = x1
        reconstructed_df.to_csv(name_of_generated_csv, index=False)


    # split the big txt file in to smaller chunks 
    def split_big_txt_file_into_smaller_ones(self):
        loc = "../../output_logs/s3_ue1_output_20210624_0900.txt"
        line_count = 0
        with open(loc, mode='r') as fi:
            lines = fi.readlines()


        # here 15 is the number of chunks 
        temp = [[] for i in range(15)]
        for i in range(15):
            temp[i].extend(lines[i*1000000:(i+1)*1000000])
            """Name = "file"+ str(i) + ".txt" 
            with open(Name, 'w') as f:
                f.write(temp)
            f.close()"""

            
        print(len(lines))
        print(f'Processed {line_count} lines.')

        for i in range(15):
            Name = "log_file_0900"+ str(i) + ".txt" 
            textfile = open(Name, "w")
            for element in temp[i]:
                textfile.write(element + "\n")
            textfile.close()

    
    # ** internal function call **
    def log_data(self, i):
        map_of_tti = np.full(100000,-1, dtype=int)
        column_names = ["time_epoch", "Time(S)", "Time(FrS)", "SNR", "RSRP", "RSSI", "MCS", "Successfully Decoded", "tti", "original_time"]
        combined_df_1 = pd.DataFrame(columns = column_names)  
        travarse_row = 0
        last_time_sec = 1624517194 
        last_time_millisec = 0.752000
        first_tti = 7562
        tap = 0
        cnt = 0
        is_div = 0
        cmp_div = 0
        flag = 0
        
        
        name = "log_file_0900" + str(int(i)) + ".txt"

        with open(name,"r") as fi:
            line_read = []
            for ln in fi:
                line_string = ln
                line_read = ln.split()
                if len(line_read) == 0:
                    continue
                check_1 = line_string.find("HOST-TIME UHD:")
                check_2 = line_string.find("<--")
                if check_1 == -1 and check_2 == -1:
                    continue
                if line_string.find("HOST-TIME UHD:")!=-1:
                    idx = line_read.index("UHD:")
                    last_time_sec = int(line_read[idx+1])
                    last_time_millisec = float(line_read[idx+2])
                    first_tti_idx = line_read.index("TTI:")
                    first_tti = int(line_read[first_tti_idx+1])
                    flag = 1
                    continue


                if line_string.find("DECODED")!=-1:
                    idx_tti = line_read.index("tti")
                    tti = int(line_read[idx_tti+1])
                    idx_mcs = line_read.index("mcs:")
                    mcs = int(line_read[idx_mcs+1])
                    map_of_tti[tti] = mcs
                    continue

                if line_string.find("TTI")!= -1:
                    idx_TTI = line_read.index("TTI")
                    tti = int(line_read[idx_TTI+1])

                    if map_of_tti[tti]!=-1:
                        mcs = map_of_tti[tti]
                        map_of_tti[tti] = -1

                # snr 
                        snr_idx = line_read.index("SNR:")
                        if snr_idx == -1:
                            snr = 0.0
                        else:
                            snr = float(line_read[snr_idx+1])
                # time setting

                        time_epoch = float(float(last_time_sec) + float(last_time_millisec))


                # RSRP
                        rsrp_idx = line_read.index("RSRP:")
                        if rsrp_idx == -1:
                            rsrp = 0.0
                        else:
                            rsrp = float(line_read[rsrp_idx+1])                    
                        
                # RSSI
                        rssi_idx = line_read.index("RSSI:")
                        if rssi_idx == -1:
                            rssi = 0.0
                        else:
                            rssi_list = line_read[rssi_idx+1]
                            rssi_list = rssi_list.split("(")
                            rssi_list = rssi_list[0]
                            rssi = float(rssi_list)

                # Successfully Decoded
                        decoded_idx = line_read.index("Decoded:")
                        if decoded_idx == -1:
                            decoded_or_not = 0
                        else:
                            decoded_or_not = int(line_read[decoded_idx+1])

                # putting values in the pandas dataframe 
                        if tti == first_tti and flag ==1:
                            combined_df_1.loc[travarse_row, 'original_time'] = 1
                            flag = 0
                        elif tti != first_tti and flag ==1:
                            tmp = abs(tti - first_tti)
                            tmp = float(tmp)
                            tmp = float(tmp)/1000.00
                            tmp = tmp + last_time_millisec
                            if tmp >= 1.00:
                                last_time_sec = last_time_sec + 1
                                tmp = tmp - 1.00
                                last_time_millisec = tmp

                            last_time_millisec = tmp
                            combined_df_1.loc[travarse_row, 'original_time'] = 1
                            flag = 0
                        else:
                            combined_df_1.loc[travarse_row, 'original_time'] = 0

                        combined_df_1.loc[travarse_row, 'time_epoch'] = time_epoch
                        combined_df_1.loc[travarse_row, 'Time(S)'] = last_time_sec
                        combined_df_1.loc[travarse_row, 'Time(FrS)'] = last_time_millisec
                        combined_df_1.loc[travarse_row, 'SNR'] = snr
                        combined_df_1.loc[travarse_row, 'RSRP'] = rsrp
                        combined_df_1.loc[travarse_row, 'RSSI'] = rssi
                        combined_df_1.loc[travarse_row, 'MCS'] = mcs
                        combined_df_1.loc[travarse_row, 'Successfully Decoded'] = decoded_or_not
                        combined_df_1.loc[travarse_row, 'tti'] = tti
                        print('done with ...', travarse_row, combined_df_1.loc[travarse_row, 'Time(S)'])
                        travarse_row = travarse_row + 1            

                    else:
                        continue

        combined_df_1['tti'] = combined_df_1['tti'].astype(int)
        combined_df_1['Time(S)'] = combined_df_1['Time(S)'].astype(int)
        combined_df_1['Time(FrS)'] = combined_df_1['Time(FrS)'].astype(float)
        combined_df_1['original_time'] = combined_df_1['original_time'].astype(int) 
        print('finished with ... ', i, 'file')
        name2 = "gen_time_0900_" + str(int(i)) + ".csv"
        combined_df_1.to_csv(name2, index=False)



    # this functions extracts data from the log file, 
    def extract_data_from_log_file(self):
        results = []
        data_frame = pd.DataFrame(L, columns=list(["index"]))
        for i,id in data_frame.iterrows():
            y = dask.delayed(self.log_data)(id)
            results.append(y)
        results = dask.compute(*results)  


    # extract info from the pcap, 
    def time_generation_from_PCAP(self):
        self.start_claster()
        data_frame = pd.DataFrame(L, columns=list(["index"]))
        results = []
        for i,id in data_frame.iterrows():
            column_names = ["time_epoch", "SNR", "RSRP", "RSSI", "NOISE POWER", "TIME(S)", "TIME(FrS)", "RX_GAIN", "Channel_Start", "Channel_Length", "Rx_Power", "MCS", "Successfully_Decoded", "new_time_epoch"]
            reconstructed_df = pd.DataFrame(columns = column_names)
            reconstructed_df['time_epoch'] = reconstructed_df['time_epoch'].astype(float)
            reconstructed_df['SNR'] = reconstructed_df['SNR'].astype(float)
            reconstructed_df['RSRP'] = reconstructed_df['RSRP'].astype(float)
            reconstructed_df['RSSI'] = reconstructed_df['RSSI'].astype(float)
            reconstructed_df['NOISE POWER'] = reconstructed_df['NOISE POWER'].astype(float)
            reconstructed_df['TIME(S)'] = reconstructed_df['TIME(S)'].astype(int)
            reconstructed_df['TIME(FrS)'] = reconstructed_df['TIME(FrS)'].astype(float)
            reconstructed_df['RX_GAIN'] = reconstructed_df['RX_GAIN'].astype(float)
            reconstructed_df['Channel_Start'] = reconstructed_df['Channel_Start'].astype(float)
            reconstructed_df['Channel_Length'] = reconstructed_df['Channel_Length'].astype(float)
            reconstructed_df['Rx_Power'] = reconstructed_df['Rx_Power'].astype(float)
            reconstructed_df['MCS'] = reconstructed_df['MCS'].astype(float)
            reconstructed_df['Successfully_Decoded'] = reconstructed_df['Successfully_Decoded'].astype(int)
            reconstructed_df['new_time_epoch'] = reconstructed_df['new_time_epoch'].astype(float)

            y = dask.delayed(self.time_generate)(id, reconstructed_df, recon_row=0)
            results.append(y)
        results = dask.compute(*results)
        


    # direct to the folder then run this, 
    def combine_all_csv_files_in_a_folder(self):
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
        combined_csv.to_csv( "interpolated_final_time.csv", index=False)


    # GPS_info contains all the information about the GPS(6 rounds combined)
    def divide_gps_data_into_6_rounds_and_save_as_csv(self):
        splitin = pd.read_csv('GPS_info.csv')
        round_1 = splitin[1311:4234]
        round_2 = splitin[4608:7646]
        round_3 = splitin[8888:13262]
        round_4 = splitin[16124:19937]
        round_5 = splitin[20276:23421]
        round_6 = splitin[24259:26701]
        round_1.to_csv('GPS_info_round_1.csv', index=False)
        round_2.to_csv('GPS_info_round_2.csv', index=False)
        round_3.to_csv('GPS_info_round_3.csv', index=False)
        round_4.to_csv('GPS_info_round_4.csv', index=False)
        round_5.to_csv('GPS_info_round_5.csv', index=False)
        round_6.to_csv('GPS_info_round_6.csv', index=False)


    # gps merge code **(internal function call)**
    def merge_with_GPS_info(self, id):
        input_file_name = 'interpolated_final_time.csv'
        gps_file_name = 'GPS_info_round_' + str(int(id)) + '.csv'
        df_ue1_s3 = pd.read_csv(input_file_name)
        merged_gps_user1_user2 = pd.read_csv(gps_file_name)
        df_ue1_s3['new_time_epoch'] = df_ue1_s3['new_time_epoch'].astype(float)
        df_ue1_s3 = df_ue1_s3.sort_values(by='new_time_epoch', ascending=True)
        df_ue1_s3['wireshark_time'] = df_ue1_s3['new_time_epoch'].round(0)
        s3_merged_gps_user1_user2 = pd.merge(left=merged_gps_user1_user2, right=df_ue1_s3, left_on='wireshark_time', right_on='wireshark_time')
        s3_merged_gps_user1_user2['distance'] = 1000 * s3_merged_gps_user1_user2['distance']
        output_file_name = 'merge_with_gps_info_round_' + str(int(id)) + '.csv'
        s3_merged_gps_user1_user2.to_csv(output_file_name, index=False)

        
    # merge with the GPS info 
    def merge_with_gps_data_and_divide_data_into_six_rounds(self):
        for i in range(6):
            self.merge_with_GPS_info(i+1)

    # Unix time for different rounds , start and end 
    def get_unix_time_for_all_rounds(self):
        r1_s = (dt.datetime(2021, 6, 24, 8, 59, 38) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r1_e = (dt.datetime(2021, 6, 24, 9, 48, 21) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r2_s = (dt.datetime(2021, 6, 24, 9, 54, 35) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r2_e = (dt.datetime(2021, 6, 24, 10, 45, 13) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r3_s = (dt.datetime(2021, 6, 24, 11, 6, 30) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r3_e = (dt.datetime(2021, 6, 24, 12, 18, 49) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r4_s = (dt.datetime(2021, 6, 24, 15, 39, 9) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r4_e = (dt.datetime(2021, 6, 24, 16, 42, 42) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r5_s = (dt.datetime(2021, 6, 24, 16, 48, 21) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r5_e = (dt.datetime(2021, 6, 24, 17, 43, 49) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r6_s = (dt.datetime(2021, 6, 24, 18, 12, 19) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        r6_e = (dt.datetime(2021, 6, 24, 19, 0, 44) - dt.datetime(1970, 1, 1, 0, 0, 0)).total_seconds() - 7200
        print('round 1 ',r1_s ,r1_e , 'round 2 ', r2_s, r2_e,'round 3 ', r3_s,r3_e, 'round 4 ', r4_s, r4_e,' round 5 ',r5_s, r5_e , ' round 6 ',r6_s, r6_e)

    def interpolation_for_regenerated_packets(self):
        df_0 = pd.read_csv('merge_with_gps_info_round_0.csv')
        df_1 = pd.read_csv('merge_with_gps_info_round_1.csv')
        df_2 = pd.read_csv('merge_with_gps_info_round_2.csv')
        df_3 = pd.read_csv('merge_with_gps_info_round_3.csv')
        df_4 = pd.read_csv('merge_with_gps_info_round_4.csv')
        df_5 = pd.read_csv('merge_with_gps_info_round_5.csv')

        df_0 = df_0.interpolate(method='nearest', limit_direction='forward', axis=0)
        df_1 = df_1.interpolate(method='nearest', limit_direction='forward', axis=0)
        df_2 = df_2.interpolate(method='nearest', limit_direction='forward', axis=0)
        df_3 = df_3.interpolate(method='nearest', limit_direction='forward', axis=0)
        df_4 = df_4.interpolate(method='nearest', limit_direction='forward', axis=0)
        df_5 = df_5.interpolate(method='nearest', limit_direction='forward', axis=0)

        df_0 = df_0.interpolate(method='nearest', limit_direction='backward', axis=0)
        df_1 = df_1.interpolate(method='nearest', limit_direction='backward', axis=0)
        df_2 = df_2.interpolate(method='nearest', limit_direction='backward', axis=0)
        df_3 = df_3.interpolate(method='nearest', limit_direction='backward', axis=0)
        df_4 = df_4.interpolate(method='nearest', limit_direction='backward', axis=0)
        df_5 = df_5.interpolate(method='nearest', limit_direction='backward', axis=0)

        df_0 = df_0.interpolate(method='pad', limit=2, axis=0)
        df_1 = df_1.interpolate(method='pad', limit=2, axis=0)
        df_2 = df_2.interpolate(method='pad', limit=2, axis=0)
        df_3 = df_3.interpolate(method='pad', limit=2, axis=0)
        df_4 = df_4.interpolate(method='pad', limit=2, axis=0)
        df_5 = df_5.interpolate(method='pad', limit=2, axis=0)

        df_0.to_csv('merge_with_gps_info_round_0.csv', index=False)
        df_1.to_csv('merge_with_gps_info_round_1.csv', index=False)
        df_2.to_csv('merge_with_gps_info_round_2.csv', index=False)
        df_3.to_csv('merge_with_gps_info_round_3.csv', index=False)
        df_4.to_csv('merge_with_gps_info_round_4.csv', index=False)
        df_5.to_csv('merge_with_gps_info_round_5.csv', index=False)


    # ** internal function call **
    def cal_max_mcs(self, id):
        import datetime
        from itertools import chain
        name = 'merge_with_gps_info_round_' + str(int(id)) + '.csv'
        df_final = pd.read_csv(name) 
        df_all = df_final
        df_ini = df_all
        df_ini = df_ini
        
        series = np.where(df_final["MCS"] == 0)
        series = list(series)
        zero_indx = np.where(df_ini["MCS"] == 0)
        zero_indx = list(zero_indx)
        zero_indx = list(chain.from_iterable(zero_indx))
        
        data = {
        'ts_gps': [],  
        'new_time_epoch': [],   
        'latitude_user1':[],
        'longitude_user1':[],
        'speed_user1':[],
        'latitude_user3':[],
        'longitude_user3':[],
        'speed_user3':[],
        'distance': [],
        'SNR': [],
        'RSRP': [],
        'RSSI': [],
        'NOISE POWER': [],
        'RX_GAIN': [],
        'Rx_Power': [],
        'MCS': []}
        
        df_marks = pd.DataFrame(data)
        row = 0
        for i in range(len(zero_indx)):
            
            nxt = zero_indx[i]
            df_tmp = df_ini.iloc[row:nxt,:]
            success_decode_indx = np.where(df_tmp["Successfully_Decoded"] == 1)
            success_decode_indx = list(success_decode_indx)
            success_decode_indx = list(chain.from_iterable(success_decode_indx))
            sec = df_tmp.iloc[success_decode_indx,:]
            if len(sec) == 0:
                max_mcs = -1
                df_marks.loc[i,'latitude_user1'] = df_tmp['latitude_user1'].mean()
                df_marks.loc[i,'longitude_user1'] = df_tmp['longitude_user1'].mean()
                df_marks.loc[i,'speed_user1'] = df_tmp['speed_user1'].mean()
                df_marks.loc[i,'latitude_user3'] = df_tmp['latitude_user3'].mean()
                df_marks.loc[i,'longitude_user3'] = df_tmp['longitude_user3'].mean()
                df_marks.loc[i,'speed_user3'] = df_tmp['speed_user3'].mean()
                df_marks.loc[i,'distance'] = df_tmp['distance'].mean()
                df_marks.loc[i,'SNR'] = df_tmp['SNR'].mean()
                df_marks.loc[i,'RSRP'] = df_tmp['RSRP'].mean()
                df_marks.loc[i,'RSSI'] = df_tmp['RSSI'].mean()
                df_marks.loc[i,'NOISE POWER'] = df_tmp['NOISE POWER'].mean()
                df_marks.loc[i,'RX_GAIN'] = df_tmp['RX_GAIN'].mean()
                df_marks.loc[i,'Rx_Power'] = df_tmp['Rx_Power'].mean()   
                df_marks.loc[i,'new_time_epoch'] = df_ini.loc[row,'new_time_epoch']
                df_marks.loc[i,'ts_gps'] = df_ini.loc[row,'ts_gps']
                df_marks.loc[i,'MCS'] = -1
            else:
                max_mcs = sec["MCS"].max()
                df_marks.loc[i,'latitude_user1'] = sec['latitude_user1'].mean()
                df_marks.loc[i,'longitude_user1'] = sec['longitude_user1'].mean()
                df_marks.loc[i,'speed_user1'] = sec['speed_user1'].mean()
                df_marks.loc[i,'latitude_user3'] = sec['latitude_user3'].mean()
                df_marks.loc[i,'longitude_user3'] = sec['longitude_user3'].mean()
                df_marks.loc[i,'speed_user3'] = sec['speed_user3'].mean()
                df_marks.loc[i,'distance'] = sec['distance'].mean()
                df_marks.loc[i,'SNR'] = sec['SNR'].mean()
                df_marks.loc[i,'RSRP'] = sec['RSRP'].mean()
                df_marks.loc[i,'RSSI'] = sec['RSSI'].mean()
                df_marks.loc[i,'NOISE POWER'] = sec['NOISE POWER'].mean()
                df_marks.loc[i,'RX_GAIN'] = sec['RX_GAIN'].mean()
                df_marks.loc[i,'Rx_Power'] = sec['Rx_Power'].mean()   
                df_marks.loc[i,'new_time_epoch'] = df_ini.loc[row,'new_time_epoch'].mean()
                tim = df_marks.loc[i,'new_time_epoch']
                tim = int(tim)
                d = datetime.datetime.fromtimestamp(tim)
                df_marks.loc[i,'ts_gps'] = d
                df_marks.loc[i,'MCS'] = max_mcs
            row = nxt
        name2 = 'dataset_for_maximum_mcs_prediction_' + str(int(id)) + '.csv'
        df_marks.to_csv(name2, index=False)

    # generate data for maximum MCS prediction 
    def generate_data_for_maximum_MCS_prediction(self):
        self.start_claster()
        data_frame = pd.DataFrame(L, columns=list(["index"]))
        results = []
        for i,id in data_frame.iterrows():
            y = dask.delayed(self.cal_max_mcs)(id+1)
            results.append(y)
        results = dask.compute(*results)  

















        





        

    



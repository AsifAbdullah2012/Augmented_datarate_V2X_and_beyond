# from algorithms.gradientboostingregression import GradientBoostingRegression
from importlib.resources import path
from data.data_loader import S3Measurements
from sklearn.model_selection import train_test_split
# from data.data_generation import DataGeneration
from algorithms.random_forest_regression import RandomForestQuantileRegressor, RandomForest, HyperparameterSearch
from algorithms.gradientboostingregression import HyperparameterSearch_Gradient_Boosting_Regression, GradientBoostingRegression, GradientBoostingQuantileRegressor
from algorithms.neuralnetwork import NeuralNetwork, NeuralNetworkRegressor
from algorithms.save_or_load_models_CQR import Save_or_Load_model_plus_CQR
import time
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
import os
from sklearn.linear_model import LinearRegression
import numpy as np



if __name__ == '__main__':
    # model = GradientBoostingRegression()
    # model.dummy_method()

    # data_generation_object = DataGeneration()
    # data_generation_object.time_generation_from_PCAP()
    # physical_devices = tf.config.list_physical_devices('CPU')
    # print('HW in the new server: ', physical_devices)
    # grad_obj = GradientBoostingRegression()
    # grad_obj.start_local_cluster()
    data_loader_obj = S3Measurements()
    # model_random_forest = RandomForest()
    # print(model_random_forest.optimal_params())
    # data = data_loader_obj.load_measurements_ml_data()
    data = data_loader_obj.load_measurements_ml_data()
    df = pd.concat([data[0], data[1]], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    test_size_fixed = 167135
    X_test = df[:test_size_fixed]
    y_test = X_test['MCS']
    X_test = X_test.drop(['MCS'], axis=1)
    # index = [data[1], data[2], data[3], data[4], data[5]]
    # df = pd.concat(index)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df = df[0:1500]
    # df.to_csv('temp.csv', index=False)
    # df = pd.read_csv('temp.csv')
    #df = df[:, 1:]

    # run and save the values
    # df_for_random = pd.DataFrame()
    # lis_idx = list()
    # lis_train_siz = list()
    # lis_test_siz = list()
    # lis_ideal_data_rate = list()
    # lis_predicted_data_rate = list()
    # lis_empiricl_quantile = list()
    # for i in range(2000):
    #     if (60000 + 380*(i+1)) > 608000:
    #         break
    #     X_train = df[60000:60000 + 380*(i+1)]
    #     y_train = X_train['MCS'] 
    #     X_train = X_train.drop(['MCS'], axis=1)
    

    # print('here .. features ..', len(data[0]), ' and target ...', len(data[1]))
    # data[0] = data[0][0:500000]
    # data[1] = data[1][0:500000]
    # X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.33, random_state=42)
    # print('train ', len(X_train), ' and target ', len(y_train))
    # param_to_optimize = model_random_forest.params_to_optimize()
    # print('## hyper parameters to optimize ..', param_to_optimize)
    # HyperparameterSearch_obj = HyperparameterSearch()
    # # data_loader_obj.start_cluster()
    # time_start = time.time()
    
    
    # best_param = HyperparameterSearch_obj.gethyperparameter_result(param_to_optimize, X_train, y_train)

    # with open('hyperparameters_for_random_forest.txt', 'w') as f:
    #     save_text = str(best_param)
    #     f.write(save_text)
    # time_end = time.time()
    # print('here ... best parameters are ...', best_param)
    # print('Hype para Opt. for random forest ', (time_end - time_start))


    # RandomForestQuantileRegressor_obj = RandomForestQuantileRegressor()
    # X_train = df[200000:]
    # y_train = X_train['MCS']
    # X_train = X_train.drop(['MCS'], axis=1) 

    # /home/FE/rokoni/rokoni/clean_code/algorithms/saved_models/Random_forest
    # ------XXXX----------taining time for random forest ----------XXXX------
    # 33% test, 77%(train(60%) + calibrated(40%))
    # test: 165000, calibrated: 134000, train: rest are test 

    # time_start = time.time()
    # with joblib.parallel_backend('dask'):
    #     RandomForestQuantileRegressor_obj.fit(X_train, y_train)
    # # saving model for cqr 
    # cqr_obj = Save_or_Load_model_plus_CQR()
    # cqr_obj.save_model(RandomForestQuantileRegressor_obj)
    # time_end = time.time()
    # print('training time .... ', (time_end - time_start))
    
    
    # -------------ends here -----------------------------------------------


    # # ------------prediction time random forest ----------------------------
    # time_start = time.time()
    # with joblib.parallel_backend('dask'):
    #     pred_res = RandomForestQuantileRegressor_obj.predict(X_test, 20)
    # time_end = time.time()
    # print('prediction time ....', (time_end - time_start))
    # # ----------------ends -------------------------------------------------


    # # -------data rate calculation -----------------------------------------
    # print(' the prediction result is ... ', pred_res, ' and type is ', type(pred_res))
    # # convert the y_test to numpy array
    # y_test = y_test.to_numpy()
    # curr_res = data_loader_obj.get_datarate_measurements(pred_res, y_test)
    # over_predicted_percentage = data_loader_obj.over_predicted_mcs(pred_res, y_test)
    
    # print('ideal data rate: ', curr_res[0], ' predicted data rate: ', curr_res[1], ' over predicted percentage ', over_predicted_percentage)

        # lis_idx.append(i)
        # lis_train_siz.append(len(X_train))
        # lis_test_siz.append(len(X_test))
        # lis_ideal_data_rate.append(curr_res[0])
        # lis_predicted_data_rate.append(curr_res[1])
        # lis_empiricl_quantile.append(over_predicted_percentage)

    # df_for_random['index'] = lis_idx
    # df_for_random['trian_size'] = lis_train_siz
    # df_for_random['test_size'] = lis_test_siz
    # df_for_random['ideal'] = lis_ideal_data_rate
    # df_for_random['predicted'] = lis_predicted_data_rate
    # df_for_random['em_quantile'] = lis_empiricl_quantile

    # df_for_random.to_csv('data_rate_vs_sample_size_Random_forest_q_20.csv')

    # -----XXXX------ends -here --------------------------XXXX-------------------------







    # ----XXXX------------GradientBoostingFit --------XXXX-------------

    # df_for_gradient_boosting = pd.DataFrame()
    # lis_idx = list()
    # lis_train_siz = list()
    # lis_test_siz = list()
    # lis_ideal_data_rate = list()
    # lis_predicted_data_rate = list()
    # lis_empiricl_quantile = list()
    # for i in range(2000):
    #     if (60000 + 380*(i+1)) > 608000:
    #         break
    #     X_train = df[60000:60000 + 380*(i+1)]
    #     y_train = X_train['MCS'] 
    #     X_train = X_train.drop(['MCS'], axis=1)

    #     optimal_parameters_for_gradient_boosting = grad_obj.optimal_params()
    #     gbr = GradientBoostingQuantileRegressor()
    #     time_start = time.time()
        
    #     with joblib.parallel_backend('dask'):
    #         gbr.fit(X_train, y_train)
    #     time_end = time.time()
    #     print('time took for fitting ', (time_end - time_start))


    #     time_start = time.time()
    #     with joblib.parallel_backend('dask'):
    #         gbr_pred = gbr.predict(X_test)
    #     time_end = time.time()
    #     print('time took for the predicting ', (time_end - time_start))
        
    #     print(' here .. pred type', type(gbr_pred), ' test type ', type(y_test))
    #     # y_test = y_test.to_numpy()
    #     curr_res = data_loader_obj.get_datarate_measurements(gbr_pred, y_test)
    #     over_predicted_percentage = data_loader_obj.over_predicted_mcs(gbr_pred, y_test)
    #     print(' here the over predicted mcs is ', over_predicted_percentage)
    #     print('ideal data rate is ', curr_res[0], ' and predicted data rate is ', curr_res[1])
    #     lis_idx.append(i)
    #     lis_train_siz.append(len(X_train))
    #     lis_test_siz.append(len(X_test))
    #     lis_ideal_data_rate.append(curr_res[0])
    #     lis_predicted_data_rate.append(curr_res[1])
    #     lis_empiricl_quantile.append(over_predicted_percentage)

    # df_for_gradient_boosting['index'] = lis_idx
    # df_for_gradient_boosting['trian_size'] = lis_train_siz
    # df_for_gradient_boosting['test_size'] = lis_test_siz
    # df_for_gradient_boosting['ideal'] = lis_ideal_data_rate
    # df_for_gradient_boosting['predicted'] = lis_predicted_data_rate
    # df_for_gradient_boosting['em_quantile'] = lis_empiricl_quantile
    # df_for_gradient_boosting.to_csv('gradientboostingregression_q_20.csv')  
    #-----XXXX------ends here ---------------------XXXXX-----------------------


    #---XXXX Neural Network ---------XXXXXX---------------------------


    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    # neural_net_obj = NeuralNetwork()
    # param_for_nn = neural_net_obj.optimal_params()
    # tau = .28
    # unit = param_for_nn['units_per_layer']
    # activation = param_for_nn['activation']
    # lr = param_for_nn['learning_rate']
    # layer = param_for_nn['hidden_layers']
    # epoch = param_for_nn['epochs']
    # nn_reg = NeuralNetworkRegressor()
    # scaler = StandardScaler()
    # train_dataset = scaler.fit_transform(X_train)
    # test_dataset = scaler.transform(X_test)
    # print('epoch: ', epoch, ' tau: ', tau, 'unit: ', unit, 'lr: ', lr, 'layer: ', layer)
    # with tf.device('CPU:0'):
    #     fited_model = nn_reg.nn_model_fit(epoch, tau, unit, activation, lr, layer, train_dataset, test_dataset, y_train, y_test)
    # print('model summary: ', fited_model.summary())
    # with tf.device('CPU:0'):
    #     nn_pred = nn_reg.predict_data(fited_model, test_dataset)
    # siz = len(nn_pred)
    # for i in range(siz):
    #     if nn_pred[i] > 19 or nn_pred[i] < 0:
    #         if nn_pred[i] > 19:
    #             nn_pred[i] = 19
    #         if nn_pred[i] < 0:
    #             nn_pred[i] = 0
    # y_test = y_test.to_numpy()
    # curr_res = data_loader_obj.get_datarate_measurements(nn_pred, y_test)
    # over_predicted_percentage = data_loader_obj.over_predicted_mcs(nn_pred, y_test)
    # neural_net_obj._cleanup_memory()
    # print(' here the over predicted mcs is ', over_predicted_percentage)
    # print('ideal data rate is ', curr_res[0], ' and predicted data rate is ', curr_res[1])

    # ideal = curr_res[0]
    # predicted = curr_res[1]
    # # str1 = "Over Predicted MCS " + str(over_predicted_percentage) + "\n"
    # # str2 = "ideal data rate is " + str(ideal) + "\n"
    # # str3 = "predicted data rate " + str(predicted) + + "\n"
    # d = {'over predicted': [over_predicted_percentage], 'Ideal': [ideal], 'predicted': [predicted]}
    # df_save = pd.DataFrame(data = d)
    # df_save.to_csv('tempval.csv')
    # # path_to_the_file = "output_files/NN_all_data_" + str(arra_job) + ".txt"
    # # with open(path_to_the_file, 'w') as f:
    # #     f.write(str1)
    # #     f.write(str2)
    # #     f.write(str3)

    #----------XXXX Neural Net ends here ----------------XXXXX------------


    #-------XXXX CQR for neural network-------XXXXXX-----------------------


    # cqr_obj = Save_or_Load_model_plus_CQR()
    # # cqr_obj.save_model(fited_model)
    # loaded_model = cqr_obj.load_model('algorithms/saved_models/NN_.28_saved_model.json', 'algorithms/saved_models/NN_.28_saved_model_weights.h5')
    # path_lo_model = 'algorithms/saved_models/NN_.28_saved_model.json'
    # path_lo_model_data = 'algorithms/saved_models/NN_.28_saved_model_weights.h5'
    # path_hi_model = 'algorithms/saved_models/NN_.72_saved_model.json'
    # path_hi_model_data = 'algorithms/saved_models/NN_.72_saved_model_weights.h5'
    # scaler = StandardScaler()
    # train_dataset = scaler.fit_transform(X_train)
    # test_dataset = scaler.transform(X_test)
    # # change the quantile here for the function call 
    # lis_cqr = cqr_obj.CQR(test_dataset, y_test, 28, path_lo_model, path_hi_model, path_lo_model_data, path_hi_model_data)
    # # 0 for the lo, and 1 for the hi, 2 for test_x, 3 for calibrated_x, 4 for test_y, 5 for calibrated_y
    # lis_cqr[4] = lis_cqr[4].to_numpy()
    # print('lo ', type(lis_cqr[0]), ' siz ', len(lis_cqr[0]), ' hi ', type(lis_cqr[1]), ' len ', len(lis_cqr[1]), ' test y ', type(lis_cqr[4]), ' siz ', len(lis_cqr[4]))
    # print('So the total parcentage that is inside is ... ', cqr_obj.In_boundary(lis_cqr[4], lis_cqr[0], lis_cqr[1]))
    # # curr_res = data_loader_obj.get_datarate_measurements(lis_cqr[0], lis_cqr[4])
    # empirical_quantile = cqr_obj.empirical_quantile(lis_cqr[4], lis_cqr[0])
    # print(' the empirical quantile is wiht CQR: ', empirical_quantile)
    # nn_pred = nn_reg.predict_data(loaded_model, lis_cqr[2])
    # # nn_pred = nn_pred.to_numpy()
    # empirical_quantile = cqr_obj.empirical_quantile(lis_cqr[4], nn_pred)
    # print(' the empirical quantile is without CQR: ', empirical_quantile)

    #----XXXX CQR ends here ---------------------XXXXXXXXXXXXXXXX------------------------

    #----- linear regression starts here XXXXXXXXXXXXXXXX-----------------------------
    # df_for_random = pd.DataFrame()
    # lis_idx = list()
    # lis_train_siz = list()
    # lis_test_siz = list()
    # lis_ideal_data_rate = list()
    # lis_predicted_data_rate = list()
    # lis_empiricl_quantile = list()
    # for i in range(2000):
    #     if (60000 + 380*(i+1)) > 608000:
    #         break
    #     X_train = df[60000:60000 + 380*(i+1)]
    #     y_train = X_train['MCS'] 
    #     X_train = X_train.drop(['MCS'], axis=1)

    #     linear_reg = LinearRegression()
    #     with joblib.parallel_backend('dask'):
    #         linear_reg.fit(X_train, y_train)
    #     with joblib.parallel_backend('dask'):
    #         pred_res = linear_reg.predict(X_test)
        
    #     print(' the prediction result is ... ', len(pred_res), ' and type is ', type(pred_res), ' and test ', len(y_test))
    #     # convert the y_test to numpy array
    #     # y_test = y_test.to_numpy()
    #     for lop_pred in range(len(pred_res)):
    #         if pred_res[lop_pred]<0:
    #             pred_res[lop_pred] = 1
    #         if pred_res[lop_pred]>19:
    #             pred_res[lop_pred] = 19
        

    #     curr_res = data_loader_obj.get_datarate_measurements(pred_res, y_test)
    #     over_predicted_percentage = data_loader_obj.over_predicted_mcs(pred_res, y_test)
        
    #     print('ideal data rate: ', curr_res[0], ' predicted data rate: ', curr_res[1], ' over predicted percentage ', over_predicted_percentage)

    #     lis_idx.append(i)
    #     lis_train_siz.append(len(X_train))
    #     lis_test_siz.append(len(X_test))
    #     lis_ideal_data_rate.append(curr_res[0])
    #     lis_predicted_data_rate.append(curr_res[1])
    #     lis_empiricl_quantile.append(over_predicted_percentage)

    # df_for_random['index'] = lis_idx
    # df_for_random['trian_size'] = lis_train_siz
    # df_for_random['test_size'] = lis_test_siz
    # df_for_random['ideal'] = lis_ideal_data_rate
    # df_for_random['predicted'] = lis_predicted_data_rate
    # df_for_random['em_quantile'] = lis_empiricl_quantile

    # df_for_random.to_csv('linear_regression_as_base_line.csv')  


    #--------XXXXXXX ----linear regression ends here ------XXXXXXXXXX---------




    #--------MULTIPLE RUN random forest -----------------XXXXXXXXXX------------

    # loop_data = [5000, 10000, 20000, 50000, 100000, 200000, 500000]
    # for i in loop_data:
    #     if (60000 + i) > 608000:
    #         break
    #     df_for_random_forest = pd.DataFrame()
    #     lis_idx = list()
    #     lis_train_siz = list()
    #     lis_test_siz = list()
    #     lis_ideal_data_rate = list()
    #     lis_predicted_data_rate = list()
    #     lis_empiricl_quantile = list()

    #     for j in range(80):       

    #         X_train = df[60000:60000 + i]
    #         X_train = X_train.sample(frac=1).reset_index(drop=True)
    #         y_train = X_train['MCS'] 
    #         X_train = X_train.drop(['MCS'], axis=1)

    #         RandomForestQuantileRegressor_obj = RandomForestQuantileRegressor()
            
    #         time_start = time.time()
            
            
    #         with joblib.parallel_backend('dask'):
    #             RandomForestQuantileRegressor_obj.fit(X_train, y_train)
    #         time_end = time.time()
    #         print('training time .... ', (time_end - time_start))


    #         time_start = time.time()
    #         with joblib.parallel_backend('dask'):
    #             pred_res = RandomForestQuantileRegressor_obj.predict(X_test, 20)
    #         time_end = time.time()
    #         print('prediction time ....', (time_end - time_start))
            
    #         print(' here .. pred type', type(gbr_pred), ' test type ', type(y_test))
    #         # y_test = y_test.to_numpy()
    #         curr_res = data_loader_obj.get_datarate_measurements(pred_res, y_test)
    #         over_predicted_percentage = data_loader_obj.over_predicted_mcs(pred_res, y_test)
    #         print(' here the over predicted mcs is ', over_predicted_percentage)
    #         print('ideal data rate is ', curr_res[0], ' and predicted data rate is ', curr_res[1])
    #         lis_idx.append(i)
    #         lis_train_siz.append(len(X_train))
    #         lis_test_siz.append(len(X_test))
    #         lis_ideal_data_rate.append(curr_res[0])
    #         lis_predicted_data_rate.append(curr_res[1])
    #         lis_empiricl_quantile.append(over_predicted_percentage)

    #     df_for_random_forest['index'] = lis_idx
    #     df_for_random_forest['trian_size'] = lis_train_siz
    #     df_for_random_forest['test_size'] = lis_test_siz
    #     df_for_random_forest['ideal'] = lis_ideal_data_rate
    #     df_for_random_forest['predicted'] = lis_predicted_data_rate
    #     df_for_random_forest['em_quantile'] = lis_empiricl_quantile
    #     name = 'multiple_run_on_samedata_point' + str(i) + '_data_rate_vs_sample_size_random_forest_quantile_q_20.csv'
    #     df_for_random_forest.to_csv(name) 



#---------XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---CQR------Random forest----------------------------------------------------------------------------
    # RandomForestQuantileRegressor_obj = RandomForestQuantileRegressor()
    # X_train = df[200000:]
    # y_train = X_train['MCS']
    # X_train = X_train.drop(['MCS'], axis=1) 



    # time_start = time.time()
    # with joblib.parallel_backend('dask'):
    #     RandomForestQuantileRegressor_obj.fit(X_train, y_train)
    # saving model for cqr 
    # cqr_obj = Save_or_Load_model_plus_CQR()
    # cqr_obj.save_model(RandomForestQuantileRegressor_obj)
    # time_end = time.time()
    # print('training time .... ', (time_end - time_start))

    #------CQR----------------------------------------------------
    # test_x, calibrated_x, test_y, calibrated_y = train_test_split(X_test, y_test, test_size=0.4)
    # pred_lo = RandomForestQuantileRegressor_obj.predict(calibrated_x, 20)
    # pred_hi = RandomForestQuantileRegressor_obj.predict(calibrated_x, 80)
    # calibrated_y = calibrated_y.to_numpy()

    # pred_lo = pred_lo.flatten()
    # pred_hi = pred_hi.flatten()
    # print(' # pred_lo ', pred_lo)
    # print(' # pred_hi ', pred_hi)
    # print(' # calibrated y ', calibrated_y)
    # errors = np.maximum(pred_lo - calibrated_y, calibrated_y - pred_hi)
    # quantile = 20
    # significance = (((quantile) * 2) / 100.00)
    # correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))
    # print('correction is: ', correction)
    # test_pred_lo = RandomForestQuantileRegressor_obj.predict(test_x, 20)
    # test_pred_hi = RandomForestQuantileRegressor_obj.predict(test_x, 80)
    # test_pred_lo = test_pred_lo - correction
    # test_pred_hi = test_pred_hi + correction


    # ------------prediction time random forest ----------------------------
    # time_start = time.time()
    # with joblib.parallel_backend('dask'):
    #     pred_res = RandomForestQuantileRegressor_obj.predict(test_x, 20)
    # time_end = time.time()
    # print('prediction time ....', (time_end - time_start))
    # ----------------ends -------------------------------------------------


    # -------data rate calculation -----------------------------------------
    # print(' the prediction result is ... ', pred_res, ' and type is ', type(pred_res))
    # test_y = test_y.to_numpy()
    # this is for the w/o cqr
    # datarate_wo_cqr = data_loader_obj.get_datarate_measurements(pred_res, test_y)
    # over_predicted_percentage_wo_cqr = data_loader_obj.over_predicted_mcs(pred_res, test_y)
    
    # this is for w cqr
    # datarate_w_cqr = data_loader_obj.get_datarate_measurements(test_pred_lo, test_y)
    # over_predicted_percentage_w_cqr = data_loader_obj.over_predicted_mcs(test_pred_lo, test_y) 


    # print('ideal data rate: ', curr_res[0], ' predicted data rate: ', curr_res[1], ' over predicted percentage ', over_predicted_percentage)
    # print('WO_CQR_ideal data rate: ', datarate_wo_cqr[0], ' predicted data rate: ', datarate_wo_cqr[1], ' over predicted percentage ', over_predicted_percentage_wo_cqr)
    # print('W_CQR_ideal data rate: ', datarate_w_cqr[0], ' predicted data rate: ', datarate_w_cqr[1], ' over predicted percentage ', over_predicted_percentage_w_cqr)

    #-----XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------------------------------------------------



    #--XXXXXXXXXXXXXXX-CQR using the gradient boosting -----XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------------

    # X_train = df[200000:]
    # y_train = X_train['MCS']
    # X_train = X_train.drop(['MCS'], axis=1) 

    # optimal_parameters_for_gradient_boosting = grad_obj.optimal_params()
    ## change the optimal paramtr for hi quantile 
    # print('optimal parameter: ', optimal_parameters_for_gradient_boosting)
    ## ** change the quantile here .....
    # gbr_lo = GradientBoostingQuantileRegressor(298, "squared_error", 17, 9, 15, 0., 0.7805291762864555, None, 0., None, 0, False, .20, 'quantile', None)
    # gbr_hi = GradientBoostingQuantileRegressor(298, "squared_error", 17, 9, 15, 0., 0.7805291762864555, None, 0., None, 0, False, .80, 'quantile', None)
    # gbr = GradientBoostingQuantileRegressor()
    # cqr starts here ....
    # test_x, calibrated_x, test_y, calibrated_y = train_test_split(X_test, y_test, test_size=0.4)
    # with joblib.parallel_backend('dask'):
    #     gbr_lo.fit(X_train, y_train)
    #     gbr_hi.fit(X_train, y_train)

    # pred_lo = gbr_lo.predict(calibrated_x)
    # pred_hi = gbr_hi.predict(calibrated_x)
    # calibrated_y = calibrated_y.to_numpy()

    # pred_lo = pred_lo.flatten()
    # pred_hi = pred_hi.flatten()
    # print(' # pred_lo ', pred_lo)
    # print(' # pred_hi ', pred_hi)
    # print(' # calibrated y ', calibrated_y)
    # errors = np.maximum(pred_lo - calibrated_y, calibrated_y - pred_hi)
    ## ** change the quantile here .....
    # quantile = 20
    # significance = (((quantile) * 2) / 100.00)
    # correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))
    # print('correction is: ', correction)
    # test_pred_lo = gbr_lo.predict(test_x)
    # test_pred_lo_wo_cqr = test_pred_lo
    # test_pred_hi = gbr_hi.predict(test_x)
    # gbr_lo.predict(test_x)
    # test_pred_lo = test_pred_lo - correction
    # test_pred_hi = test_pred_hi + correction


    # -------data rate calculation -----------------------------------------
    # print(' the prediction result is ... ', pred_lo, ' and type is ', type(pred_lo))
    # test_y = test_y.to_numpy()
    # this is for the w/o cqr
    # datarate_wo_cqr = data_loader_obj.get_datarate_measurements(test_pred_lo_wo_cqr, test_y)
    # over_predicted_percentage_wo_cqr = data_loader_obj.over_predicted_mcs(test_pred_lo_wo_cqr, test_y)
    
    # this is for w cqr
    # datarate_w_cqr = data_loader_obj.get_datarate_measurements(test_pred_lo, test_y)
    # over_predicted_percentage_w_cqr = data_loader_obj.over_predicted_mcs(test_pred_lo, test_y) 


    # print('ideal data rate: ', curr_res[0], ' predicted data rate: ', curr_res[1], ' over predicted percentage ', over_predicted_percentage)
    # print('WO_CQR_ideal data rate: ', datarate_wo_cqr[0], ' predicted data rate: ', datarate_wo_cqr[1], ' over predicted percentage ', over_predicted_percentage_wo_cqr)
    # print('W_CQR_ideal data rate: ', datarate_w_cqr[0], ' predicted data rate: ', datarate_w_cqr[1], ' over predicted percentage ', over_predicted_percentage_w_cqr)


    #-----XXXXXXXXXXXXX end of Gradient Boosting ------XXXXXXXXXXXXXXXXXXX


    #-----XXXXXXX multiple run on neural network ------XXXXXXXXXXXXXXx
    # cqr config: so, 25% test data fixed, (from train data 40% calibration and 60%train) will be fine , right ?

    # len_of_train_plus_calibration = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    # len_of_train_plus_calibration = [1000]
    # for i in len_of_train_plus_calibration:
    #     if (test_size_fixed + i) > 608000:
    #         break

    df_for_neural_network = pd.DataFrame()
    lis_idx = list()
    lis_train_siz = list()
    lis_calibration_siz = list()
    lis_test_siz = list()
    lis_ideal_data_rate = list()
    lis_predicted_data_rate = list()
    lis_empiricl_quantile = list()
    lis_predicted_data_rate_corrected_lo = list()
    lis_predicted_data_rate_corrected_hi = list()
    lis_empirical_quantile_after_correction_lo = list()
    lis_empirical_quantile_after_correction_hi = list()

    for j in range(80):
        X_train = df[test_size_fixed:]
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        y_train = X_train['MCS'] 
        X_train = X_train.drop(['MCS'], axis=1)
        scaler = StandardScaler()
        train_dataset = scaler.fit_transform(X_train)
        test_dataset = scaler.transform(X_test)
        X_train, calibrated_x, y_train, calibrated_y = train_test_split(train_dataset, y_train, test_size=0.4)

        # start NN
        neural_net_obj = NeuralNetwork()
        param_for_nn = neural_net_obj.optimal_params()
        tau_lo = .20
        tau_hi = .80
        unit = param_for_nn['units_per_layer']
        activation = param_for_nn['activation']
        lr = param_for_nn['learning_rate']
        layer = param_for_nn['hidden_layers']
        epoch = param_for_nn['epochs']
        nn_reg = NeuralNetworkRegressor()
        print('epoch: ', epoch, ' tau_lo: ', tau_lo, 'tau_hi: ', tau_hi, 'unit: ', unit, 'lr: ', lr, 'layer: ', layer)

        # cqr 
        with tf.device('GPU:0'):
            NN_low = nn_reg.nn_model_fit(epoch, tau_lo, unit, activation, lr, layer, X_train, test_dataset, y_train, y_test)
            NN_hi = nn_reg.nn_model_fit(epoch, tau_hi, unit, activation, lr, layer, X_train, test_dataset, y_train, y_test)
            pred_lo = nn_reg.predict_data(NN_low, calibrated_x)
            pred_hi = nn_reg.predict_data(NN_hi, calibrated_x)
            calibrated_y = calibrated_y.to_numpy()
            pred_lo = pred_lo.flatten()
            pred_hi = pred_hi.flatten()
            print(' # pred_lo ', pred_lo)
            print(' # pred_hi ', pred_hi)
            print(' # calibrated y ', calibrated_y)
            errors = np.maximum(pred_lo - calibrated_y, calibrated_y - pred_hi)
            quantile = 20
            significance = (((quantile) * 2) / 100.00)
            correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))

            test_pred_lo = nn_reg.predict_data(NN_low, test_dataset)
            test_pred_lo_wo_cqr = test_pred_lo
            test_pred_hi = nn_reg.predict_data(NN_hi, test_dataset)
            test_pred_lo = test_pred_lo - correction
            test_pred_hi = test_pred_hi + correction
            # y_test = y_test.to_numpy()

                # this is for the w/o cqr
            datarate_wo_cqr = data_loader_obj.get_datarate_measurements(test_pred_lo_wo_cqr, y_test)
            over_predicted_percentage_wo_cqr = data_loader_obj.over_predicted_mcs(test_pred_lo_wo_cqr, y_test)

                # this is for w cqr low
            datarate_w_cqr_lo = data_loader_obj.get_datarate_measurements(test_pred_lo, y_test)
            over_predicted_percentage_w_cqr_lo = data_loader_obj.over_predicted_mcs(test_pred_lo, y_test) 

                # this is for w cqr high 
            datarate_w_cqr_hi = data_loader_obj.get_datarate_measurements(test_pred_hi, y_test)
            over_predicted_percentage_w_cqr_hi = data_loader_obj.over_predicted_mcs(test_pred_hi, y_test) 

        lis_idx.append(len(X_train))
        lis_train_siz.append(len(X_train))
        lis_calibration_siz.append(len(calibrated_x))
        lis_test_siz.append(len(test_dataset))
        lis_ideal_data_rate.append(datarate_wo_cqr[0])
        lis_predicted_data_rate.append(datarate_wo_cqr[1])
        lis_empiricl_quantile.append(over_predicted_percentage_wo_cqr)
        lis_predicted_data_rate_corrected_lo.append(datarate_w_cqr_lo[1])
        lis_predicted_data_rate_corrected_hi.append(datarate_w_cqr_hi[1])
        lis_empirical_quantile_after_correction_lo.append(over_predicted_percentage_w_cqr_lo)
        lis_empirical_quantile_after_correction_hi.append(over_predicted_percentage_w_cqr_hi)
        print('XXX 1 Run fininshed -------------------------------------------------------------> workingg !!!!')

    df_for_neural_network['index'] = lis_idx
    df_for_neural_network['train_size'] = lis_train_siz
    df_for_neural_network['calibration_size'] = lis_calibration_siz
    df_for_neural_network['test_size'] = lis_test_siz
    df_for_neural_network['ideal_data_rate'] = lis_ideal_data_rate
    df_for_neural_network['predicted_data_rate'] = lis_predicted_data_rate
    df_for_neural_network['empirical_quantile_w_o_cqr'] = lis_empiricl_quantile
    df_for_neural_network['predicted_data_rate_corrected_lo'] = lis_predicted_data_rate_corrected_lo
    df_for_neural_network['lis_predicted_data_rate_corrected_hi'] = lis_predicted_data_rate_corrected_hi
    df_for_neural_network['lis_empirical_quantile_after_correction_lo'] = lis_empirical_quantile_after_correction_lo
    df_for_neural_network['lis_empirical_quantile_after_correction_hi'] = lis_empirical_quantile_after_correction_hi
    name = 'ALL_multiple_run_on_samedata_point' + '_all' + '_NN_q_.20.csv'
    df_for_neural_network.to_csv(name,index=False)



    
    # with tf.device('CPU:0'):
    #     fited_model = nn_reg.nn_model_fit(epoch, tau, unit, activation, lr, layer, train_dataset, test_dataset, y_train, y_test)
    # print('model summary: ', fited_model.summary())
    # with tf.device('CPU:0'):
    #     nn_pred = nn_reg.predict_data(fited_model, test_dataset)
    # siz = len(nn_pred)
    # for i in range(siz):
    #     if nn_pred[i] > 19 or nn_pred[i] < 0:
    #         if nn_pred[i] > 19:
    #             nn_pred[i] = 19
    #         if nn_pred[i] < 0:
    #             nn_pred[i] = 0
    # y_test = y_test.to_numpy()
    # curr_res = data_loader_obj.get_datarate_measurements(nn_pred, y_test)
    # over_predicted_percentage = data_loader_obj.over_predicted_mcs(nn_pred, y_test)
    # neural_net_obj._cleanup_memory()
    # print(' here the over predicted mcs is ', over_predicted_percentage)
    # print('ideal data rate is ', curr_res[0], ' and predicted data rate is ', curr_res[1])

    # ideal = curr_res[0]
    # predicted = curr_res[1]
    # # str1 = "Over Predicted MCS " + str(over_predicted_percentage) + "\n"
    # # str2 = "ideal data rate is " + str(ideal) + "\n"
    # # str3 = "predicted data rate " + str(predicted) + + "\n"
    # d = {'over predicted': [over_predicted_percentage], 'Ideal': [ideal], 'predicted': [predicted]}
    # df_save = pd.DataFrame(data = d)
    # df_save.to_csv('tempval.csv')
    # # path_to_the_file = "output_files/NN_all_data_" + str(arra_job) + ".txt"
    # # with open(path_to_the_file, 'w') as f:
    # #     f.write(str1)
    # #     f.write(str2)
    # #     f.write(str3)    






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pointbiserialr

from DataPreparation.data_preparation import DataPrep
from MLModels.isolation_forrest import IsolationForrest

data_path = os.path.join(os.getcwd(), 'Dataset\\Anomaly\\MIXED\\MIXED\\mixed1')


# Load the first CSV
df_platform_sample= pd.read_csv(os.path.join(data_path ,"platform.csv"))

# Load the second CSV
#df_radio_sample = pd.read_csv(os.path.join(data_path, "radio.csv"))

def plot_histogram_from_list(data, title, x_label, y_label, bins):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_simple_scatter(df, x, y, title):
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.show()

def plot_simple(df, x, y, title):
    
    plt.figure(figsize=(10, 5))
    plt.plot(df[x], df[y], "g-")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

# dumgr_du_1_DUMGR_LOGGER_1_max	dumgr_du_1_dumgr_du_1_max	gnb_cu_l3_l3_main_max	gnb_cu_pdcp_0_0_f1_worker_0_max	gnb_cu_pdcp_0_0_pdcp_master_0_max	gnb_cu_pdcp_0_0_pdcp_worker_0_max	gnb_cu_pdcp_0_0_recv_data_0_max	gnb_du_layer2_LowPrio_DU1_C0_max	gnb_du_layer2_TxCtrl_DU1_C0_max	gnb_du_layer2_f1_du_worker_0_max	gnb_du_layer2_pr_accumulator_max	gnb_du_layer2_rlcAccum_DU1_max	gnb_du_layer2_rlcTimer_DU1_max	gnb_du_layer2_rlcWorkrDU1__max	l1app_main_ebbupool_act_0_max	l1app_main_ebbupool_act_1_max	l1app_main_ebbupool_act_2_max	l1app_main_ebbupool_act_3_max	l1app_main_fh_main_poll-22_max	l1app_main_fh_rx_bbdev-21_max	phc2sys_phc2sys_max	ptp4l_ptp4l_max	dumgr_du_1_DUMGR_LOGGER_1_total_events	dumgr_du_1_dumgr_du_1_total_events	gnb_cu_l3_l3_main_total_events

def get_correlation(df, col1, col2):

    return df[col1].corr(df[col2])

def save_csv(filename, list_of_correlations):
    
    df = pd.DataFrame(list_of_correlations, columns=["Column 1", "Column 2", "Correlation Coefficient"])
    df.to_csv(filename, index=False)


def correlation_test(column_names, df_platform_sample_cut, df_platform_sample, data_prep):
    correlated_columns = []
    uncorrelated_columns = []

    for i in range(len(column_names)):
        for j in range(i+1, len(column_names)):
            correlation_coefficient = get_correlation(df_platform_sample_cut, column_names[i], column_names[j])
            print(f"{column_names[i]} - {column_names[j]}: {correlation_coefficient}")
            if abs(correlation_coefficient) > 0.95:
                new_df_sample_cut = data_prep.split_data_by_time(200,300 , df_platform_sample)
                new_correlation_coefficient = get_correlation(new_df_sample_cut, column_names[i], column_names[j])
                print(f"New correlation coefficient: {new_correlation_coefficient}")
                if abs(new_correlation_coefficient) > 0.95:
                    correlated_columns.append([column_names[i], column_names[j], correlation_coefficient])
            elif abs(correlation_coefficient) < 0.1:
                uncorrelated_columns.append([column_names[i], column_names[j], correlation_coefficient])
    
    return correlated_columns, uncorrelated_columns

def correlation_test_anomaly(column_names, df_platform_sample, data_prep):
    correlated_columns = []
    uncorrelated_columns = []
    correlation_coefficients = []

    for i in range(len(column_names)):
        correlation_coefficient = get_correlation(df_platform_sample, column_names[i], "anomaly")
        print(f"{column_names[i]} - anomaly: {correlation_coefficient}")
        if abs(correlation_coefficient) >= 0.1:
            correlated_columns.append([column_names[i],"anomaly" ,correlation_coefficient])
        elif abs(correlation_coefficient) < 0.1:
            uncorrelated_columns.append([column_names[i], "anomaly" ,correlation_coefficient])
        correlation_coefficients.append(correlation_coefficient)

    return correlated_columns, uncorrelated_columns, correlation_coefficients

def correlation_test_anomaly_biserial(column_names, df_platform_sample, data_prep):
    correlated_columns = []
    uncorrelated_columns = []
    correlation_coefficients = []

    for i in range(len(column_names)):
        correlation_coefficient = pointbiserialr(y = df_platform_sample[column_names[i]], x = df_platform_sample["anomaly"])
        print(f"{column_names[i]} - anomaly: {correlation_coefficient}")
        if abs(correlation_coefficient.correlation) > 0.80:
            correlated_columns.append([column_names[i],"anomaly" ,correlation_coefficient])
        elif abs(correlation_coefficient.correlation) < 0.1:
            uncorrelated_columns.append([column_names[i], "anomaly" ,correlation_coefficient])
        correlation_coefficients.append(correlation_coefficient.correlation)

    return correlated_columns, uncorrelated_columns, correlation_coefficients

if __name__ == '__main__':
    
    data_prep = DataPrep()
    iso_forrest = IsolationForrest()

    df_platform_sample = data_prep.normalize_time_and_convert_to_seconds(df_platform_sample, "timestamp")
    print(df_platform_sample)
 
    df_platform_sample_cut = data_prep.split_data_by_time(0,100 , df_platform_sample)
    df_platform_sample_cut = data_prep.fill_empty_rows(df_platform_sample_cut)
    
    column_names = df_platform_sample.columns[2:-1]
    
    #correlated_columns, uncorrelated_columns = correlation_test(column_names, df_platform_sample_cut, df_platform_sample, data_prep)
    correlated_columns, uncorrelated_columns, correlation_coefficients = correlation_test_anomaly(column_names, df_platform_sample, data_prep)
    input_columns = [item[0] for item in correlated_columns]
    print("Input columns: ", input_columns)
    iso_forrest.fit_model(df_platform_sample, input_columns)
    df_platform_sample = iso_forrest.predict(df_platform_sample, column_names)
    print(len(df_platform_sample[df_platform_sample["anomaly_predicted"] == df_platform_sample["anomaly"]]))
    print(len(df_platform_sample[df_platform_sample["anomaly_predicted"] != df_platform_sample["anomaly"]]))

    #correlated_columns_biserial, uncorrelated_columns_biserial, correlation_coefficients_biserial = correlation_test_anomaly_biserial(column_names, df_platform_sample, data_prep)
    #save_csv("correlated_columns_with_anomaly.csv", correlated_columns)
    #save_csv("uncorrelated_columns_with_anomaly.csv", uncorrelated_columns)
    
    #plot_histogram_from_list(data=correlation_coefficients, title="Correlation Coefficients", x_label="Correlation Coefficient", y_label="Frequency", bins=100)
    #plot_histogram_from_list(data=correlation_coefficients_biserial, title="Correlation Coefficients Biserial", x_label="Correlation Coefficient", y_label="Frequency", bins=100)



    """
    print(uncorrelated_columns)

    for column in uncorrelated_columns:
        plot_simple(df_platform_sample_cut, x="timestamp", y=column[0], title="Platform Sample")
        plot_simple(df_platform_sample_cut, x="timestamp", y=column[1], title="Platform Sample")
        plot_simple_scatter(df_platform_sample_cut, x=column[0], y=column[1], title="Platform Sample")
    """
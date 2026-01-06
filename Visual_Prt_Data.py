import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from log_code import setup_logging
logger = setup_logging('Visual_Prt_Data')


class ChurnVisualization:

    def __init__(self, file_path, save_dir="plots"):
        try:
            self.df = pd.read_csv(file_path)
            self.save_dir = save_dir

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            logger.info("Dataset Info ===================================")
            logger.info(f"Dataset Shape: {self.df.shape}")
            logger.info(f"Dataset Columns: {self.df.columns.tolist()}")
            logger.info(f"Dataset Dtypes:\n{self.df.dtypes}")
            logger.info("Dataset loaded successfully")

        except Exception:
            etype, emsg, eline = sys.exc_info()
            logger.critical(f"ERROR in Line {eline.tb_lineno} : {emsg}")


    def add_synthetic_columns(self):
        try:
            np.random.seed(42)

            self.df['Region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(self.df))
            self.df['Device_Type'] = np.random.choice(['Android', 'iOS', 'Feature Phone'],size=len(self.df), p=[0.65, 0.25, 0.10])
            self.df['Network_Type'] = np.random.choice(['2G', '3G', '4G', '5G'],size=len(self.df), p=[0.10, 0.20, 0.50, 0.20])
            self.df['SIM_Provider'] = np.random.choice(['Airtel', 'Jio', 'VI', 'BSNL'],size=len(self.df), p=[0.35, 0.40, 0.15, 0.10])
            self.df['Usage_Pattern'] = np.random.choice(['Low', 'Medium', 'High'],size=len(self.df), p=[0.30, 0.45, 0.25])

            logger.info("Synthetic columns added successfully")

        except Exception:
            etype, emsg, eline = sys.exc_info()
            logger.error(f"ERROR in Line {eline.tb_lineno} : {emsg}")


    def _save_plot(self, filename):
        try:
            plt.tight_layout()
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path)
            plt.show()

        except Exception:
            etype, emsg, eline = sys.exc_info()
            logger.error(f"ERROR in Line {eline.tb_lineno} : {emsg}")


    def churn_distribution(self):
        plt.figure(figsize=(5,3))
        self.df['Churn'].value_counts().plot(kind='bar')
        plt.title("Churn Distribution")
        plt.xlabel("Churn")
        plt.ylabel("Count")
        self._save_plot("Churn_Distribution.png")


    def tenure_vs_churn(self):
        plt.figure(figsize=(5,3))
        self.df.boxplot(column='tenure', by='Churn')
        plt.title("Tenure vs Churn")
        plt.suptitle("")
        self._save_plot("Tenure_vs_Churn.png")


    def monthly_charges_vs_churn(self):
        plt.figure(figsize=(5,3))
        self.df.boxplot(column='MonthlyCharges', by='Churn')
        plt.title("Monthly Charges vs Churn")
        plt.suptitle("")
        self._save_plot("MonthlyCharges_vs_Churn.png")


    def contract_distribution(self):
        plt.figure(figsize=(5,3))
        self.df['Contract'].value_counts().plot(kind='bar')
        plt.title("Contract Type Distribution")
        self._save_plot("Contract_Distribution.png")


    def internet_service_distribution(self):
        plt.figure(figsize=(5,3))
        self.df['InternetService'].value_counts().plot(kind='bar')
        plt.title("Internet Service Distribution")
        self._save_plot("InternetService_Distribution.png")


    def churn_by_contract(self):
        pd.crosstab(self.df['Contract'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Contract")
        self._save_plot("Churn_by_Contract.png")


    def churn_by_region(self):
        pd.crosstab(self.df['Region'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Region (Synthetic)")
        self._save_plot("Churn_by_Region.png")


    def churn_by_device_type(self):
        pd.crosstab(self.df['Device_Type'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Device Type (Synthetic)")
        self._save_plot("Churn_by_Device_Type.png")


    def churn_by_network_type(self):
        pd.crosstab(self.df['Network_Type'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Network Type (Synthetic)")
        self._save_plot("Churn_by_Network_Type.png")


    def churn_by_sim_provider(self):
        pd.crosstab(self.df['SIM_Provider'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by SIM Provider (Synthetic)")
        self._save_plot("Churn_by_SIM_Provider.png")


    def churn_by_usage_pattern(self):
        pd.crosstab(self.df['Usage_Pattern'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Usage Pattern (Synthetic)")
        self._save_plot("Churn_by_Usage_Pattern.png")


    def churn_by_senior_citizen(self):
        pd.crosstab(self.df['SeniorCitizen'], self.df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn by Senior Citizen")
        self._save_plot("Churn_by_Senior_Citizen.png")


    def churn_by_senior_gender(self):
        senior_df = self.df[self.df['SeniorCitizen'] == 1]
        pd.crosstab(senior_df['gender'], senior_df['Churn']).plot(kind='bar', figsize=(5,3))
        plt.title("Churn among Senior Citizens by Gender")
        self._save_plot("Churn_SeniorCitizen_by_Gender.png")


    def correlation_heatmap(self):
        num_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr = num_df.corr()

        plt.figure(figsize=(6,5))
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        self._save_plot("Heatmap.png")



if __name__ == "__main__":
    try:
        obj = ChurnVisualization(
            'D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\archive (2)\\WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\Visuals')

        obj.add_synthetic_columns()
        obj.churn_distribution()
        obj.tenure_vs_churn()
        obj.monthly_charges_vs_churn()
        obj.contract_distribution()
        obj.internet_service_distribution()
        obj.churn_by_contract()
        obj.churn_by_region()
        obj.churn_by_device_type()
        obj.churn_by_network_type()
        obj.churn_by_sim_provider()
        obj.churn_by_usage_pattern()
        obj.churn_by_senior_citizen()
        obj.churn_by_senior_gender()
        obj.correlation_heatmap()

    except Exception:
        etype, emsg, eline = sys.exc_info()
        logger.error(f"ERROR in Line {eline.tb_lineno} : {emsg}")

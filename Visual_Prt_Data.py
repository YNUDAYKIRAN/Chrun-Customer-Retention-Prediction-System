import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from log_code import setup_logging
logger = setup_logging('Visual_Prt_Data')

class ChurnVisualization:

    def __init__(self, file_path, save_dir):
        try:
            self.df = pd.read_csv(file_path)
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

            logger.info("===== BEFORE DATASET INFO =====")
            logger.info(f"Shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            logger.info(f"Dtypes:\n{self.df.dtypes}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- SAVE ----------
    def _save(self, name):
        try:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, name))
            plt.show()
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- CROSS TAB PERCENT ----------
    def _ct_pct(self, col, title, fname, df=None):
        try:
            d = self.df if df is None else df
            ct = pd.crosstab(d[col], d['Churn'], normalize='index') * 100
            ax = ct.plot(kind='bar', figsize=(6,4))
            plt.title(title)
            plt.ylabel("Percentage")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

            self._save(fname)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- SYNTHETIC FEATURE GENERATION ----------
    def add_synthetic_columns(self):
        try:
            np.random.seed(42)

            self.df['Region'] = np.random.choice(['Urban','Rural'], len(self.df))
            self.df['Device_Type'] = np.random.choice(['Android','iOS','Feature Phone'], len(self.df),
                                                      p=[0.65,0.25,0.10])
            self.df['Network_Type'] = np.random.choice(['2G','3G','4G','5G'], len(self.df),
                                                       p=[0.10,0.20,0.50,0.20])
            self.df['SIM_Provider'] = np.random.choice(['Airtel','Jio','VI','BSNL'], len(self.df),
                                                       p=[0.35,0.40,0.15,0.10])
            self.df['Usage_Pattern'] = np.random.choice(['Low','Medium','High'], len(self.df),
                                                        p=[0.30,0.45,0.25])

            logger.info("===== AFTER SYNTHETIC INFO =====")
            logger.info(f"Shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            logger.info(f"Dtypes:\n{self.df.dtypes}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- BASIC CHURN VISUALS ----------
    def churn_distribution(self):
        try:
            ct = self.df['Churn'].value_counts(normalize=True) * 100
            ax = ct.plot(kind='bar', figsize=(6,4))
            plt.title("Churn Distribution (%)")
            plt.ylabel("Percentage")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

            self._save("Churn_Distribution.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def churn_by_gender(self):
        self._ct_pct('gender', "Churn by Gender (%)", "Churn_by_Gender.png")

    def churn_by_senior(self):
        self._ct_pct('SeniorCitizen', "Churn by Senior Citizen (%)", "Churn_by_SeniorCitizen.png")

    def churn_senior_by_gender(self):
        try:
            s = self.df[self.df['SeniorCitizen']==1]
            self._ct_pct('gender', "Churn among Seniors by Gender (%)", "Churn_Senior_by_Gender.png", s)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- SERVICE CATEGORICAL ----------
    def churn_by_contract(self):
        self._ct_pct('Contract', "Churn by Contract Type (%)", "Churn_by_Contract.png")

    def churn_by_internet(self):
        self._ct_pct('InternetService', "Churn by Internet Service (%)", "Churn_by_InternetService.png")

    def churn_by_phone(self):
        self._ct_pct('PhoneService', "Churn by Phone Service (%)", "Churn_by_PhoneService.png")

    def churn_by_multiplelines(self):
        self._ct_pct('MultipleLines', "Churn by Multiple Lines (%)", "Churn_by_MultipleLines.png")

    def churn_by_paymentmethod(self):
        self._ct_pct('PaymentMethod', "Churn by Payment Method (%)", "Churn_by_PaymentMethod.png")

    def churn_by_paperless(self):
        self._ct_pct('PaperlessBilling', "Churn by Paperless Billing (%)", "Churn_by_PaperlessBilling.png")

    def churn_by_service_addons(self):
        try:
            cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
            for c in cols:
                self._ct_pct(c, f"Churn by {c} (%)", f"Churn_by_{c}.png")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- GROUPING ----------
    def churn_by_tenure_group(self):
        try:
            self.df['Tenure_Group'] = pd.cut(self.df['tenure'], [0,12,24,48,72,100],
                                             labels=['0-12','13-24','25-48','49-72','73+'])
            self._ct_pct('Tenure_Group', "Churn by Tenure Group (%)", "Churn_by_TenureGroup.png")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def churn_by_charges_group(self):
        try:
            self.df['Charges_Group'] = pd.qcut(self.df['MonthlyCharges'], 3, labels=['Low','Medium','High'])
            self._ct_pct('Charges_Group', "Churn by Monthly Charges Group (%)", "Churn_by_ChargesGroup.png")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- SIM COMBINATIONS ----------
    def churn_by_sim(self):
        self._ct_pct('SIM_Provider', "Churn by SIM Provider (%)", "Churn_by_SIMProvider.png")

    def churn_by_gender_sim(self):
        try:
            ct = pd.crosstab([self.df['gender'], self.df['SIM_Provider']], self.df['Churn'], normalize='index') * 100
            ax = ct.plot(kind='bar', figsize=(8,4))
            plt.title("Churn by Gender & SIM Provider (%)")
            plt.ylabel("Percentage")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

            self._save("Churn_by_Gender_SIMProvider.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def churn_by_senior_sim(self):
        try:
            seniors = self.df[self.df['SeniorCitizen']==1]
            self._ct_pct('SIM_Provider', "Churn by SIM Provider (Senior Citizens) (%)", "Churn_by_Senior_SIMProvider.png", seniors)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    def churn_by_sim_gender_senior(self):
        try:
            df = self.df.copy()
            df['SIM_Gender_Senior'] = df['SIM_Provider'] + '-' + df['gender'] + '-' + df['SeniorCitizen'].map({0:'NonSenior',1:'Senior'})
            ct = pd.crosstab(df['SIM_Gender_Senior'], df['Churn'], normalize='index') * 100
            ax = ct.plot(kind='bar', figsize=(10,5))
            plt.title("Churn by SIM, Gender & Senior (%)")
            plt.ylabel("Percentage")
            plt.xlabel("SIM - Gender - Senior")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

            plt.xticks(rotation=45, ha='right')
            self._save("Churn_by_SIM_Gender_Senior.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- QUARTERLY TENURE ----------
    def quarterly_tenure_analysis(self):
        try:
            df = self.df.copy()
            df['Quarter'] = ((df['tenure'] - 1)//3) + 1
            ct = pd.crosstab(df['Quarter'], df['Churn'], normalize='index') * 100
            ax = ct.plot(kind='bar', figsize=(8,4))
            plt.title("Churn by Quarterly Tenure (%)")
            plt.ylabel("Percentage")
            plt.xlabel("Quarter")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

            self._save("Churn_by_QuarterlyTenure.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- REGION + GENDER + SENIOR ----------
    def region_gender_senior(self):
        try:
            non_sen = self.df[self.df['SeniorCitizen']==0]
            sen = self.df[self.df['SeniorCitizen']==1]

            grp_non = non_sen.groupby(['Region','gender']).size().unstack(fill_value=0)
            grp_sen = sen.groupby(['Region','gender']).size().unstack(fill_value=0)

            regions = grp_non.index
            x = np.arange(len(regions))
            width = 0.35

            fig, axes = plt.subplots(1,2,figsize=(12,5), sharey=True)

            axes[0].bar(x-width/2, grp_non['Male'], width, label='Male')
            axes[0].bar(x+width/2, grp_non['Female'], width, label='Female')
            axes[0].set_title("Region vs Gender (Non-Senior)")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(regions)

            axes[1].bar(x-width/2, grp_sen['Male'], width, label='Male')
            axes[1].bar(x+width/2, grp_sen['Female'], width, label='Female')
            axes[1].set_title("Region vs Gender (Senior Citizens)")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(regions)

            plt.suptitle("Region vs Gender with Senior Citizen")
            plt.tight_layout()
            self._save("Region_Gender_Senior.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- AVERAGE MONTHLY CHARGES ----------
    def avg_monthly_charges_by_sim(self):
        try:
            avg = self.df.groupby(['SIM_Provider','Churn'])['MonthlyCharges'].mean().unstack()
            x = np.arange(len(avg.index))
            width = 0.35

            no_churn = avg.get('Not Churned', 0)
            yes_churn = avg.get('Churned', 0)

            plt.figure(figsize=(10,5))
            bars1 = plt.bar(x - width/2, no_churn, width, label='Not Churned')
            bars2 = plt.bar(x + width/2, yes_churn, width, label='Churned')

            for i in range(len(x)):
                total = no_churn[i] + yes_churn[i]
                if total > 0:
                    pct_no = (no_churn[i] / total) * 100
                    pct_yes = (yes_churn[i] / total) * 100
                    plt.text(x[i]-width/2, no_churn[i], f"{pct_no:.1f}%", ha='center', va='bottom')
                    plt.text(x[i]+width/2, yes_churn[i], f"{pct_yes:.1f}%", ha='center', va='bottom')

            plt.xticks(x, avg.index)
            plt.xlabel("SIM Provider")
            plt.ylabel("Average Monthly Charges")
            plt.title("Avg Monthly Charges by SIM Provider & Churn (%)")
            plt.legend()
            plt.tight_layout()
            self._save("Avg_MonthlyCharges_by_SIM.png")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

    # ---------- DEVICE ----------
    def churn_by_device(self):
        self._ct_pct('Device_Type', "Churn by Device Type (%)", "Churn_by_Device.png")

    # ---------- NETWORK ----------
    def churn_by_network(self):
        self._ct_pct('Network_Type', "Churn by Network Type (%)", "Churn_by_Network.png")

    # ---------- USAGE ----------
    def churn_by_usage(self):
        self._ct_pct('Usage_Pattern', "Churn by Usage Pattern (%)", "Churn_by_UsagePattern.png")

    # ---------- CORRELATION ----------
    def correlation(self):
        try:
            corr = self.df.select_dtypes(include=['int64','float64']).corr()
            plt.figure(figsize=(6,5))
            im = plt.imshow(corr, cmap='coolwarm')
            plt.colorbar(im)
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title("Correlation Heatmap")
            self._save("Correlation_Heatmap.png")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")


# ---------------- RUN ----------------
if __name__ == "__main__":

    obj = ChurnVisualization('D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\archive (2)\\WA_Fn-UseC_-Telco-Customer-Churn.csv',
                             'D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\Visuals')

    obj.add_synthetic_columns()
    obj.churn_distribution()
    obj.churn_by_gender()
    obj.churn_by_senior()
    obj.churn_senior_by_gender()
    obj.churn_by_contract()
    obj.churn_by_internet()
    obj.churn_by_phone()
    obj.churn_by_multiplelines()
    obj.churn_by_paymentmethod()
    obj.churn_by_paperless()
    obj.churn_by_service_addons()
    obj.churn_by_tenure_group()
    obj.churn_by_charges_group()
    obj.churn_by_sim()
    obj.churn_by_gender_sim()
    obj.churn_by_senior_sim()
    obj.churn_by_sim_gender_senior()
    obj.quarterly_tenure_analysis()
    obj.avg_monthly_charges_by_sim()
    obj.churn_by_device()
    obj.churn_by_network()
    obj.churn_by_usage()
    obj.region_gender_senior()
    obj.correlation()

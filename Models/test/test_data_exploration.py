import pandas as pd
import numpy as np
import yaml
from Models.utils.helpers import MongoExtractor
from dotenv import load_dotenv
import os
import scipy.stats as stats

load_dotenv()

config_file = 'Models/configs/config_v1_2.yaml'

# Create a MongoExtractor instance
mongo_extractor = MongoExtractor(
        connection_string=os.getenv('MONGO_URI'),
        database_name=os.getenv('MONGO_DB'),
        collection_name=os.getenv('MONGO_COLLECTION'),
    config_file=config_file
)
config = MongoExtractor.load_config(config_file)

# Example DataFrame
df = mongo_extractor.get_dataframe()
# df = df.fillna(value=0)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming df is your DataFrame
# Step 1: Preprocessing
# df = pd.read_csv('/home/andrea/Desktop/Sarcoma-DT/output.csv')
# # Select relevant features
# features = df[['doc_1.diagnosis.initial_size_a',
#                 'doc_1.diagnosis.initial_size_b',
#                 'doc_1.diagnosis.initial_size_c',
#                 'patient_info.age_at_admission',
#                 'status.metastasis_initial_presentation_exists']]

# # Standardize the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Encode the labels
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(df['status.status_last_patient_contact_information'])

# # Create a mapping for new labels
# new_labels = {0: 'NED', 1: 'AWD', 2: 'DOD'}  # Update indices as necessary based on your data

# # Step 2: t-SNE Implementation
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(scaled_features)

# # Step 3: Plotting
# plt.figure(figsize=(10, 6))
# palette = sns.color_palette("viridis", n_colors=len(new_labels))  # Define color palette
# sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=palette, legend='full')

# # Update the legend with new labels
# handles, _ = plt.gca().get_legend_handles_labels()
# new_labels_list = [new_labels[i] for i in range(len(new_labels))]
# plt.legend(handles, new_labels_list, title='Last Patient Contact Status', loc='upper right')

# plt.title('')
# plt.xlabel('Dim 1')
# plt.ylabel('Dim 2')
# plt.show()




dc = config['data_config']

# Define a function to summarize numerical features
import numpy as np
from scipy import stats

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def summarize_numerical(df, column_name, ci_method='median', confidence_level=0.95):
    """
    Summarizes a numerical column with mean, median, and confidence interval.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to summarize.
        ci_method (str): The method to calculate confidence intervals.
                         Options: 'parametric', 'median', 'log_transformed'.
        confidence_level (float): Confidence level for the CI (default 0.95).

    Returns:
        str: Summary string with mean, median, and confidence interval.
    """
    data = df[column_name].dropna()  # Remove NaN values
    mean = data.mean()              # Calculate the mean
    median = data.median()          # Calculate the median
    std = data.std()                # Calculate the standard deviation
    n = len(data)                   # Sample size

    ci = "95% CI: Not available (n ≤ 1)"  # Initialize confidence interval
    
    if n > 1:
        if ci_method == 'parametric':
            # Parametric CI for mean
            margin_of_error = stats.norm.ppf(1 - (1 - confidence_level) / 2) * (std / np.sqrt(n))
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error

        elif ci_method == 'median':
            # Median-based CI using binomial distribution
            data_sorted = np.sort(data)
            alpha = 1 - confidence_level
            lower_index = int(np.floor((alpha / 2) * n))
            upper_index = int(np.ceil((1 - (alpha / 2)) * n)) - 1
            lower_bound = data_sorted[lower_index]
            upper_bound = data_sorted[upper_index]

        elif ci_method == 'log_transformed':
            # Log-transformed CI for skewed data
            if (data <= 0).any():
                raise ValueError("Data contains non-positive values, cannot apply log transformation.")
            log_data = np.log(data)
            mean_log = log_data.mean()
            std_log = log_data.std()
            margin_of_error = stats.norm.ppf(1 - (1 - confidence_level) / 2) * (std_log / np.sqrt(n))
            lower_log = mean_log - margin_of_error
            upper_log = mean_log + margin_of_error
            lower_bound = np.exp(lower_log)
            upper_bound = np.exp(upper_log)
        else:
            raise ValueError("ci_method must be one of ['parametric', 'median', 'log_transformed']")

        ci = f"{int(confidence_level * 100)}% CI: [{lower_bound:.3f}, {upper_bound:.3f}]"

    # Plot the distribution
    # plt.figure(figsize=(10, 6))
    # data.plot(kind='hist', bins=30, alpha=0.7, edgecolor='black')
    # plt.title(f'Distribution of {column_name}', fontsize=16)
    # plt.xlabel('Value', fontsize=14)
    # plt.ylabel('Frequency', fontsize=14)
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()

    return f"Mean: {mean:.3f}, Median: {median:.3f}, {ci}"




# Define a function to summarize categorical features
def summarize_categorical(df, column_name):
    counts = df[column_name].value_counts()
    percentages = 100 * counts / len(df)
    summary = '; '.join([f"{value} = {percent:.2f}%({count})" for value, count, percent in zip(counts.index, counts, percentages)])
    return summary


categorical_features = dc['categorical_fields']
all_features = df.columns.tolist()
numerical_features = list(set(all_features) - set(categorical_features) - set(dc['exclude_fields']))



# Create a summary DataFrame
summary_data = []
for col in numerical_features:
    summary_data.append({
        'Variable': col,
        'Clinical traits (Numbers in parentheses are quantities)': summarize_numerical(df, col)
    })

for col in categorical_features:
    summary_data.append({
        'Variable': col,
        'Clinical traits (Numbers in parentheses are quantities)': summarize_categorical(df, col)
    })

summary_df = pd.DataFrame(summary_data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(summary_df)

# Your script or processing steps here
print("Press Enter to continue...")
input()  # Waits for user input

# >>> df.iloc[0]
# Age at Diagnosis Mean ± SD = 61.110 ± 12.965
# Type of Breast Surgery Mastectomy = 63.59%(1196); Breast Conserving = 36.40%(784)
# Cancer Type Breast Cancer = 100%(1980)
# HER2 Status Positve = 87.52%(1733); Negative = 12.48%(247)
# Tumor Size Mean ± SD = 26.27 ± 15.39
# Gender Female = 100%(1980)
# Cellularity High = 51.44%(1019); Moderate = 37.70%(746); Low = 10.86%(215)
# Chemotherapy No = 79.18%(1568); Yes = 20.82%(412)
# Pam50 + Claudin-low subtype LumA = 35.52%(704); LumB = 23.85%(472); Her2 = 11.32%(224); claudin-low = 10.97%(217); Basal = 10.56%(209); Normal
# = 7.48%(148); NC = 0.30%(6)
# Cohort 3 = 38.55%(764); 1 = 26.43%(523); 2 = 14.55%(288); 4 = 11.87%(235); 5 = 8.59%(170)
# ER status measured by IHC Positve = 77.82%(1541); Negative = 22.18%(439)
# ER Status Positve = 76.15%(1508); Negative = 23.85%(472)
# Neoplasm Histologic Grade 3 = 52.45%(1039); 2 = 39.01%(772); 1 = 8.54%(169)
# HER2 status measured by SNP6 Neutral = 72.51%(1436); Gain = 22.13%(438); Loss = 5.10%(101); Undef = 0.25%(5)
# Cancer Type Detailed Breast Invasive Ductal Carcinoma = 77.61%(1537); Breast Mixed Ductal and Lobular Carcinoma = 10.66%(211); Breast
# Invasive Lobular Carcinoma = 7.38%(146); Invasive Breast Carcinoma = 2.22%(44); Breast Invasive Mixed Mucinous
# Carcinoma = 1.16%(23); Breast = 0.86%(17); Metaplastic Breast Cancer = 0.10%(2)
# Tumor Other Histologic Subtype Ductal/NST = 77.51%(1535); Mixed = 10.66%(211); Lobular = 7.38%(146); Medullary = 1.26%(25); Mucinous = 1.16%(23);
# Tubular/ cribriform = 1.06%(21); Other = 0.86%(17); Metaplastic = 0.10%(1)
# Hormone Therapy Yes = 61.55%(1219); No = 38.45%(761)
# Inferred Menopausal State left = 78.58%(1556); right = 21.42%(424)
# Nottingham prognostic index 1∼3 = 9.04%(179); 3∼6 = 81.05%(1605); 6∼6.68 = 10.06%(199)
# Primary Tumor Laterality Positve = 54.27%(1075); Negative = 45.73%(905)
# Lymph nodes examined positive 0∼10 = 95.45%(1890); 11∼20 = 3.69%(73); 21∼41 = 0.96%(19)
# Mutation Count 0∼10 = 92.88%(1839); 11∼20 = 6.42%(127); 21∼80 = 1.01%(20)
# Integrative Cluster 8 = 15.12%(300); 3 = 14.66%(291); 4ER+ = 13.04%(258); 10 = 11.43%(226); 5 = 9.61%(190); 7 = 9.61%(190); 9 =
# 7.38%(146); 1 = 7.03%(139); 6 = 4.30%(85); 4ER- = 4.20%(83); 2 = 3.64%(72)
# Oncotree Code IDC = 77.56%(1536); MDLC = 10.66%(211); ILC = 7.23%(145); BRCA = 2.22%(44); IMMC = 1.16%(23); BREAST = 0.86%(17);
# PBS = 0.10%(2); MBC = 0.10%(2)
# Relapse Free Status (Months) Mean ± SD =110.10 ± 76.33
# Relapse Free Status Not Recurred = 59.50%(1179); Recurred = 40.50%(801)
# Tumor Stage 2 = 66.75%(1322); 1 = 26.07%(516); 3 = 5.96%(118); 0 = 0.61%(12); 4 = 0.51%(10); 1.5 = 0.10%(2)
# 3-Gene classifier subtype ER+/HER2-Low Prolif = 40.17%(796); ER+/HER2-High Prolif = 33.75%(668); ER-/HER2- = 16.07%(318); HER2+ = 10.01%(198)
# PR Status Positve = 52.53%(1040); Negative = 47.47%(940)
# Radio Therapy Yes = 59.19%(1172); No = 40.81%(808)

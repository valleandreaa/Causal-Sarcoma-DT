import pandas as pd

# Load the Excel file
file_path = '/home/andrea/Desktop/Sarcoma-DT/ETL/data/adjumed_export_20241027-120926.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load each sheet
cases_df = excel_data.parse('Cases')
interventions_df = excel_data.parse('interventions')

# Ensure "Patient ID (PID)" is in each dataset; replace 'Patient ID (PID)' with the actual column name if different
pid_column = 'Patient ID (PID)'

# Sample 200 unique patients by Patient ID from each dataset
cases_sample = cases_df.drop_duplicates(subset=pid_column).sample(n=200, random_state=42)

# Filter interventions based on the selected Patient IDs from cases_sample
selected_pids = cases_sample[pid_column]
interventions_sample = interventions_df[interventions_df[pid_column].isin(selected_pids)]

# Save the samples to a new Excel file for the test database
with pd.ExcelWriter('ETL/data/adjumed_export_20241027-120926_test.xlsx') as writer:
    cases_sample.to_excel(writer, sheet_name='Cases', index=False)
    interventions_sample.to_excel(writer, sheet_name='interventions', index=False)

print("Sampled data saved to 'test_db_sample.xlsx'")
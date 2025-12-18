import yaml
from Models.utils.helpers import MongoExtractor


# Create a MongoExtractor instance
mongo_extractor = MongoExtractor(
    connection_string='mongodb://localhost:27017/',
    database_name='SarcomaDB',
    collection_name='Patients',
    config_file='Models/configs/config_v1_1.yaml'
)


data_list = mongo_extractor.fetch_data()

df = mongo_extractor.get_dataframe()

# >>> df.iloc[0]
# _id                                                                   663344
# dates.date_of_biopsy                                     2024-06-11 00:00:00
# patient_info.patient_id                                               663344
# patient_info.year_of_entry                                              2024
# patient_info.clinic_no.0                                                   3
# patient_info.clinic_no.1                                               30301
# patient_info.entry_date                                  2024-06-04 00:00:00
# patient_info.newest_date_of_sarcomaboard                 2024-06-11 00:00:00
# patient_info.year_of_birth                                              1955
# patient_info.gender                                                        2
# patient_info.responsible_physician_first_presentation                      3
# patient_info.age_at_admission                                           69.0
# patient_info.age_class                                                   7.0
# Name: 0, dtype: object

feature_mapping = {

    'static': {
        'general': {
            'gender': 'Gender',
            'date_of_diagnosis': 'Date of histological diagnosis',
            'birth_date': 'Date of birth',
            'ajcc_uicc': 'ajcc_uicc_Timo',
            'sarculator_five': 'sarculator_five_Timo',
            'sarculator_ten': 'sarculator_five_Timo',
            'prom_score': 'prom_score_Timo',
            'prom_score_date': 'date_prom_Timo',
            'metastasis_post_treatment': 'Metastasis (After First Treatment)',
            'date_death':'date_death_Timo', 
            'date_follow_up': 'Date of last follow-up',
            'status': 'Status', 
            'date_first_patientcontact': 'date_first_patientcontact_Timo'
            
        },

        'tumor_characteristics': {
            'histological_diagnosis': 'Histological diagnosis',
            'grading_fnclcc': 'Grading (FNCLCC)',
            'lesion_site': 'Site of lesion (Basis)',
            'affected_tissue': 'Affected tissue',
            'lesion_side': 'Anatomic side of the lesion',
            'whoops': 'Whoops',
            'initial_metastasis': 'metastasis_initial_Timo',
            'number_metastasis': 'number_metastasis_Timo', 
            'initial_size_a': 'initialsize_a_Timo',
            'initial_size_b':  'initialsize_b_Timo',
            'initial_size_c':  'initialsize_c_Timo',
            'anatomicregion': 'anatomicregion_group_Timo',
            'anatomic_side_lesion' :'Anatomic side of lesion',
            'local_situation': '(0) OP_638 - Local situation_Timo',
        },

        'treatments': {
            'number_surgeries': 'number_all_operation_Timo',
            'disciplines_first_surgery': 'disciplines_first_planned_surgery_Timo',
            'complexity_first_planned_surger': 'complexity_first_planned_surgery_Timo',
            'chemo_first_indication': 'chemo_first_indication_Timo',
            'redonc_first_indication':'redonc_first_indication_Timo',
            'Date of 1st local recurrence': 'Date of 1st local recurrence',
            'Treatment of 1st LR': 'Treatment of 1st LR', 
            'initial_whoops': 'whoops_initial_diagnosis_Timo', 
            'whoops_surgeon': 'whoops_surgeon_Timo',
            'whoops_reoperation_margin_r_Timo': 'whoops_reoperation_margin_r_Timo'

          },
    },
    'dynamic': {
        'diagnosis': {
            'date_sarcomaboard': 'Date of SarcomaBoard',
            'fields': {
                'local_situation': '(0) OP_638 - Local situation_Timo',
                'whoops': 'Whoops',
                'date_localrecurrence_Timo': 'date_localrecurrence_Timo',
                'resection_grading' :'resection_grading_Timo',
                'resection_necrosis': 'resection_necrosis_timo',
                'resection_mitotic':'resection_mitotic_Timo',
                'resection_margin_mm': 'resection_margin_mm_Timo',
                'resection_margin_barrier': 'resection_margin_barrier_Timo',    
                'cci': 'cci_Timo',
                'dignity': 'dignity_timo',
                'biopsy_neoadjuvant': 'biopsy_neoadjuvant_Timo',
                
            }
        },

        'surgery': {
            'date_field': 'Date of index surgery',
            'date_sarcomaboard': 'Date of SarcomaBoard',
            'fields': {
                'tumor_max_size': 'Tumor maximal size before surgery',
                'disciplines': 'Disciplines first planned surgery Timo',
                'type_surgery': 'Type of index surgery',
                'margin_status': 'Margin status',
                'duration': 'duration_first_planned_surgery_Timo',
                'reoperation': 'Reoperation'
            }
        },

        'chemotherapy': {
            'date_field': 'Start date of line',
            'date_sarcomaboard': 'Date of SarcomaBoard',
            'fields': {
                'chemo_reason': 'Reason for Chemotherapy',
                'chemo_indication': 'chemo_indication_Timo',
                'chemo_type': 'Type of ChTx',                
                'chemo_substance': 'chemo_substance_Timo',
                'chemo_start': 'Start date of line',
                'chemo_end': 'Optional: End date of line',
                'chemo_discontinuation': 'chemo_discontinuation_Timo',
                'chemo_response': 'chemo_treatmentresponse_Timo'
            }
        },

        'radiotherapy': {
            'date_field': 'date_redonc_start_Timo',
            'date_sarcomaboard': 'Date of SarcomaBoard',
            'fields': {
                'radiotherapy_indication': 'Indication for radiotherapy',
                'radiotherapy_total_dose': 'Total dose used in radiotherapy',
                'start_radio':'date_redonc_start_Timo',
                'end_radio':'date_redonc_end',
                'redonc_indication': 'redonc_indication_Timo',
                'redonc_type':'redonc_type_Timo',
                'redonc_fractions': 'redonc_fractions_Timo',
                'redonc_total_gray': 'redonc_total_gray_Timo',
                'redonc_fractions_gray': 'redonc_fractions_gray_Timo',
                'hyperthermia_additional': '(all) OP_1170 - Additional, and concurrent treatments: (insb. Hyperthermia)_Timo',
                'hyperthermia_sessions': '(all) Hyperthermia (Number of sessions)_Timo',
                'endpoint': 'endpoint_Timo'
            }},

        'metastasis':{
            'date_sarcomaboard': 'Date of SarcomaBoard',
            'fields':{'Date of pulmonary metastasis': 'Date of pulmonary metastasis',
                      'Date of extrapulmonary metastasis': 'Date of extrapulmonary metastasis',
                      'treatment_metastasis': 'Treatment of metastasis',
                      'presence_metastasis':'Disease Status diagnosis / Presence of metastasis',
                      'follow_up':'metastasis_followup_Timo', 
                      'date_metastasis': 'date_metastasis_Timo'

        }
        }
    }
}

label_dict = {

    "resection_margin_barrier":{
        "mapping": {
                "1 fascia": "fascia",
                "2 periostem": "periosteum"
            },
        "mode": "token"
    },
    "dignity": {
        "mapping": {
            "benigne": "benigne",
            "maligne": "maligne",
            "intermidiate": "intermediate"
        },
        "mode": "token"
    },
    "treatment_metastasis": {
        "mapping": {
            "[0] Surgery": "surgery",
            "[1] Chemotherapy": "chemotherapy",
            "[2] Radiotherapy": "radiotherapy",
            "[3] Surgery + Radiotherapy": "surgery_radiotherapy",
            "[4] Surgery + Chemotherapy": "surgery_chemotherapy",
            "[5] Surgery + Chemotherapy + Radiotherapy": "surgery_chemotherapy_radiotherapy",
            "[7] Radiotherapy + Chemotherapy": "radiotherapy_chemotherapy",
            "[6] Best supportive care": "best_supportive_care"
        },
        "mode": "substring"
    },
    "chemo_type": {
        "mapping": {
            "[0] Cyto-toxic chemotherapy": "cytotoxic",
            "[1] Targeted therapy": "targeted",
            "[2] Immunotherapy": "immuno",
            "[3] Chemotherapy with Concomitant Hyperthermia": "chemo+hyperthermia",
            "[4] Isolated limb perfusion": "ILP",
            "[5] Electrochemotherapy": "electrochemo",
            "[6] Other: CAR-T-Cell, vaccines etc. (within framework of trials)": "experimental"
        },
        "mode": "token"
    },
    "ajcc_uicc": {
        "mapping": {
            1: "IA",
            2: "IB",
            3: "II",
            4: "IIA",
            5: "IIB",
            6: "III",
            7: "IIIA",
            8: "IIIB",
            9: "IV",
            10: "IVA"
        },
        "mode": "substring" 
    },
    "resection_mitotic": {
        "mapping": {
            1: "0-9",
            2: "10-19",
            3: ">29",
        },
    "mode": "substring" 
    },
    "resection_necrosis": {
        "mapping": {
            1: "<10%",
            2: "10-20%",
            3: "21-30%",
            4: "31-40%",
            5: "41-50%",
            6: "51-60%",
            7: "61-70%",
            8: "71-80%",
            9: "81-90%",
            10: ">90%"
        },
        "mode": "substring"
    },
    "histological_diagnosis": {
        "mapping": {
            "[0] Angiosarcoma of soft tissues": "Angiosarcoma of soft tissues",
            "[1] chondromatous": "Atypical chondromatous tumor",
            "[2] lipomatous  well differentiated": "Atypical lipomatous tumor",
            "[3] Chondrosarcoma": "Chondrosarcoma",
            "[4] Conventional osteosarcoma": "Conventional osteosarcoma",
            "[5] Dedifferentiated": "Dedifferentiated liposarcoma",
            "[6] Desmoid type fibromatosis": "Desmoid‐type fibromatosis",
            "[7] Ewing": "Ewing sarcoma",
            "[8] Giant cell of bone": "Giant cell tumor of bone",
            "[9] GIST": "GIST",
            "[10] Intramuscular myxoma (incl. variants)": "Intramuscular myxoma",
            "[11] Leiomyosarcoma (excluding skin)": "Leiomyosarcoma",
            "[12] Myxofibrosarcoma": "Myxofibrosarcoma",
            "[13] Myxoid": "Myxoid liposarcoma",
            "[14] Pleomorphic": "Pleomorphic liposarcoma",
            "[15] Solitary fibrous": "Solitary fibrous tumor",
            "[16] Synovial": "Synovial sarcoma",
            "[17] Tenosynovial giant cell": "Tenosynovial giant cell tumor",
            "[18] Undifferentiated unclassified": "Undifferentiated / unclassified sarcoma"
        },
        "mode": "token"
    },
    "grading_fnclcc": {
        "mapping": {
            "benigne": "benigne",
            "G1 low": "G1",
            "G2 intermidiate": "G2",
            "G3 high": "G3"
        },
        "mode": "token"
    },
    "lesion_site": {
        "mapping": {
                "Foot": "Foot",
                "knee": "knee",
                "Popliteal fossa": "Popliteal fossa/knee",
                "Buttocks inguinal": "Buttocks/inguinal",
                "Hand": "Hand",
                "limb": "limb",
                "Elbow": "Elbow",
                "Axilla scapula": "Axilla/scapula",
                "Head neck": "Head/neck",
                "Chest": "Chest",
                "Abdominal": "Abdominal",
                "Paraspinal": "Paraspinal",
                "Retroperitoneal": "Retroperitoneal",
                "Uterus": "Uterus",
                "Other visceral": "Other visceral",
                "Intrathoracic C-spine": "Intrathoracic C-spine",
                "T-spine": "T-spine",
                "L-spine": "L-spine",
                "Sacrum": "Sacrum",
                "Scapula": "Scapula",
                "humerus": "humerus",
                "forearm": "forearm",
                "Carpus": "Carpus",
                "Hand": "Hand",
                "Clavicle": "Clavicle",
                "pelvis": "pelvis",
                "Acetabulum": "Acetabulum",
                "femur": "femur",
                "tibia": "tibia",
                "Foot": "Foot"
            },
            "mode": "token"
        },
        "anatomic_side_lesion": {
            "mapping": {
                "left": "left",
                "right": "right",
                "midline": "midline"},
            "mode": "token"
        },
        "local_situation": {
            "mapping": {
                "no prior interventions therapies": "no prior interventions",
                "locally advanced incl path fracture": "locally advanced incl path fracture",
                "whoops": "whoops"
            },
            "mode": "token"
        },
        "whoops": {
            "mapping": {
                "no": 0,
                "yes": 1,
            },
            "mode": "token"
        },
        "initial_metastasis":
            {
                "mapping": {
                    "0 0.0 no": 0,
                    "1 1.0 yes": 1,
                },
                "mode": "token"
            },
        "presence_metastasis": {
            "mapping": {
                "0 0.0 no metastasis": 0,
                "1 1.0 metastasis": 1,
            },
            "mode": "token"
        },

        "type_surgery": {
            "mapping": {
                "[1] 1st surgery for this reason": "first_surgery_reason",
                "[2] 1st surgery after whoops": "first_surgery_after_whoops",
                "[3] Pathological fracture": "pathological_fracture",
                "[4] 1st revision surgery": "first_revision_surgery",
                "[5] 2nd or more revision surgery": "subsequent_revision_surgery",
                "[6] 1st surgery for local recurrence": "first_local_recurrence",
                "[7] 2nd or more surgery for local recurrence": "subsequent_local_recurrence",
                "[8] Other reason": "other_reason",
                "[9] 1st surgery for metastasis": "first_metastasis_surgery",
                "[10] 2nd or more surgery for metastasis": "subsequent_metastasis_surgery"
            },
            "mode": "token"
        },
        "margin_status": {
            "mapping": {
                "R0": "R0",
                "R1": "R1",
                "R2": "R2"
            },
            "mode": "token"
        },
        "chemo_reason": {
            "mapping": {
                "adjuvant": "adjuvant",
                "neoadjuvant": "neoadjuvant",
                "palliative": "palliative",
            },
            "mode": "token"
        },
        "radiotherapy_indication": {
            "mapping": {
            "[0] no radiotherapy": "no radiotherapy",
            "[1] preoperative": "preoperative",
            "[2] postoperative": "postoperative",
            "[3] definitive": "definitive",
            "[5] palliative": "palliative",
            "[8] other unknown": "other/ unknown"
            },
            "mode": "token"
        },
        "hyperthermia_additional": {
            "mapping": {
                "hyperthermia": "hyperthermia",
            },
            "mode": "token"
        },
        "status": {
            "mapping": {
                "[1] 1  (NED)": "NED",
                "[2] 2  (AWD)": "AWD",
                "[3] 3  (DOD)": "DOD",
            },
            "mode": "token"
                }
}

column_renaming_map = {
    "Pat ID": "patient_id",
    'Gender': 'gender',
    'Date of histological diagnosis': 'date_of_diagnosis',
    'Date of birth': 'birth_date',
    'Anatomic side of lesion':'anatomic_side_lesion',
    'biopsy_neoadjuvant_Timo': 'biopsy_neoadjuvant',
    'ajcc_uicc_Timo': 'ajcc_uicc',
    'sarculator_five_Timo': 'sarculator_five',
    'sarculator_five_Timo': 'sarculator_ten',
    'prom_score_Timo': 'prom_score',
    'date_prom_Timo': 'prom_score_date',
    'endpoint_Timo': 'endpoint',
    'Metastasis (After First Treatment)': 'metastasis_post_treatment',
    'date_death_Timo': 'date_death',
    'Date of last follow-up': 'date_follow_up',
    'Status': 'status',
    'Date of SarcomaBoard': 'date_sarcomaboard',
    'date_first_patientcontact_Timo': 'date_first_patientcontact',
    'Histological diagnosis': 'histological_diagnosis',
    'Grading (FNCLCC)': 'grading_fnclcc',
    'Site of lesion (Basis)': 'lesion_site',
    'Affected tissue': 'affected_tissue',
    'Anatomic side of the lesion': 'lesion_side',
    'Whoops': 'whoops',
    'metastasis_initial_Timo': 'initial_metastasis',
    'number_metastasis_Timo': 'number_metastasis',
    'initialsize_a_Timo': 'initial_size_a',
    'initialsize_b_Timo': 'initial_size_b',
    'initialsize_c_Timo': 'initial_size_c',
    'anatomicregion_group_Timo': 'anatomicregion',
    
    'number_all_operation_Timo': 'number_surgeries',
    'disciplines_first_planned_surgery_Timo': 'disciplines_first_surgery',
    'complexity_first_planned_surgery_Timo': 'complexity_first_planned_surger',
    'chemo_first_indication_Timo': 'chemo_first_indication',
    'redonc_first_indication_Timo': 'redonc_first_indication',
    'Date of 1st local recurrence': 'Date of 1st local recurrence',
    'Treatment of 1st LR': 'Treatment of 1st LR',
    'whoops_initial_diagnosis_Timo': 'initial_whoops',
    'whoops_surgeon_Timo': 'whoops_surgeon',
    'whoops_reoperation_margin_r_Timo': 'whoops_reoperation_margin_r_Timo',
    
    '(0) OP_638 - Local situation_Timo': 'local_situation',
    'date_localrecurrence_Timo': 'date_localrecurrence_Timo',
    'resection_grading_Timo': 'resection_grading',
    'resection_necrosis_timo': 'resection_necrosis',
    'resection_mitotic_Timo': 'resection_mitotic',
    'resection_margin_mm_Timo': 'resection_margin_mm',
    'resection_margin_barrier_Timo': 'resection_margin_barrier',
    'cci_Timo': 'cci',
    'dignity_timo': 'dignity',

    'Date of index surgery': 'date_field',
    'Tumor maximal size before surgery': 'tumor_max_size',
    'indication_first_planned_surgery_Timo': 'indication_first_treatment',
    'Disciplines first planned surgery Timo': 'disciplines',
    'Type of index surgery': 'type_surgery',
    'Margin status': 'margin_status',
    'duration_first_planned_surgery_Timo': 'duration',
    'Reoperation': 'reoperation',
    'chemo_discontinuation_Timo': 'chemo_discontinuation',

    'Start date of line': 'chemo_start',
    'Reason for Chemotherapy': 'chemo_reason',
    'chemo_indication_Timo': 'chemo_indication',
    'Type of ChTx': 'chemo_type',
    'chemo_substance_Timo': 'chemo_substance',
    'Optional: End date of line': 'chemo_end',
    'chemo_treatmentresponse_Timo': 'chemo_response',
    
    'date_redonc_start_Timo': 'start_radio',
    'Indication for radiotherapy': 'radiotherapy_indication',
    'Total dose used in radiotherapy': 'radiotherapy_total_dose',
    'date_redonc_end': 'end_radio',
    'redonc_indication_Timo': 'redonc_indication',
    'redonc_type_Timo': 'redonc_type',
    'redonc_fractions_Timo': 'redonc_fractions',
    'redonc_total_gray_Timo': 'redonc_total_gray',
    'redonc_fractions_gray_Timo': 'redonc_fractions_gray',
    '(all) OP_1170 - Additional, and concurrent treatments: (insb. Hyperthermia)_Timo': 'hyperthermia_additional',
    '(all) Hyperthermia (Number of sessions)_Timo': 'hyperthermia_sessions',

    'Date of pulmonary metastasis': 'Date of pulmonary metastasis',
    'Date of extrapulmonary metastasis': 'Date of extrapulmonary metastasis',
    'Treatment of metastasis': 'treatment_metastasis',
    'Disease Status diagnosis / Presence of metastasis': 'presence_metastasis',
    'metastasis_followup_Timo': 'follow_up',
    'date_metastasis_Timo': 'date_metastasis'
}



data_type_mapping = {

    'gender': str,
    'date_of_diagnosis': 'datetime',
    'date_sarcomaboard': 'datetime',
    'birth_date': int,
    'biopsy_neoadjuvant': int,
    'mitotic_count': int,
    'ajcc_uicc': str,
    'date_first_patientcontact': 'datetime',
    'sarculator_5y': float,
    'sarculator_10y': float,
    'prom_score': float,
    'prom_score_date': 'datetime',
    'endpoint': int,
    'chemo_discontinuation': list, 
    'metastasis_post_treatment': int,
    'date_death': 'datetime',
    'date_follow_up': 'datetime',
    'status': str,
    'anatomic_side_lesion': str,
    'histological_diagnosis': str,
    'grading_fnclcc': str,
    'lesion_site': str,
    'affected_tissue': str,
    'lesion_side': str,
    'whoops': str,
    'initial_metastasis': str,
    'number_metastasis': int,
    'initial_size_a': float,
    'initial_size_b': float,
    'initial_size_c': float,
    'anatomicregion': str,
    'number_surgeries': int,
    'disciplines_first_surgery': int,
    'complexity_first_planned_surger': int,
    'chemo_first_indication': int,
    'redonc_first_indication': int,
    'initial_whoops': int,
    'whoops_surgeon': str,
    'whoops_reoperation_margin_r_Timo': int,
    'sarculator_five': float,
    'sarculator_ten': float,
    'local_situation': str,
    'date_localrecurrence_Timo': 'datetime',
    'resection_grading': int,
    'resection_necrosis': str,
    'resection_mitotic': str,
    'resection_margin_mm': float,
    'resection_margin_barrier': str,
    'cci': float,
    'dignity': int,

    'tumor_max_size': int,
    'indication_first_treatment': str,
    'disciplines': str,
    'type_surgery': str,
    'margin_status': str,
    'duration': float,
    'reoperation': int,

    'chemo_reason': str,
    'chemo_indication': int,
    'chemo_type': str,
    'chemo_substance': list,
    'chemo_start': 'datetime',
    'chemo_end': 'datetime',
    'chemo_response': int,

    'radiotherapy_indication': str,
    'radiotherapy_total_dose': float,
    'start_radio': 'datetime',
    'end_radio': 'datetime',
    'redonc_indication': list,
    'redonc_type': int,
    'redonc_fractions': int,
    'redonc_total_gray': int,
    'redonc_fractions_gray': float,
    'hyperthermia_additional': str,
    'hyperthermia_sessions': int,

    'date_field': 'datetime',
    'Date of 1st local recurrence': 'datetime',
    'Treatment of 1st LR': int,

    'Date of pulmonary metastasis': 'datetime',
    'Date of extrapulmonary metastasis': 'datetime',
    'treatment_metastasis': str,
    'presence_metastasis': str,
    'follow_up': int,
    'date_metastasis': 'datetime'
}
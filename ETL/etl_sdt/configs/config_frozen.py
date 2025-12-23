feature_mapping = {

    "static": {
        "general": [
            "gender",
            "institution",
            "date_pathology_report"
        ],

        "tumor_characteristics": [
            "biopsy_grading",
            "who_diagnosis_code",
            "anatomic_region_code",
            "anatomic_region_grouping",
            "anatomic_region_side",
            "metastasis_present_at_diagnosis"
        ],

        "treatments": [
            "date_whoops",
            "whoops_margin_status",
            "whoops_surgery_institution",
            "date_index_surgery",
            "surgery_indication",
            "surgery_institution",
            "tumor_max_size_before_surgery",
            "pathologist_margin_judgement",
        ],
    },

    "dynamic": {
        "diagnosis": {
            "date_sarcoma_board": "date_sarcoma_board",
            "fields": [
                "ctcae",
                "ctcae_reason_latest",
                "comment_code",
                "comment",
                "other_diagnoses",
                "patient_history_latest",
                "whoops_initial_diagnosis",
                "whoops_initial_diagnosis_code",
                "whoops_surgeon",
                "whoops_reoperation_margin_r",
            ],
        },

        "surgery": {
            "date_field": "date_index_surgery",
            "episodes_field": "episodes_surgery",
            "fields": [
                "resection_grading",
                "whoops",
                "whoops_margin_status",
                "tumor_max_size_before_surgery",
                "pathologist_margin_judgement",
                "surgery_indication",
                "surgery_institution",

                
            ],
        },

        "systemic_therapy": {
            "date_field": "cycle_start_date",
            "episodes_field": "episodes_systemic",
            "fields": [
                "systemic_treatment_reason",
                "line_of_treatment",
                "systemic_therapy_type",
                "drug_name",
                "systemic_therapy_discontinuation_reason",
                "num_cycles_executed",
                "toxicity_type",
                "ctcae_grade",
            ],
        },

        "radiotherapy": {
            "date_field": "radiotherapy_start_date",
            "episodes_field": "episodes_radiotherapy",
            "fields": [
                "radiotherapy_start_date",
                "radiotherapy_end_date",
                "radiotherapy_indication",
                "radiotherapy_num_fractions",
                "radiotherapy_type",
                "radiotherapy_ptv",
                "radiotherapy_gtv",
                "radiotherapy_total_dose"
            ],
        },

        "local_recurrence": {
            "date_field": "date_local_recurrence",
            "episodes_field": "episodes_local_recurrence",
            "fields": [
                "date_local_recurrence",
                "treatment_local_recurrence",
                "date_local_recurrence_surgery",
            ],
        },

        "metastasis": {
            "date_field": "date_metastasis",
            "episodes_field": "episodes_metastasis",
            "fields": [
                "metastasis_type",  # pulmonary or extrapulmonary
                "site_extrapulmonary_metastasis",
            ],
        },

        "follow_up": {
            "date_field": "date_last_follow_up",
            "episodes_field": "episodes_follow_up",
            "fields": [
                "last_status",
                "date_death",
                "metastasis_at_follow_up",
            ],
        },
}}


label_dict = {
    "gender": {
        "mapping": {
            "[1] male": "male",
            "[2] female": "female",
        },
        "mode": "token",
    },

    "general_consent_agreed": {
        "mapping": {
            "no": 0,
            "yes": 1,
            "unknown": -1,
        },
        "mode": "token",
    },

    "biopsy_type": {
        "mapping": {
            "0 no biopsy": "no_biopsy",
            "1 fine needle": "fine_needle",
            "2 core biopsy (imaging guided)": "core_biopsy_imaging_guided",
            "3 open incisional: without suspicion of sarcoma": "open_incisional_without_suspicion_sarcoma",
            "4 open incisional: with suspicion of sarcoma": "open_incisional_with_suspicion_sarcoma",
            "5 excisional: without suspicion of sarcoma": "excisional_without_suspicion_sarcoma",
            "6 excisional: with suspicion of sarcoma": "excisional_with_suspicion_sarcoma",
            "7 curretage": "curettage",
            "8 unknown": "unknown",
        },
        "mode": "token",
    },

    "biopsy_neoadjuvant": {
        "mapping": {
            "0 no": "no",
            "1 yes": "yes",
        },
        "mode": "token",
    },

    "biopsy_grading": {
        "mapping": {
            "Not a sarcoma": "not_a_sarcoma",
            "[1] G1": "G1",
            "[2] G2": "G2",
            "[3] G3": "G3",
            "benign": "benign",
            "[5] suspicious of malignancy": "suspicious_of_malignancy",
            "[6] non-diagnostic": "non_diagnostic",
            "NA": "na",
        },
        "mode": "token",
    },

    "resection_grading": {
        "mapping": {
            "Not a sarcoma": "not_a_sarcoma",
            "[1] G1": "G1",
            "[2] G2": "G2",
            "[3] G3": "G3",
            "benign": "benign",
            "[5] suspicious of malignancy": "suspicious_of_malignancy",
            "[6] non-diagnostic": "non_diagnostic",
            "NA": "na",
        },
        "mode": "token",
    },

    "who_diagnosis_code": {
        "mapping": {
            "[0] Angiosarcoma of soft tissues": "angiosarcoma_soft_tissues",
            "[1] Atypical chondromatous tumor": "atypical_chondromatous_tumor",
            "[2] Atypical lipomatous tumor / well differentiated liposarcoma": "atypical_lipomatous_tumor_wd_liposarcoma",
            "[3] Chondrosarcoma": "chondrosarcoma",
            "[4] Conventional osteosarcoma": "conventional_osteosarcoma",
            "[5] Dedifferentiated liposarcoma": "dedifferentiated_liposarcoma",
            "[6] Desmoid‐type fibromatosis": "desmoid_type_fibromatosis",
            "[7] Ewing sarcoma": "ewing_sarcoma",
            "[8] Giant cell tumor of bone": "giant_cell_tumor_bone",
            "[9] GIST": "gist",
            "[10] Intramuscular myxoma (incl. variants)": "intramuscular_myxoma",
            "[11] Leiomyosarcoma (excluding skin)": "leiomyosarcoma_excluding_skin",
            "[12] Myxofibrosarcoma": "myxofibrosarcoma",
            "[13] Myxoid liposarcoma": "myxoid_liposarcoma",
            "[14] Pleomorphic liposarcoma": "pleomorphic_liposarcoma",
            "[15] Solitary fibrous tumor": "solitary_fibrous_tumor",
            "[16] Synovial sarcoma": "synovial_sarcoma",
            "[17] Tenosynovial giant cell tumor": "tenosynovial_giant_cell_tumor",
            "[18] Undifferentiated / unclassified sarcoma": "undifferentiated_unclassified_sarcoma",
        },
        "mode": "token",
    },

    "anatomic_region_code": {
        "mapping": {
            "[0] Foot": "foot",
            "[1] Distal lower limb below knee - anterior compartment": "distal_lower_limb_below_knee_anterior",
            "[2] Distal lower limb below knee - lateral compartment": "distal_lower_limb_below_knee_lateral",
            "[3] Distal lower limb below knee - posterior compartment": "distal_lower_limb_below_knee_posterior",
            "[4] Distal lower limb below knee - more than one compartment": "distal_lower_limb_below_knee_multi_compartment",
            "[5] Popliteal fossa/knee": "popliteal_fossa_knee",
            "[6] Proximal lower limb at or above knee - anterior compartment": "proximal_lower_limb_at_or_above_knee_anterior",
            "[7] Proximal lower limb at or above knee - posterior compartment": "proximal_lower_limb_at_or_above_knee_posterior",
            "[8] Proximal lower limb at or above knee - more than one compartment": "proximal_lower_limb_at_or_above_knee_multi_compartment",
            "[9] Buttocks and inguinal region": "buttocks_inguinal_region",
            "[10] Hand": "hand",
            "[11] Distal upper limb below elbow - ventral compartment": "distal_upper_limb_below_elbow_ventral",
            "[12] Distal upper limb below elbow - dorsal compartment": "distal_upper_limb_below_elbow_dorsal",
            "[13] Distal upper limb below elbow - more than one compartment": "distal_upper_limb_below_elbow_multi_compartment",
            "[14] Elbow": "elbow",
            "[15] Proximal upper limb at or above the elbow - ventral compartment": "proximal_upper_limb_at_or_above_elbow_ventral",
            "[16] Proximal upper limb at or above the elbow - dorsal compartment": "proximal_upper_limb_at_or_above_elbow_dorsal",
            "[17] Proximal upper limb at or above the elbow - more than one compartment": "proximal_upper_limb_at_or_above_elbow_multi_compartment",
            "[18] Axilla/scapula": "axilla_scapula",
            "[19] Head and neck": "head_neck",
            "[20] Chest wall": "chest_wall",
            "[21] Abdominal wall": "abdominal_wall",
            "[22] Paraspinal": "paraspinal",
            "[23] Retroperitoneal": "retroperitoneal",
            "[24] Uterus": "uterus",
            "[25] Other visceral": "other_visceral",
            "[26] Intrathoracic": "intrathoracic",
            "[27] T-spine": "t_spine",
            "[28] L-spine": "l_spine",
            "[29] Sacrum": "sacrum",
            "[30] Scapula": "scapula",
            "[31] Prox humerus": "prox_humerus",
            "[32] Mid humerus": "mid_humerus",
            "[33] Dist humerus": "dist_humerus",
            "[34] Prox forearm": "prox_forearm",
            "[35] Mid forearm": "mid_forearm",
            "[36] Dist forearm": "dist_forearm",
            "[37] Carpus": "carpus",
            "[38] Hand": "hand_2",
            "[39] Clavicle": "clavicle",
            "[40] Post pelvis": "post_pelvis",
            "[41] Acetabulum": "acetabulum",
            "[42] Ant pelvis": "ant_pelvis",
            "[43] Prox femur": "prox_femur",
            "[44] Mid femur": "mid_femur",
            "[45] Dist femur": "dist_femur",
            "[46] Prox tibia": "prox_tibia",
            "[47] Mid tibia": "mid_tibia",
            "[48] Dist tibia": "dist_tibia",
            "[49] Fibula": "fibula",
            "[50] Foot": "foot_2",
        },
        "mode": "token",
    },

    "anatomic_region_grouping": {
        "mapping": {
            "[0] Superficial Soft Tissue": "superficial_soft_tissue",
            "[1] Deep Soft Tissue (according to the investing fascia; note that a tumor located superficially but invading the investing fascia should be recorded as deep)": "deep_soft_tissue",
            "[2] Bone": "bone",
        },
        "mode": "token",
    },

    "anatomic_region_side": {
        "mapping": {
            "right": "right",
            "left": "left",
            "midline": "midline",
        },
        "mode": "token",
    },

    "whoops": {
        "mapping": {
            "[0] no": 0,
            "[1] yes": 1,
        },
        "mode": "token",
    },

    "surgery_indication": {
        "mapping": {
            "[1] 1st surgery for this reason": "first_surgery_for_reason",
            "[2]1st surgery after whoops": "first_surgery_after_whoops",
            "[3] Pathological fracture": "pathological_fracture",
            "[4] 1st revision surgery": "first_revision_surgery",
            "[5] 2nd or more revision surgery": "second_or_more_revision_surgery",
            "[6] 1st surgery for local recurrence": "first_surgery_for_local_recurrence",
            "[7] 2nd or more surgery for local recurrence": "second_or_more_surgery_for_local_recurrence",
            "[8] Other reason": "other_reason",
            "[9] 1st surgery for metastasis": "first_surgery_for_metastasis",
            "[10] 2nd or more surgery for metastasis": "second_or_more_surgery_for_metastasis",
        },
        "mode": "token",
    },

    "radiology_exam_type": {
        "mapping": {
            "1 conventional x-ray": "conventional_xray",
            "2 MRI": "mri",
            "3 CT": "ct",
            "4 ultrasound": "ultrasound",
            "5 PET-CT": "pet_ct",
        },
        "mode": "token",
    },

    "initial_radiology_exam_type": {
        "mapping": {
            "1 conventional x-ray": "conventional_xray",
            "2 MRI": "mri",
            "3 CT": "ct",
            "4 ultrasound": "ultrasound",
            "5 PET-CT": "pet_ct",
        },
        "mode": "token",
    },

    "first_radiotherapy_indication": {
        "mapping": {
            "1 preoperative": "preoperative",
            "2 postoperative": "postoperative",
            "3 intraoperativ": "intraoperative",
            "4 stereotactic": "stereotactic",
            "5 palliativ": "palliative",
        },
        "mode": "token",
    },

    "first_radiotherapy_type": {
        "mapping": {
            "1 intensity modulated radiotherapy": "imrt",
            "2 volumetric Arc VMAT": "vmat",
            "3 conventional 3D": "conventional_3d",
            "4 stereotactic": "stereotactic",
            "5 proton": "proton",
        },
        "mode": "token",
    },

    "treatment_local_recurrence_1": {
        "mapping": {
            "[0] Surgery": "surgery",
            "[1] Chemotherapy": "chemotherapy",
            "[2] Radiotherapy": "radiotherapy",
            "[3] Surgery + Radiotherapy": "surgery_radiotherapy",
            "[4] Surgery + Chemotherapy": "surgery_chemotherapy",
            "[5] Surgery + Chemotherapty + Radiotherapy": "surgery_chemotherapy_radiotherapy",
            "[7] Radiotherapy + Chemotherapy": "radiotherapy_chemotherapy",
            "[6] Best supportive care": "best_supportive_care",
        },
        "mode": "token",
    },

    "treatment_local_recurrence_2": {
        "mapping": {
            "[0] Surgery": "surgery",
            "[1] Chemotherapy": "chemotherapy",
            "[2] Radiotherapy": "radiotherapy",
            "[3] Surgery + Radiotherapy": "surgery_radiotherapy",
            "[4] Surgery + Chemotherapy": "surgery_chemotherapy",
            "[5] Surgery + Chemotherapy + Radiotherapy": "surgery_chemotherapy_radiotherapy",
            "[7] Radiotherapy + Chemotherapy": "radiotherapy_chemotherapy",
            "[6] Best supportive care": "best_supportive_care",
        },
        "mode": "token",
    },

    "metastasis_present_at_diagnosis": {
        "mapping": {
            "[0] No": "no",
            "[1] Yes": "yes",
            "[0] Localized disease": "localized_disease",
            "[1] Oligometastatic disease (5 or less lesions)": "oligometastatic_5_or_less",
            "[2] Polymetastatic disease (6 and more lesions)": "polymetastatic_6_or_more",
            "[3] unknown": "unknown",
        },
        "mode": "token",
    },

    "metastasis_at_follow_up": {
        "mapping": {
            "0 no": "no",
            "1 yes": "yes",
        },
        "mode": "token",
    },

    "last_status": {
        "mapping": {
            "[1] no evidence of disease (NED)": "NED",
            "[2] alive with disease (AWD)": "AWD",
            "[3] dead of disease (DOD)": "DOD",
            "[4] dead of other reasons (DOR)": "DOR",
            "[5] no assessment possible": "no_assessment_possible",
            "[6] lost to follow up": "lost_to_follow_up",
            "[8] other status": "other_status",
            "[9] unknown": "unknown",
        },
        "mode": "token",
    },
    "radiotherapy_indication": {
        "mapping": {
            "[1] preoperative": "preoperative",
            "[2] postoperative": "postoperative",
            "[3] definitive": "definitive",
            "[4] palliative": "palliative",
            "[8] other/ unknown": "other_unknown",
        },
        "mode": "token",
    },

    "radiotherapy_type": {
        "mapping": {
            "[1] Intensity modulated radiotherapy (IMRT)": "imrt",
            "[2] volumetric arc VMAT": "vmat",
            "[3] conventional 3D": "conventional_3d",
            "[4] stereotactic radiotherapy": "stereotactic",
            "[5] proton therapy": "proton",
            "[6] intraoperative (Linac)": "intraoperative_linac",
            "[7] intraoperative brachytherapy": "intraoperative_brachytherapy",
        },
        "mode": "token",
    },

    "hyperthermia_status": {
        "mapping": {
            "[0] no": 0,
            "[1] yes": 1,
        },
        "mode": "token",
    },
    "systemic_treatment_reason": {
        "mapping": {
            "[1] Curative intent: neoadjuvant": "curative_neoadjuvant",
            "[2] Curative intent: adjuvant": "curative_adjuvant",
            "[3] Palliative": "palliative",
            "[4] other: Freitext": "other_freetext",
        },
        "mode": "token",
    },

    "line_of_treatment": {
        "mapping": {
            "[1] 1° line": "line_1",
            "[2] 2° line": "line_2",
            "[3] 3° line": "line_3",
            "[4] 4° line": "line_4",
            "[5] further lines": "line_5_plus",
        },
        "mode": "token",
    },

    "systemic_therapy_type": {
        "mapping": {
            "[0] Cyto-toxic chemotherapy": "cytotoxic_chemotherapy",
            "[1] Targeted therapy": "targeted_therapy",
            "[2] Immunotherapy": "immunotherapy",
            "[3] Chemotherapy with Concomitant Hyperthermia": "chemo_with_hyperthermia",
            "[4] Isolated limb perfusion": "isolated_limb_perfusion",
            "[5] Electrochemotherapy": "electrochemotherapy",
            "[6] Other: CAR-T-Cell, vaccines etc. (within framework of trials)": "other_experimental",
        },
        "mode": "token",
    },

    "clinical_trial": {
        "mapping": {
            "[0] no": "no",
            "[1]  Phase I": "phase_1",
            "[2] Phase II": "phase_2",
            "[3] Phase III": "phase_3",
            "[4] Basket Trials (specify targets or mutations)": "basket_trial",
        },
        "mode": "token",
    },

    "systemic_therapy_discontinuation_reason": {
        "mapping": {
            "[0] [0] completed": "completed",
            "[1] [1] discontinued due to toxity": "discontinued_toxicity",
            "[2] [2] discontinued due to progressive disease": "discontinued_progressive_disease",
            "[3] discontinued due patient's wish": "discontinued_patient_wish",
            "[4] discontinued to patient's death": "discontinued_patient_death",
            "[5] other": "other",
        },
        "mode": "token",
    },

    "toxicity_type": {
        "mapping": {
            "Fever in neutropenia": "fever_in_neutropenia",
            "Cardio-vascular": "cardiovascular",
            "Renal insufficiency": "renal_insufficiency",
            "Polyneuropathy": "polyneuropathy",
            "Encephalopathy": "encephalopathy",
            "Endocine": "endocrine",
            "Second cancer": "second_cancer",
            "Fertility": "fertility",
            "Other, please specify": "other_specify",
        },
        "mode": "token",
    },

    # CTCAE grade is recorded as "Grade 1-5" (numeric), so you typically don't need a mapping.
    # But if you want to normalize string forms, you can keep this:
    "ctcae_grade": {
        "mapping": {
            "Grade 1": 1,
            "Grade 2": 2,
            "Grade 3": 3,
            "Grade 4": 4,
            "Grade 5": 5,
        },
        "mode": "token",
    },
}



column_renaming_map = {
    # identifiers
    "Patient ID (PID)": "patient_id",

    # demographics / consent
    "Alt PID": "alt_patient_id",
    "Gender": "gender",
    "Year of birth": "year_of_birth",
    "General Consent agreed": "general_consent_agreed",

    # pathology / diagnosis
    "Date of pathology report": "date_pathology_report",
    "type_biopsy_Timo": "biopsy_type",
    "biopsy_neoadjuvant_Timo": "biopsy_neoadjuvant",
    "Grading of biopsy": "biopsy_grading",
    "Grading of resection": "resection_grading",
    "WHO Diagnosis Code": "who_diagnosis_code",
    "If other, please specify": "who_diagnosis_other_specify",

    # tumor / location
    "Anatomic Region Code": "anatomic_region_code",
    "Anatomic region code grouping (calculated)": "anatomic_region_grouping",
    "Anatomic region: Side": "anatomic_region_side",

    # timeline / contact / institutions
    "date_first_patientcontact_Timo": "date_first_patient_contact",
    "Institution": "institution",

    # surgery / whoops
    "Whoops": "whoops",
    "Date of Whoops": "date_whoops",
    "Whoops margin status": "whoops_margin_status",
    "Whoops surgery institution": "whoops_surgery_institution",
    "Date of index surgery": "date_index_surgery",
    "Indication for surgery": "surgery_indication",
    "Institution of surgery": "surgery_institution",
    "Tumor maximal size before surgery": "tumor_max_size_before_surgery",
    "Pathologist's judgement of resected tumor margin": "pathologist_margin_judgement",

    # sizes (resection / imaging / general)
    "Size A [mm] (largest dimension)": "size_a_mm",
    "Size B [mm]": "size_b_mm",
    "Size C [mm]": "size_c_mm",

    # radiology (1. Teil)
    "Date of radiology exam (1. Teil)": "radiology_exam_date",
    "Type of radiology exam (1. Teil)": "radiology_exam_type",
    "Date of initial radiology exam (1. Teil)": "initial_radiology_exam_date",
    "Type of initial radiology exam (1. Teil)": "initial_radiology_exam_type",
    "Radiology exams listed?": "radiology_exams_listed",

    # staging / scores (keep only those present as columns)
    "ctcae_Timo": "ctcae",
    "(newest) Reason for CTCAE_Timo": "ctcae_reason_latest",
    "Kommentar Code_Timo": "comment_code",
    "Kommentar_Timo": "comment",
    "whoops_initial_diagnosis_Timo": "whoops_initial_diagnosis",
    "whoops_initial_diagnosis_code_Timo": "whoops_initial_diagnosis_code",
    "whoops_surgeon_Timo": "whoops_surgeon",
    "whoops_reoperation_margin_r_Timo": "whoops_reoperation_margin_r",

    # sarcoma board / history / follow-up
    "Date of SarcomaBoard": "date_sarcoma_board",
    "Zuordnung SB zu (0), (A), (Z)_Timo": "sarcoma_board_assignment",
    "Reason for (first) SB presentation": "sarcoma_board_reason_first",
    "Follow up SB presentation (Reason for SB presentation)": "sarcoma_board_reason_follow_up",
    "(W) Other diagnoses?_Timo": "other_diagnoses",
    "(newest) Patient history (clinics, therapy) - latest to newest_Timo": "patient_history_latest",
    "Date of last follow-up": "date_last_follow_up",
    "Last status": "last_status",
    "Date of death": "date_death",

    # radiotherapy (top-level block)
    "Start Radiotherapy": "radiotherapy_start_date",
    "End Radiotherapy": "radiotherapy_end_date",
    "Indication for radiotherapy": "radiotherapy_indication",
    "Number of fractions given": "radiotherapy_num_fractions",
    "Type of radiotherapy": "radiotherapy_type",
    "PTV": "radiotherapy_ptv",
    "GTV": "radiotherapy_gtv",
    "Total dose used in radiotherapy": "radiotherapy_total_dose",
    "Hyperthermia status": "hyperthermia_status",
    "Bemerkungen": "notes",

    # radiotherapy (1. Teil block) — renamed to "first_*" to keep it distinct
    "Start Radiotherapy (1. Teil)": "first_radiotherapy_start_date",
    "End Radiotherapy (1. Teil)": "first_radiotherapy_end_date",
    "Indication for radiotherapy (1. Teil)": "first_radiotherapy_indication",
    "Type of radiotherapy (1. Teil)": "first_radiotherapy_type",
    "Number of fractions given (1. Teil)": "first_radiotherapy_num_fractions",
    "Total dose used in radiotherapy (1. Teil)": "first_radiotherapy_total_dose",
    "redonc_first_indication_Timo": "first_redonc_indication",
    "redonc_first_type_Timo": "first_redonc_type",
    "redonc_fractions_gray_Timo": "radiotherapy_fractions_gray",
    "Hyperthermia status & interventionell radiology": "hyperthermia_status_interventional_radiology",

    # systemic therapy (SystTx)
    "Reason for systemic treatment": "systemic_treatment_reason",
    "Line of Treatment": "line_of_treatment",
    "Type of SystTx\n=nicht mappen": "systemic_therapy_type",
    "Name of drug": "drug_name",
    "Softtissue Protocol Name": "soft_tissue_protocol_name",
    "Bone Protocol Name": "bone_protocol_name",
    "Clinical Trial": "clinical_trial",
    "Start date of cycle": "cycle_start_date",
    "End date of cycle": "cycle_end_date",
    "Reason for discontinuation of systemic therapy": "systemic_therapy_discontinuation_reason",
    "Number of cycles executed": "num_cycles_executed",

    # dosing / toxicity
    "Unit": "dose_unit",
    "Dose reduction": "dose_reduction",
    "applied Dose (mg/m2)": "applied_dose_mg_m2",
    "Toxicity which days of cycle": "toxicity_days_of_cycle",
    "Type of Toxicity": "toxicity_type",
    "CTCAE Grade": "ctcae_grade",

    # events (generic)
    "event_type": "event_type",
    "event_date": "event_date",

    # recurrences / metastases
    "Date of 1st local recurrence": "date_local_recurrence_1",
    "Treatment of 1st LR": "treatment_local_recurrence_1",
    "Date of 1st LR surgery": "date_local_recurrence_surgery_1",
    "Date of 2nd LR": "date_local_recurrence_2",
    "Treatment of 2nd LR": "treatment_local_recurrence_2",
    "Date of 2nd LR surgery": "date_local_recurrence_surgery_2",
    "Date of pulmonary metastasis": "date_pulmonary_metastasis",
    "Date of extrapulmonary metastasis": "date_extrapulmonary_metastasis",
    "Site of extrapulmonary metastasis": "site_extrapulmonary_metastasis",
    "Presence of metastasis at diagnosis": "metastasis_present_at_diagnosis",
    "Metastasis at follow-up": "metastasis_at_follow_up",
}




data_type_mapping = {
    # identifiers
    "patient_id": str,
    "alt_patient_id": str,

    # demographics / consent
    "gender": str,
    "year_of_birth": int,
    "general_consent_agreed": int,  # often 0/1; change to bool if you normalize it

    # pathology / diagnosis
    "date_pathology_report": "datetime",
    "biopsy_type": str,
    "biopsy_neoadjuvant": int,  # often 0/1
    "biopsy_grading": str,
    "resection_grading": str,
    "who_diagnosis_code": str,
    "who_diagnosis_other_specify": str,

    # tumor / location
    "anatomic_region_code": str,
    "anatomic_region_grouping": str,
    "anatomic_region_side": str,

    # timeline / institutions
    "date_first_patient_contact": "datetime",
    "institution": str,

    # surgery / whoops
    "whoops": str,  # sometimes yes/no or category; keep str unless you enforce boolean
    "date_whoops": "datetime",
    "whoops_margin_status": str,
    "whoops_surgery_institution": str,
    "date_index_surgery": "datetime",
    "surgery_indication": str,
    "surgery_institution": str,
    "tumor_max_size_before_surgery": float,
    "pathologist_margin_judgement": str,

    # sizes
    "size_a_mm": float,
    "size_b_mm": float,
    "size_c_mm": float,

    # radiology (1. Teil)
    "radiology_exam_date": "datetime",
    "radiology_exam_type": str,
    "initial_radiology_exam_date": "datetime",
    "initial_radiology_exam_type": str,
    "radiology_exams_listed": str,  # could be int/bool depending on encoding

    # ctcae / comments
    "ctcae": str,
    "ctcae_reason_latest": str,
    "ctcae_grade": float,  # sometimes integer; float is safer if you get 2.5 etc
    "comment_code": str,
    "comment": str,
    "whoops_initial_diagnosis": str,
    "whoops_initial_diagnosis_code": str,
    "whoops_surgeon": str,
    "whoops_reoperation_margin_r": str,

    # sarcoma board / history / follow-up
    "date_sarcoma_board": "datetime",
    "sarcoma_board_assignment": str,
    "sarcoma_board_reason_first": str,
    "sarcoma_board_reason_follow_up": str,
    "other_diagnoses": str,
    "patient_history_latest": str,
    "date_last_follow_up": "datetime",
    "last_status": str,
    "date_death": "datetime",

    # radiotherapy (top-level)
    "radiotherapy_start_date": "datetime",
    "radiotherapy_end_date": "datetime",
    "radiotherapy_indication": str,
    "radiotherapy_num_fractions": float,  # some datasets store as numeric with missing/decimals
    "radiotherapy_type": str,
    "radiotherapy_ptv": float,
    "radiotherapy_gtv": float,
    "radiotherapy_total_dose": float,
    "radiotherapy_fractions_gray": float,
    "hyperthermia_status": str,
    "hyperthermia_status_interventional_radiology": str,
    "notes": str,
    "redonc_indication": str,
    "redonc_type": str,

    # systemic therapy (SystTx)
    "systemic_treatment_reason": str,
    "line_of_treatment": float,  # often integer but can be missing -> float safer
    "systemic_therapy_type": str,
    "drug_name": str,
    "soft_tissue_protocol_name": str,
    "bone_protocol_name": str,
    "clinical_trial": str,  # could become bool if normalized
    "cycle_start_date": "datetime",
    "cycle_end_date": "datetime",
    "systemic_therapy_discontinuation_reason": str,
    "num_cycles_executed": float,

    # dosing / toxicity
    "dose_unit": str,
    "dose_reduction": str,          # often yes/no/percent/free-text
    "applied_dose_mg_m2": float,
    "toxicity_days_of_cycle": str,
    "toxicity_type": str,

    # events (generic)
    "event_type": str,
    "event_date": "datetime",

    # recurrences / metastases
    "date_local_recurrence_1": "datetime",
    "treatment_local_recurrence_1": str,
    "date_local_recurrence_surgery_1": "datetime",
    "date_local_recurrence_2": "datetime",
    "treatment_local_recurrence_2": str,
    "date_local_recurrence_surgery_2": "datetime",
    "date_pulmonary_metastasis": "datetime",
    "date_extrapulmonary_metastasis": "datetime",
    "site_extrapulmonary_metastasis": str,
    "metastasis_present_at_diagnosis": str,  # make int/bool if normalized
    "metastasis_at_follow_up": str,
}

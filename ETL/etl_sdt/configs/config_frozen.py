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
            # ========== AGGREGATED CATEGORIES (15 main groups) ==========
            
            # Legacy numbered codes (backward compatibility)
            "[0] Angiosarcoma of soft tissues": "vascular_sarcoma",
            "[1] Atypical chondromatous tumor": "chondrosarcoma",
            "[2] Atypical lipomatous tumor / well differentiated liposarcoma": "liposarcoma",
            "[3] Chondrosarcoma": "chondrosarcoma",
            "[4] Conventional osteosarcoma": "osteosarcoma",
            "[5] Dedifferentiated liposarcoma": "liposarcoma",
            "[6] Desmoid‐type fibromatosis": "fibroblastic_tumor",
            "[7] Ewing sarcoma": "ewing_sarcoma",
            "[8] Giant cell tumor of bone": "giant_cell_tumor",
            "[9] GIST": "gist",
            "[10] Intramuscular myxoma (incl. variants)": "other_soft_tissue_sarcoma",
            "[11] Leiomyosarcoma (excluding skin)": "leiomyosarcoma",
            "[12] Myxofibrosarcoma": "fibroblastic_tumor",
            "[13] Myxoid liposarcoma": "liposarcoma",
            "[14] Pleomorphic liposarcoma": "liposarcoma",
            "[15] Solitary fibrous tumor": "fibroblastic_tumor",
            "[16] Synovial sarcoma": "other_soft_tissue_sarcoma",
            "[17] Tenosynovial giant cell tumor": "giant_cell_tumor",
            "[18] Undifferentiated / unclassified sarcoma": "undifferentiated_sarcoma",
            
            # Non-neoplastic / benign conditions
            "0 non-neoplastic / tumor simulator": "benign_non_neoplastic",
            "non-neoplastic / tumor simulator": "benign_non_neoplastic",
            "99 not yet established": "unknown",
            " ": "unknown",
            
            # ========== 1. LIPOSARCOMA (all adipocytic malignancies) ==========
            "2.2.1. atypical lipomatous tumor / well differentiated liposarcoma": "liposarcoma",
            "2.3.1. dedifferentiated liposarcoma": "liposarcoma",
            "2.3.2. myxoid liposarcoma": "liposarcoma",
            "2.3.3. pleomorphic liposarcoma": "liposarcoma",
            
            # ========== 2. BENIGN ADIPOCYTIC (lipomas) ==========
            "2.1.1. lipoma": "benign_adipocytic",
            "2.1.2. lipomatosis": "benign_adipocytic",
            "2.1.4. lipoblastoma / lipoblastomatosis": "benign_adipocytic",
            "2.1.5. angiolipoma": "benign_adipocytic",
            "2.1.8. extra-renal angiomyolipoma": "benign_adipocytic",
            "2.1.9. extra‐adrenal myelolipoma": "benign_adipocytic",
            "2.1.10. spindle cell / pleomorphic lipoma": "benign_adipocytic",
            "2.1.11. hibernoma": "benign_adipocytic",
            "1.1.10. Atypical spindle cell / pleomorphic lipomatous tumour": "benign_adipocytic",
            "1.1.10. Atypical spindle cell / pleomorphic lipomatous tumour ": "benign_adipocytic",
            
            # ========== 3. FIBROBLASTIC TUMOR (all fibro/myofibroblastic) ==========
            # Benign fibroblastic
            "3.1.1. nodular fasciitis": "fibroblastic_tumor",
            "3.1.4. myositis ossificans": "fibroblastic_tumor",
            "3.1.5. fibro‐osseous pseudotumor of digits": "fibroblastic_tumor",
            "3.1.7. elastofibroma": "fibroblastic_tumor",
            "3.1.12. fibroma of tendon sheath": "fibroblastic_tumor",
            "3.1.13. desmoplastic fibroblastoma": "fibroblastic_tumor",
            "3.1.17. cellular angiofibroma": "fibroblastic_tumor",
            "3.2.1. palmar/plantar fibromatosis": "fibroblastic_tumor",
            # Intermediate fibroblastic
            "3.2.2. desmoid‐type fibromatosis": "fibroblastic_tumor",
            "3.2.3. lipofibromatosis": "fibroblastic_tumor",
            "3.2.5. dermatofibrosarcoma protuberans": "fibroblastic_tumor",
            "3.2.6. fibrosarcomatous dermatofibrosarcoma protuberans": "fibroblastic_tumor",
            "3.2.7. pigmented dermatofibrosarcoma protuberans": "fibroblastic_tumor",
            "3.2.8. solitary fibrous tumor": "fibroblastic_tumor",
            "3.2.9. solitary fibrous tumor, malignant": "fibroblastic_tumor",
            "3.2.10. inflammatory myofibroblastic tumor": "fibroblastic_tumor",
            "3.2.11. low‐grade myofibroblastic sarcoma": "fibroblastic_tumor",
            "3.2.12. myxoinflammatory fibroblastic sarcoma": "fibroblastic_tumor",
            # Malignant fibroblastic
            "3.3.2. myxofibrosarcoma": "fibroblastic_tumor",
            "3.3.3. low‐grade fibromyxoid sarcoma (Evans tumor)": "fibroblastic_tumor",
            "3.3.4. sclerosing epithelioid fibrosarcoma": "fibroblastic_tumor",
            "17.2.1. desmoplastic fibroma of bone": "fibroblastic_tumor",
            "18.1.1. benign fibrous histiocytoma": "fibroblastic_tumor",
            "18.1.2. non‐ossifying fibroma": "fibroblastic_tumor",
            
            # ========== 4. GIANT CELL TUMOR (bone and soft tissue) ==========
            "4.1.1. deep benign fibrous histiocytoma": "giant_cell_tumor",
            "4.2.1. tenosynovial giant cell tumor": "giant_cell_tumor",
            "4.2.2. plexiform fibrohistiocytic tumor": "giant_cell_tumor",
            "4.2.3. giant cell tumor of soft tissues": "giant_cell_tumor",
            "21.2.1. giant cell tumor of bone": "giant_cell_tumor",
            "Keratin positive giant cell tumor / xanthogranulomatous epithelial tumor": "giant_cell_tumor",
            "Xanthogranulomatous epithelial tumor (XGET)": "giant_cell_tumor",
            
            # ========== 5. LEIOMYOSARCOMA (smooth muscle) ==========
            "5.1.1. deep leiomyoma": "leiomyosarcoma",
            "5.3.1. leiomyosarcoma (excluding skin)": "leiomyosarcoma",
            "atypical intradermal smooth muscle tumor": "leiomyosarcoma",
            "smooth muscle tumor of uncertain malignant potential (STUMP)": "leiomyosarcoma",
            "24.3.1. leiomyosarcoma of bone": "leiomyosarcoma",
            "Low-Grade Endometrial Stromal Sarcoma": "leiomyosarcoma",
            "Low-grade endometrial stromal sarcoma": "leiomyosarcoma",
            "prostatic stromal sarcoma": "leiomyosarcoma",
            
            # ========== 6. PERICYTIC (glomus, myopericytoma, angioleiomyoma) ==========
            "6.1.1. angioleiomyoma": "pericytic_tumor",
            "6.2.1. glomus tumor and variants": "pericytic_tumor",
            "6.2.2. myopericytoma": "pericytic_tumor",
            
            # ========== 7. RHABDOMYOSARCOMA (skeletal muscle) ==========
            "7.3.1. embryonal rhabdomyosarcoma": "rhabdomyosarcoma",
            "7.3.2. alveolar rhabdomyosarcoma": "rhabdomyosarcoma",
            "7.3.3. pleomorphic rhabdomyosarcoma": "rhabdomyosarcoma",
            "7.3.4. spindle cell / sclerosing rhabdomyosarcoma": "rhabdomyosarcoma",
            "7.3.4. Spindle cell / sclerosing rhabdomyosarcoma": "rhabdomyosarcoma",
            "partiell myogen differentiated spindle cell sarcoma": "rhabdomyosarcoma",
            
            # ========== 8. VASCULAR SARCOMA (angiosarcoma, hemangioendothelioma) ==========
            # Benign vascular
            "8.1.1. hemangioma": "benign_vascular",
            "8.1.3. angiomatosis": "benign_vascular",
            "8.1.4. lymphangioma": "benign_vascular",
            "23.1.1. hemangioma": "benign_vascular",
            "23.2.1. epithelioid hemangioma": "benign_vascular",
            # Intermediate/malignant vascular
            "8.2.4. composite hemangioendothelioma": "vascular_sarcoma",
            "8.2.5. pseudomyogenic (epithelioid sarcoma‐like) hemangioendothelioma": "vascular_sarcoma",
            "8.2.6. kaposi sarcoma": "vascular_sarcoma",
            "8.3.1. epithelioid hemangioendothelioma": "vascular_sarcoma",
            "8.3.2. angiosarcoma of soft tissues": "vascular_sarcoma",
            "23.3.1. epithelioid hemangioendothelioma": "vascular_sarcoma",
            "23.3.2. angiosarcoma": "vascular_sarcoma",
            
            # ========== 9. CHONDROSARCOMA (all cartilage malignancies) ==========
            # Benign cartilage
            "9.1.1. soft tissue chondroma": "benign_cartilage",
            "15.1.1. osteochondroma": "benign_cartilage",
            "15.1.1. osteochondroma: Diagnose nicht nachvollziehbar, ist wohl Fehler im Adjumed. Pat hat Hypopharynxkarzinom": "benign_cartilage",
            "15.1.2. chondroma": "benign_cartilage",
            "15.1.2. chondroma / echondroma": "benign_cartilage",
            "15.1.4. subungual exostosis": "benign_cartilage",
            "15.1.5. bizarre parosteal osteochondromatous proliferation": "benign_cartilage",
            "Bizarre parosteal osteochondromatous proliferations (BPOP)": "benign_cartilage",
            "15.1.6. synovial chondromatosis": "benign_cartilage",
            "15.2.1. chondromyxoid fibroma": "benign_cartilage",
            "15.2.3. chondroblastoma": "benign_cartilage",
            "27.0.7. multiple osteochondromas": "benign_cartilage",
            # Malignant cartilage
            "15.2.2. atypical chondromatous tumors / chondrosarcoma grade I": "chondrosarcoma",
            "15.3.1. chondrosarcoma grade II, grade III": "chondrosarcoma",
            "15.3.2. dedifferentiated chondrosarcoma": "chondrosarcoma",
            "15.3.2 dedifferentiated chondrosarcoma": "chondrosarcoma",
            "15.3.3. mesenchymal chondrosarcoma": "chondrosarcoma",
            "15.3.4. clear cell chondrosarcoma": "chondrosarcoma",
            "9.3.1. extraskeletal mesenchymal chondrosarcoma": "chondrosarcoma",
            
            # ========== 10. OSTEOSARCOMA (bone forming) ==========
            # Benign bone forming
            "16.1.1. osteoma": "benign_bone",
            "16.1.2. osteoid osteoma": "benign_bone",
            "16.2.1. osteoblastoma": "benign_bone",
            # Malignant bone forming
            "16.3.1. conventional osteosarcoma": "osteosarcoma",
            "16.3.2. conventional osteosarcoma": "osteosarcoma",
            "16.3.5. secondary osteosarcoma": "osteosarcoma",
            "16.3.6. parosteal osteosarcoma": "osteosarcoma",
            "16.3.7. periosteal osteosarcoma": "osteosarcoma",
            "9.3.2. extraskeletal osteosarcoma": "osteosarcoma",
            
            # ========== 11. EWING SARCOMA ==========
            "19.3.1. ewing sarcoma": "ewing_sarcoma",
            "12.3.6. extraskeletal ewing sarcoma": "ewing_sarcoma",
            
            # ========== 12. GIST ==========
            "10.3.3. gist": "gist",
            
            # ========== 13. NERVE SHEATH TUMOR ==========
            # Benign nerve sheath
            "11.1.1. schwannoma (including variants)": "benign_nerve_sheath",
            "11.1.3. neurofibroma (incl. variants)": "benign_nerve_sheath",
            "11.1.4. perineurioma": "benign_nerve_sheath",
            "11.1.5. granular cell tumor": "benign_nerve_sheath",
            "11.1.7. solitary circumscribed neuroma": "benign_nerve_sheath",
            "11.1.11. hybrid nerve sheath tumors": "benign_nerve_sheath",
            # Malignant nerve sheath
            "11.3.1. malignant peripheral nerve sheath tumor": "malignant_nerve_sheath",
            "11.3.1. Ntrk-Rearranged spindle cell neoplasm": "malignant_nerve_sheath",
            
            # ========== 14. UNDIFFERENTIATED SARCOMA ==========
            "13.2.1. dermal undifferentiated sarcoma": "undifferentiated_sarcoma",
            "13.3.1. undifferentiated / unclassified sarcoma": "undifferentiated_sarcoma",
            "26.3.1. undifferentiated high grade pleomorphic sarcoma of bone": "undifferentiated_sarcoma",
            "undifferentiated SMARCA4-deficient tumor": "undifferentiated_sarcoma",
            
            # ========== 15. OTHER SOFT TISSUE SARCOMA (diverse group) ==========
            "12.1.1. acral fibromyxoma": "other_soft_tissue_sarcoma",
            "12.1.2. intramuscular myxoma (incl. variants)": "other_soft_tissue_sarcoma",
            "12.1.3. juxta‐articular myxoma": "other_soft_tissue_sarcoma",
            "12.1.4. deep (\"aggressive\") angiomyxoma": "other_soft_tissue_sarcoma",
            "12.1.5. pleomorphic hyalinizing angietctaic tumor": "other_soft_tissue_sarcoma",
            "12.2.1. hemosiderotic fibrolipomatous tumor": "other_soft_tissue_sarcoma",
            "12.2.3. angiomatoid fibrous histiocytoma": "other_soft_tissue_sarcoma",
            "12.2.4. ossifying fibromyxoid tumor": "other_soft_tissue_sarcoma",
            "12.2.4 ossifying fibromyxoid tumor": "other_soft_tissue_sarcoma",
            "12.2.5. mixed tumor NOS": "other_soft_tissue_sarcoma",
            "12.2.6. myoepithelioma": "other_soft_tissue_sarcoma",
            "12.2.7. phosphaturic mesenchymal tumor": "other_soft_tissue_sarcoma",
            "12.3.1. synovial sarcoma": "other_soft_tissue_sarcoma",
            "12.3.2. epithelioid sarcoma": "other_soft_tissue_sarcoma",
            "12.3.3. alveolar soft‐part sarcoma": "other_soft_tissue_sarcoma",
            "12.3.3. CIC-rearranged sarcoma": "other_soft_tissue_sarcoma",
            "12.3.4. clear cell sarcoma of soft tissue": "other_soft_tissue_sarcoma",
            "12.3.5. extraskeletal myxoid chondrosarcoma": "other_soft_tissue_sarcoma",
            "12.3.7. desmoplastic small round cell tumor": "other_soft_tissue_sarcoma",
            "12.3.9. neoplasms with perivascular epithelioid differentiation (PEComa)": "other_soft_tissue_sarcoma",
            "12.3.10. intimal sarcoma": "other_soft_tissue_sarcoma",
            "22.3.1. chordoma": "other_soft_tissue_sarcoma",
            
            # ========== 16. BENIGN BONE LESIONS ==========
            "25.1.1. simple bone cyst": "benign_bone_lesion",
            "25.1.2. fibrous dysplasia": "benign_bone_lesion",
            "25.2.1. aneurysmal bone cyst": "benign_bone_lesion",
            "25.2.1. aneurysmal bone cyst bzw. undefinierte Entität": "benign_bone_lesion",
            "25.2.2. Langerhans cell histiocytosis": "benign_bone_lesion",
            "25.2.2. langerhans cell histiocytosis": "benign_bone_lesion",
            
            # ========== 17. METASTASIS (all M.x codes) ==========
            "M.1. lung": "metastasis",
            "M.1. lung ": "metastasis",
            "M.2. breast": "metastasis",
            "M.2. breast ": "metastasis",
            "M.3. prostate": "metastasis",
            "M.3. prostate ": "metastasis",
            "M.4. kidney": "metastasis",
            "M.4. kidney ": "metastasis",
            "M.5. thyroid": "metastasis",
            "M.5. thyroid ": "metastasis",
            "M.6. melanoma": "metastasis",
            "M.6. melanoma ": "metastasis",
            "M.7. colon / lower GI": "metastasis",
            "M.9. hepato-biliary-pancreas": "metastasis",
            "M.9. hepato-biliary-pancreas ": "metastasis",
            "M.10. cervix / endometrium": "metastasis",
            "M.12. CUP": "metastasis",
            "M.13. other metastasis (please specify in next field)": "metastasis",
            
            # ========== 18. HEMATOLOGIC MALIGNANCIES ==========
            "BL.0 lymphoma": "hematologic_malignancy",
            "BL.1 myeloma": "hematologic_malignancy",
            "BL.2 eukemia": "hematologic_malignancy",
            
            # ========== 19. OTHER CARCINOMAS (non-sarcoma malignancies) ==========
            "NET G1 (GI-Tumor)": "other_carcinoma",
            "Sarcomatoid carcinoma": "other_carcinoma",
            "Spinaliom": "other_carcinoma",
            "Vulvar carcinoma": "other_carcinoma",
            "Karzinosarkom": "other_carcinoma",
            "Adenosarcoma": "other_carcinoma",
            "Sarcoid lung carcinoma": "other_carcinoma",
            "Sarkomatoides Pleuramesotheliom": "other_carcinoma",
            "Wenig differenziertes kleinzelliges neuroendokrines Karzinom des Beckens, a.e. entartete Tailgut-Zyst": "other_carcinoma",
            
            # ========== 20. BENIGN / NON-NEOPLASTIC ==========
            "Pseudogicht": "benign_non_neoplastic",
            "Ganglion": "benign_non_neoplastic",
            "Ganglion-like structure": "benign_non_neoplastic",
            "Epidermoidcyst": "benign_non_neoplastic",
            "Dermoidcyst": "benign_non_neoplastic",
            "Atheroma": "benign_non_neoplastic",
            "Endometriom": "benign_non_neoplastic",
            "Fettgewebsnekrose": "benign_non_neoplastic",
            "Tuberkulose": "benign_non_neoplastic",
            "TBC": "benign_non_neoplastic",
            "Tumor calcinosis": "benign_non_neoplastic",
            "Phylloidestumor": "benign_non_neoplastic",
            "Fibroadenoma": "benign_non_neoplastic",
            "Fibrosiertes Weichteilgewebeexzisat": "benign_non_neoplastic",
            "sklerotische ossäre Läsionen": "benign_non_neoplastic",
            "Intraossäres Meningeom vom fibrösen Typ": "benign_non_neoplastic",
            "Ganglioneuroma": "benign_non_neoplastic",
            "Neuroblastoma": "benign_non_neoplastic",
            "Lymphadenopathie": "benign_non_neoplastic",
            "inflammation": "benign_non_neoplastic",
        },
        "mode": "code",
    },

    "anatomic_region_code": {
        "mapping": {
        # ---------- B (bones / skeletal) ----------
        "B1 face, head, neck": "head_neck",
        "B2 clavicle": "clavicle",
        "B3.1.A partial intraarticular scapula": "scapula",
        "B3.1.B partial extraarticular scapula": "scapula",
        "B3.2.A complete intraarticular scapula": "scapula",
        "B3.2.B complete extraarticular scapula": "scapula",
        "B3.3 acromion (from base of scapular spine to AC-joint)": "scapula",

        "B4.1 intraarticular proximal humerus": "humerus",
        "B4.2 extraarticular proximal humerus": "humerus",
        "B4.3 diaphyseal humerus": "humerus",
        "B4.4 intraarticular distal humerus": "humerus",
        "B4.5 extraarticular distal humerus": "humerus",
        "B4.6 intraarticular humerus combinations": "humerus",
        "B4.7 extraarticular humerus combinations": "humerus",

        "B5.2 extraarticular elbow joint": "elbow",

        "B6.1 intraarticular radius proximal": "forearm_bones",
        "B6.2 extraarticular radius proximal": "forearm_bones",
        "B6.4 intraarticular radius distal": "forearm_bones",
        "B6.5 extraarticular radius distal": "forearm_bones",
        "B6.9 extraarticular ulna proximal": "forearm_bones",
        "B6.10 ulna diaphyseal": "forearm_bones",
        "B6.11 intraarticular ulna distal": "forearm_bones",
        "B6.12 extraarticular ulna distal": "forearm_bones",

        "B8.2 metacarpal": "hand",
        "B8.3 digits": "hand",

        "B9.1 cervical spine": "spine_torso",
        "B9.2 thoracic spine": "spine_torso",
        "B9.3 lumbar spine": "spine_torso",
        "B9.5 ribs": "spine_torso",
        "B9.6 torso combinations": "spine_torso",
        "B9.7 sternum": "spine_torso",

        "B10.1 ilium": "pelvis",
        "B10.2 ischium": "pelvis",
        "B10.3 pubis": "pelvis",
        "B10.4 pelvis combinations": "pelvis",
        "B10.5 acetabulum / hip joint: intraarticular": "pelvis",
        "B10.6 acetabulum / hip joint: extraarticular": "pelvis",
        "B10.7 sacrum": "pelvis",

        "B12.1 intraarticular proximal femur": "femur",
        "B12.3 diaphyseal femur": "femur",
        "B12.4 intraarticular distal femur": "femur",
        "B12.5 extraarticular distal femur": "femur",
        "B12.7 extraarticular femur combinations": "femur",

        "B13.1 patella intraarticular": "patella",
        "B13.2 patella extraarticular": "patella",
        "B13.3 patella combinations": "patella",

        "B14.1 intraarticular tibia proximal": "tibia_fibula",
        "B14.2 extraarticular tibia proximal": "tibia_fibula",
        "B14.3 tibia diaphyseal": "tibia_fibula",
        "B14.4 intraarticular tibia distal": "tibia_fibula",
        "B14.5 extraarticular tibia distal": "tibia_fibula",
        "B14.6 tibia combinations": "tibia_fibula",
        "B14.7 intraarticular fibula proximal": "tibia_fibula",
        "B14.8 extraarticular fibula proximal": "tibia_fibula",
        "B14.9 fibula diaphyseal": "tibia_fibula",
        "B14.11 extraarticular fibula distal": "tibia_fibula",

        "B15.2 extraarticular ankle joint": "ankle",

        "B16.1 hindfoot": "foot",
        "B16.2 forefoot": "foot",
        "B16.3 digits": "foot",
        "B16.4 foot combinations": "foot",

        # ---------- C / misc ----------
        "C6-Th1": "spine_torso",
        "Orbita": "orbit",
        "unknown": "unknown",

        # ---------- D (soft tissue / compartments / joints / viscera) ----------
        "D1 face": "face",
        "D2 head": "head_neck",
        "D2.1 clavicle": "clavicle",
        "D2.2 brachial plexus": "head_neck",

        "D3 anterior neck": "neck",
        "D3.1 deltoid region": "shoulder_girdle",
        "D3.2 superior shoulder": "shoulder_girdle",
        "D3.3 dorsal to scapula": "scapular_region",
        "D3.4 inferior to scapula": "scapular_region",
        "D3.5 axilla (base / lateral)": "axilla",
        "D3.6 axilla-scapulo-thoracic-chest wall (medial / deep)": "chest_wall",

        "D4 posterior neck": "neck",
        "D4.1.1 anterior / flexor compartment: proximal": "upper_arm",
        "D4.1.2 anterior / flexor compartment: diaphyseal": "upper_arm",
        "D4.1.3 anterior / flexor compartment: distal": "upper_arm",
        "D4.1.4 anterior / flexor compartment: combinations": "upper_arm",
        "D4.2.1 posterior / extensor compartment: proximal": "upper_arm",
        "D4.2.2 posterior / extensor compartment: diaphyseal": "upper_arm",
        "D4.2.3 posterior / extensor compartment: distal": "upper_arm",
        "D4.2.4 posterior / extensor compartment: combinations": "upper_arm",
        "D4.3.1 compartment combinations: proximal": "upper_arm",
        "D4.3.2 compartment combinations: diaphyseal": "upper_arm",
        "D4.3.3 compartment combinations: distal": "upper_arm",
        "D4.3.4 compartment combinations": "upper_arm",

        "D5.1 intraarticular elbow joint": "elbow",
        "D5.2 extraarticular elbow joint": "elbow",

        "D6.1.1 anterior / flexor compartment: proximal": "forearm",
        "D6.1.2 anterior / flexor compartment: diaphyseal": "forearm",
        "D6.1.3 anterior / flexor compartment: distal": "forearm",
        "D6.1.4 anterior / flexor compartment: combinations": "forearm",
        "D6.2.1 posterior / extensor compartment: proximal": "forearm",
        "D6.2.2 posterior / extensor compartment: diaphyseal": "forearm",
        "D6.2.4 posterior / extensor compartment: combinations": "forearm",
        "D6.3.1 compartment combinations: proximal": "forearm",
        "D6.3.2 compartment combinations: diaphyseal": "forearm",
        "D6.3.3 compartment combinations: distal": "forearm",
        "D6.3.4 compartment combinations": "forearm",

        "D7.1 intraarticular wrist joint": "wrist",
        "D7.2 extraarticular wrist joint": "wrist",

        "D8.1 dorsum manus": "hand",
        "D8.2 palma manus": "hand",
        "D8.3 hand combinations": "hand",
        "D8.4 fingers": "hand",

        "D9.1 cervical spine": "spine_torso",
        "D9.2 thoracic spine / para-scapular region": "spine_torso",
        "D9.3 lumbar spine / lumbar region": "spine_torso",
        "D9.5 combinations (incl. ilium)": "spine_torso",

        "D10.1 iliacus–psoas": "pelvis_intrapelvic",
        "D10.1.1 uterus": "pelvis_intrapelvic",
        "D10.2 intrapelvic (other)": "pelvis_intrapelvic",
        "D10.2.2 pararectal": "pelvis_intrapelvic",
        "D10.3 abductors / gluteal muscles": "gluteal_region",
        "D10.4 extrapelvic, other": "pelvis_extrapelvic",
        "D10.5 uterus": "pelvis_intrapelvic",
        "D10.6.1 anterior through inguina": "pelvis_extrapelvic",
        "D10.6.2 lateral through sciatic notch": "pelvis_extrapelvic",
        "D10.6.3 both intra- and extrapelvic, other": "pelvis_extrapelvic",
        "D10.6.4 anterior through foramen obturatum": "pelvis_extrapelvic",

        "D11.1 inguinal soft tissues": "inguinal_region",

        "D12.5.1 medial / adductor compartment: proximal": "thigh",
        "D12.5.2 medial / adductor compartment: diaphyseal": "thigh",
        "D12.5.3 medial / adductor compartment: distal": "thigh",
        "D12.5.4 medial / adductor compartment: combinations": "thigh",
        "D12.6.1 anterior / extensor compartment: proximal": "thigh",
        "D12.6.2 posterior / extensor compartment: diaphyseal": "thigh",
        "D12.6.3 posterior / extensor compartment: distal": "thigh",
        "D12.6.4 anterior / extensor compartment: combinations": "thigh",
        "D12.7.1 posterior / flexor compartment: proximal": "thigh",
        "D12.7.2 posterior / flexor compartment: diaphyseal": "thigh",
        "D12.7.3 posterior / flexor compartment: distal": "thigh",
        "D12.7.4 anterior / flexor compartment: combinations": "thigh",
        "D12.8.1 compartment combinations: anterior – posterior": "thigh",
        "D12.8.2 compartment combinations: anterior – medial": "thigh",
        "D12.8.3 compartment combinations: medial – posterior": "thigh",
        "D12.8.4 compartment combinations: anterior – medial – posterior": "thigh",

        "D13.1 periarticular, intraarticular knee joint": "knee",
        "D13.2 periarticular, extraarticular knee joint": "knee",
        "D13.3 popliteal, extraarticular knee joint": "knee",
        "D13.4 popliteal, extraarticular knee joint": "knee",

        "D14.1.1 lateral / fibular compartment: proximal": "lower_leg",
        "D14.1.2 lateral / fibular compartment: diaphyseal": "lower_leg",
        "D14.1.3 lateral / fibular compartment: distal": "lower_leg",
        "D14.1.4 lateral / fibular compartment: combinations": "lower_leg",
        "D14.2.1 anterior / extensor compartment: proximal": "lower_leg",
        "D14.2.2 anterior / extensor compartment: diaphyseal": "lower_leg",
        "D14.2.3 anterior / extensor compartment: distal": "lower_leg",
        "D14.2.4 anterior / extensor compartment: combinations": "lower_leg",
        "D14.3.1 posterior / flexor compartment: proximal": "lower_leg",
        "D14.3.2 posterior / flexor compartment: diaphyseal": "lower_leg",
        "D14.3.3 posterior / flexor compartment: distal": "lower_leg",
        "D14.3.4 posterior / flexor compartment: combinations": "lower_leg",
        "D14.4.1 compartment combinations: anterior – posterior": "lower_leg",

        "D15.1 intraarticular ankle joint": "ankle",
        "D15.2 extraarticular ankle joint": "ankle",

        "D16.1 dorsum pedis": "foot",
        "D16.2 planta pedis": "foot",
        "D16.3 foot combinations": "foot",
        "D16.4 digits": "foot",

        "D17.1 intraarticular shoulder joint": "shoulder_joint",
        "D17.2 extraarticular shoulder joint": "shoulder_joint",

        "D18.1 abdominal wall: anterior": "abdominal_wall",
        "D18.2 abdominal wall: lateral": "abdominal_wall",
        "D18.3 intraperitoneal": "abdomen_peritoneum",
        "D18.4 retro-/extraperitoneal": "retroperitoneum",

        "D19.1 chest: sternal": "thorax",
        "D19.2 chest: pectoral": "thorax",
        "D19.3 chest: lateral": "thorax",
        "D19.3.1 chest: posterior": "thorax",
        "D19.4 intrathoracic": "intrathoracic",
        "D19.4.1 lung": "intrathoracic",
        "D19.4.2 pleura": "intrathoracic",
        "D19.4.3 mediastinum": "intrathoracic",
        "D19.4.4 heart": "intrathoracic",
        "D19.4.5 intrathoracic, other": "intrathoracic",

        "D20.1 intraarticular hip joint": "hip_joint",
        "D20.2 extraarticular hip joint": "hip_joint",

        # ---------- S (surface regions) ----------
        "S1 face": "face",
        "S2 head": "head_neck",

        "S3 anterior neck": "neck",
        "S3.1 regio suprascapularis": "scapular_region",
        "S3.2 regio scapularis": "scapular_region",
        "S3.3 regio infrascapularis": "scapular_region",
        "S3.4 regio deltoidea": "shoulder_girdle",
        "S3.5 regio axillaris": "axilla",

        "S4 posterior neck": "neck",
        "S4.1 regio brachialis anterior": "upper_arm",
        "S4.2 regio brachialis posterior": "upper_arm",
        "S4.3 combinations of regio brachialis anterior and posterior": "upper_arm",

        "S5.1 regio cubitalis anterior": "elbow_region",
        "S5.2 regio cubitalis posterior": "elbow_region",

        "S6.1 regio antebrachii anterior": "forearm",
        "S6.2 regio antebrachii posterior": "forearm",
        "S6.3 combinations of regio antebrachii anterior and posterior": "forearm",

        "S7 clavicle region": "clavicle",

        "S8.1 dorsum manus": "hand",
        "S8.3 digits": "hand",

        "S9.1 regio vertebralis": "spine_torso",
        "S9.2 regio lumbalis": "spine_torso",

        "S10.1 regio glutealis": "gluteal_region",
        "S10.2 regio pubica": "pelvis_extrapelvic",
        "S10.3 regio urogenitalis / perinealis / analis": "pelvis_extrapelvic",

        "S11.1 regio inguinalis": "inguinal_region",

        "S12.1 regio femoris anterior proximal": "thigh",
        "S12.2 regio femoris posterior proximal": "thigh",
        "S12.3 regio femoris anterior diaphyseal": "thigh",
        "S12.4 regio femoris posterior diaphyseal": "thigh",
        "S12.5 regio femoris anterior distal": "thigh",
        "S12.6 regio femoris posterior distal": "thigh",
        "S12.7 distal femur combinations": "thigh",

        "S13.1 regio genus anterior": "knee",
        "S13.2 regio genus posterior": "knee",

        "S14.1 regio cruralis anterior": "lower_leg",
        "S14.2 regio cruralis posterior": "lower_leg",

        "S15.1 regio calcanea": "foot",

        "S16.1 dorsum pedis": "foot",
        "S16.2 planta pedis": "foot",
        "S16.3 digits": "foot",
        "S16.4 foot combinations": "foot",

        "S17.1 regio hypochondrica": "abdomen",
        "S17.2 regio umbilicalis": "abdomen",
        "S17.3 regio abdominis lateralis": "abdomen",

        "S18.1 regio presternalis": "thorax",
        "S18.2 regio pectoralis": "thorax",
    },
        "mode": "code",
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

    "pathologist_margin_judgement": {
        "mapping": {
            "R0": "R0",
            "R1": "R1",
            "R2": "R2",
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
            "NED": "NED",
            "AWD": "AWD",
            "DOD": "DOD",
            "DOR": "DOR",
        },
        "mode": "substring",
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
        "mode": "substring",
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

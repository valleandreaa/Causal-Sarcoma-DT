import os
import json
from pathlib import Path
from typing import Dict, Any

import torch
from Models.models.GANs import LSTMGenerator
from Models.models.decoder import Decoder
from Models.models.utils import MetadataHandler
from Models.utils.helpers import MongoExtractor
from Inference.refined_evidence import filter_ajcc_stage_iii
import argparse
from Inference.refined_evidence_package import EvidencePackageGenerator, EvidenceComponents

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Temporal inference high-risk STS (Research Question 1)",
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="saved_models/temporal_cycle_gan_20250731_182445/metadata.json",
        help="Path to the metadata file for the temporal cycle GAN.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="Models/configs/temporal_inference.yaml",
        help="Path to the Mongo configuration file.",
    )
    parser.add_argument(
        "--print_probabilities",
        action="store_true",
        help="Print variable probabilities before decoding labels.",
    )
    return parser.parse_args()


def load_temporal_cycle_gan(meta_path: Path, device: torch.device):
    """Load Gx generator from saved metadata."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    params = meta.get("model_params", {})
    embedding_dim = params.get("embedding_dim", 192)
    hidden_dim = params.get("hidden_dim", 128)
    num_layers = params.get("num_layers", 1)

    gx = LSTMGenerator(
        input_dim=embedding_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)
    gx.load_state_dict(torch.load(meta["Gx_path"], map_location=device))
    gx.eval()
    return gx, meta


def initialize_models(meta_file: Path, device: torch.device):
    """Load generator, decoder, and metadata handlers."""
    gx, meta = load_temporal_cycle_gan(meta_file, device)
    meta_clin = MetadataHandler(meta["metadata_clinical_file"])
    meta_treat = MetadataHandler(meta["metadata_treatment_file"])
    decoder_treat, feat_spec, oh_map, ord_map, num_ranges = load_decoder(
        meta["metadata_treatment_file"], meta["decoder_treatment"], device
    )
    return (
        gx,
        meta_clin,
        meta_treat,
        decoder_treat,
        feat_spec,
        oh_map,
        ord_map,
        num_ranges,
    )


def load_decoder(meta_file: str, decoder_path: str, device: torch.device):
    """Load decoder network using parameters stored in metadata."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta.get("model_params", {})
    hidden_dim = params.get("hidden_dim")
    output_dim = params.get("input_dim")
    dec = Decoder(hidden_dim, output_dim).to(device)
    dec.load_state_dict(torch.load(decoder_path, map_location=device))
    dec.eval()
    feature_spec = meta.get("feature_spec")
    one_hot_mappings = meta.get("one_hot_mappings", {})
    ord_mappings = meta.get("ord_mappings", {})
    numeric_ranges = meta.get("numeric_ranges", {})
    return dec, feature_spec, one_hot_mappings, ord_mappings, numeric_ranges


def create_extractor(config_file: str) -> MongoExtractor:
    """Initialise MongoExtractor using environment variables."""
    return MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=config_file,
    )


def run_inference(args: argparse.Namespace) -> None:
    """Run the high-risk inference pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        gx,
        meta_clin,
        meta_treat,
        decoder_treat,
        feat_spec,
        oh_map,
        ord_map,
        num_ranges,
    ) = initialize_models(Path(args.meta_file), device)

    extractor = create_extractor(args.config_file)

    df = extractor.get_dataframe()
    stage_iii_df = filter_ajcc_stage_iii(df)
    if stage_iii_df.empty:
        print("No AJCC Stage III patients found")
        return

    components = EvidenceComponents(
        extractor=extractor,
        meta_clin=meta_clin,
        meta_treat=meta_treat,
        gx=gx,
        decoder_treat=decoder_treat,
        feat_spec=feat_spec,
        oh_map=oh_map,
        ord_map=ord_map,
        num_ranges=num_ranges,
        device=device,
        print_probabilities=args.print_probabilities,
    )
    generator = EvidencePackageGenerator(stage_iii_df, components)
    results = generator.run()
    for k, v in results.items():
        print(f"{k}: {v}")


def main() -> None:
    args = parse_arguments()
    run_inference(args)


if __name__ == "__main__":
    main()
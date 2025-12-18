from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import torch

from Models.dataset.tabular_dataset import TabularDatasetPID
from Models.models.decoder_embedders import decode_embedding
from torch.utils.data import DataLoader

from .refined_evidence import (
    compute_survival_columns,
    cohort_segments,
)


@dataclass
class EvidenceComponents:
    extractor: Optional[Any]
    meta_clin: Optional[Any]
    meta_treat: Optional[Any]
    gx: Optional[Any]
    decoder_treat: Optional[Any]
    feat_spec: Optional[Any]
    oh_map: Optional[Any]
    ord_map: Optional[Any]
    num_ranges: Optional[Any]
    device: Optional[torch.device]
    print_probabilities: bool = False


class EvidencePackageGenerator:
    """Refactored generator for the refined evidence package."""

    def __init__(self, stage_iii_df: pd.DataFrame, components: EvidenceComponents):
        self.stage_iii_df = stage_iii_df
        self.components = components
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _ensure_survival_columns(self) -> None:
        """Compute survival columns if missing."""
        if {
            "time_difference_years",
            "event",
        }.issubset(self.stage_iii_df.columns):
            return
        self.stage_iii_df = compute_survival_columns(self.stage_iii_df)

    # ------------------------------------------------------------------
    def _generate_counterfactuals(self) -> pd.DataFrame:
        """Generate counterfactual outcomes for all treatment scenarios."""
        comp = self.components
        patient_ids = self.stage_iii_df["_id"].unique()
        dataset = TabularDatasetPID(self.stage_iii_df, comp.meta_clin, comp.meta_treat, seq_len=5)
        loader = DataLoader(dataset, batch_size=min(48, len(patient_ids)), shuffle=False)
        embeddings = []
        for _, clin_emb_batch, _ in loader:
            embeddings.append(clin_emb_batch.to(comp.device))
        clin_emb_all = torch.cat(embeddings, dim=0)

        scenarios = {
            "S": [1, 0, 0],
            "S_CT": [1, 1, 0],
            "S_RT": [1, 0, 1],
            "S_RT_CT": [1, 1, 1],
        }
        generated = []
        for name, t_vec in scenarios.items():
            treat_tensor = torch.zeros(
                (clin_emb_all.shape[0], clin_emb_all.shape[1], 3),
                dtype=torch.float32,
                device=comp.device,
            )
            treat_tensor[:, 0, :] = torch.tensor(t_vec, dtype=torch.float32, device=comp.device)
            with torch.no_grad():
                fake = comp.gx(clin_emb_all, treat_tensor)
            for i, pid in enumerate(patient_ids):
                decoded = decode_embedding(
                    comp.decoder_treat(fake[i, -1]),
                    comp.feat_spec,
                    comp.oh_map,
                    comp.ord_map,
                    comp.num_ranges,
                )
                if decoded.empty:
                    continue
                record = decoded.to_dict("records")[0]
                record.update({
                    "_id": pid,
                    "treatment_scenario": name,
                    "surgery": t_vec[0],
                    "chemotherapy": t_vec[1],
                    "radiotherapy": t_vec[2],
                })
                generated.append(record)
        return pd.DataFrame(generated)

    # ------------------------------------------------------------------
    def _analyze_cohort(self, df: pd.DataFrame) -> None:
        """Compute simple cohort counts and store in results."""
        _, counts = cohort_segments(df)
        self.results["cohort_counts"] = counts
        self.results["total_stage_iii"] = len(df)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline and return computed results."""
        self._ensure_survival_columns()
        comp = self.components
        if all(
            [
                comp.extractor,
                comp.meta_clin,
                comp.meta_treat,
                comp.gx,
                comp.decoder_treat,
                comp.feat_spec,
                comp.oh_map,
                comp.ord_map,
                comp.num_ranges,
                comp.device,
            ]
        ):
            generated_df = self._generate_counterfactuals()
            if not generated_df.empty:
                df = generated_df
                self.results["used_inference"] = True
                # Save the generated counterfactuals to CSV
                generated_df.to_csv("generated_counterfactuals_high_risk.csv", index=False)
            else:
                df = self.stage_iii_df
                self.results["used_inference"] = False
        else:
            df = self.stage_iii_df
            self.results["used_inference"] = False

        self._analyze_cohort(df)
        return self.results


def generate_refined_evidence_package(
    stage_iii_df: pd.DataFrame,
    extractor: Any = None,
    meta_clin: Any = None,
    meta_treat: Any = None,
    gx=None,
    decoder_treat=None,
    feat_spec=None,
    oh_map=None,
    ord_map=None,
    num_ranges=None,
    device: Optional[torch.device] = None,
    print_probabilities: bool = False,
) -> Dict[str, Any]:
    """Backward compatible wrapper that uses :class:`EvidencePackageGenerator`."""
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
        print_probabilities=print_probabilities,
    )
    generator = EvidencePackageGenerator(stage_iii_df, components)
    return generator.run()
"""
Data Preprocessing Script for Irembo Voice AI Intent Dataset

Prepares the Irembo Voice AI dataset for model training and evaluation.
Handles:
- CSV to JSONL conversion for training
- Text normalization for ASR noise
- Train/Val/Test split verification
- Data quality checks
"""

import pandas as pd
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class IremboDataPreprocessor:
    """Preprocessor for Irembo Voice AI intent dataset"""
    
    # Valid intents based on data dictionary
    VALID_INTENTS = {
        "check_application_status",
        "start_new_application",
        "requirements_information",
        "fees_information",
        "appointment_booking",
        "payment_help",
        "reset_password_login_help",
        "speak_to_agent",
        "cancel_or_reschedule_appointment",
        "update_application_details",
        "document_upload_help",
        "service_eligibility",
        "complaint_or_support_ticket",
    }
    
    VALID_LANGUAGES = {"en", "rw", "mixed"}
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize utterance text for ASR noise.
        Preserves code-switching and Kinyarwanda characters.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Normalize whitespace (multiple spaces to single)
        text = re.sub(r'\s+', ' ', text)
        
        # Don't lowercase - preserve case for proper nouns
        # Don't remove punctuation - useful for sentence structure
        
        return text
    
    def validate_sample(self, row: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single data sample"""
        errors = []
        
        # Check required fields
        if not row.get("utterance_text"):
            errors.append("Missing utterance_text")
        
        if not row.get("intent"):
            errors.append("Missing intent")
        elif row["intent"] not in self.VALID_INTENTS:
            errors.append(f"Invalid intent: {row['intent']}")
        
        if not row.get("language"):
            errors.append("Missing language")
        elif row["language"] not in self.VALID_LANGUAGES:
            errors.append(f"Invalid language: {row['language']}")
        
        return len(errors) == 0, errors
    
    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and validate CSV file"""
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe"""
        # Normalize text
        df["utterance_text_clean"] = df["utterance_text"].apply(self.normalize_text)
        
        # Validate samples
        validation_results = df.apply(
            lambda row: self.validate_sample(row.to_dict()), axis=1
        )
        df["is_valid"] = validation_results.apply(lambda x: x[0])
        
        # Log invalid samples
        invalid_count = (~df["is_valid"]).sum()
        if invalid_count > 0:
            print(f"  Warning: {invalid_count} invalid samples found")
        
        return df
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute dataset statistics"""
        stats = {
            "total_samples": len(df),
            "valid_samples": df["is_valid"].sum() if "is_valid" in df.columns else len(df),
            "intent_distribution": df["intent"].value_counts().to_dict(),
            "language_distribution": df["language"].value_counts().to_dict(),
            "avg_utterance_length": df["utterance_text"].str.len().mean(),
            "avg_asr_confidence": df["asr_confidence"].mean() if "asr_confidence" in df.columns else None,
            "channel_distribution": df["channel"].value_counts().to_dict() if "channel" in df.columns else {},
            "region_distribution": df["region"].value_counts().to_dict() if "region" in df.columns else {},
        }
        return stats
    
    def to_jsonl(self, df: pd.DataFrame, output_path: Path):
        """Convert dataframe to JSONL format for training"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                sample = {
                    "id": row.get("utterance_id", ""),
                    "text": row.get("utterance_text_clean", row.get("utterance_text", "")),
                    "original_text": row.get("utterance_text", ""),
                    "intent": row.get("intent", ""),
                    "language": row.get("language", "en"),
                    "asr_confidence": float(row.get("asr_confidence", 1.0)),
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Saved {len(df)} samples to {output_path}")
    
    def to_huggingface_format(self, df: pd.DataFrame, output_path: Path):
        """Convert to HuggingFace datasets format"""
        # Create label mapping
        label2id = {intent: i for i, intent in enumerate(sorted(self.VALID_INTENTS))}
        id2label = {i: intent for intent, i in label2id.items()}
        
        samples = []
        for _, row in df.iterrows():
            intent = row.get("intent", "")
            if intent in label2id:
                samples.append({
                    "text": row.get("utterance_text", ""),
                    "label": label2id[intent],
                    "language": row.get("language", "en"),
                })
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "data": samples,
                "label2id": label2id,
                "id2label": id2label,
                "num_labels": len(label2id),
            }, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved HuggingFace format to {output_path}")
    
    def process_all(self):
        """Process all dataset files"""
        print("=" * 60)
        print("IREMBO VOICE AI DATA PREPROCESSING")
        print("=" * 60)
        
        # Load datasets
        train_df = self.load_csv(self.data_dir / "voiceai_intent_train.csv")
        val_df = self.load_csv(self.data_dir / "voiceai_intent_val.csv")
        test_df = self.load_csv(self.data_dir / "voiceai_intent_test.csv")
        
        # Preprocess
        print("\nPreprocessing...")
        train_df = self.preprocess_dataframe(train_df)
        val_df = self.preprocess_dataframe(val_df)
        test_df = self.preprocess_dataframe(test_df)
        
        # Compute and print statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            stats = self.compute_statistics(df)
            print(f"\n{name} Set:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Avg utterance length: {stats['avg_utterance_length']:.1f} chars")
            if stats['avg_asr_confidence']:
                print(f"  Avg ASR confidence: {stats['avg_asr_confidence']:.3f}")
            print(f"  Language distribution: {stats['language_distribution']}")
        
        # Print intent distribution (from full dataset)
        all_df = pd.concat([train_df, val_df, test_df])
        intent_dist = all_df["intent"].value_counts()
        print("\n" + "=" * 60)
        print("INTENT DISTRIBUTION (All Splits)")
        print("=" * 60)
        for intent, count in intent_dist.items():
            pct = (count / len(all_df)) * 100
            print(f"  {intent:35} {count:4} ({pct:5.1f}%)")
        
        # Save processed files
        print("\n" + "=" * 60)
        print("SAVING PROCESSED FILES")
        print("=" * 60)
        
        # JSONL format
        self.to_jsonl(train_df, self.processed_dir / "train.jsonl")
        self.to_jsonl(val_df, self.processed_dir / "val.jsonl")
        self.to_jsonl(test_df, self.processed_dir / "test.jsonl")
        
        # HuggingFace format
        self.to_huggingface_format(train_df, self.processed_dir / "train_hf.json")
        self.to_huggingface_format(val_df, self.processed_dir / "val_hf.json")
        self.to_huggingface_format(test_df, self.processed_dir / "test_hf.json")
        
        # Save combined metadata
        metadata = {
            "dataset_name": "irembo_voiceai_intent",
            "splits": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "intents": list(self.VALID_INTENTS),
            "languages": list(self.VALID_LANGUAGES),
            "preprocessing_version": "1.0",
        }
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✅ Preprocessing complete!")
        print(f"   Processed files saved to: {self.processed_dir}")


def main():
    """Main entry point"""
    preprocessor = IremboDataPreprocessor(data_dir="data")
    preprocessor.process_all()


if __name__ == "__main__":
    main()

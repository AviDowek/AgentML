"""Check MileageBucket formula in experiments."""
import sys
sys.path.insert(0, '.')

from app.core.database import SessionLocal
from app.models.experiment import Experiment
from app.models.dataset_spec import DatasetSpec
import json

db = SessionLocal()

# Find the recent experiments
exps = db.query(Experiment).filter(
    Experiment.name.like('%v5_add_human_assignment%')
).order_by(Experiment.updated_at.desc()).limit(1).all()

for exp in exps:
    print(f"Experiment: {exp.name}")
    print(f"Dataset Spec ID: {exp.dataset_spec_id}")
    print()

    # Check dataset spec thoroughly
    if exp.dataset_spec_id:
        ds = db.query(DatasetSpec).filter(DatasetSpec.id == exp.dataset_spec_id).first()
        if ds:
            print(f"Dataset Spec Name: {ds.name}")
            print(f"spec_json keys: {list((ds.spec_json or {}).keys())}")

            spec = ds.spec_json or {}

            # Print all keys and their types
            for k, v in spec.items():
                if isinstance(v, list):
                    print(f"  {k}: list with {len(v)} items")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with keys {list(v.keys())[:5]}")
                else:
                    print(f"  {k}: {type(v).__name__} = {str(v)[:50]}")

            # Check target_creation
            tc = spec.get('target_creation', {})
            if tc:
                print(f"\n*** TARGET CREATION ***")
                print(f"  Column: {tc.get('column_name')}")
                print(f"  Data Type: {tc.get('data_type')}")
                print(f"  Formula: {tc.get('formula', '')[:200]}...")

            # Check for feature_engineering specifically - stored as 'engineered_features'
            fe = spec.get('engineered_features', []) or spec.get('feature_engineering', [])
            print(f"\nFeature Engineering ({len(fe)} items):")
            for i, f in enumerate(fe):
                name = f.get('name') or f.get('output_column', 'unknown')
                formula = f.get('formula', '')
                print(f"\n{i+1}. {name}:")
                print(f"   Formula: {formula[:150]}..." if len(formula) > 150 else f"   Formula: {formula}")

db.close()

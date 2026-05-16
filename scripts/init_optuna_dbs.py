#!/usr/bin/env python
"""Initialize Optuna SQLite databases for HPO experiments."""

import optuna
from pathlib import Path

# Database files to create
databases = [
    "optuna_resnet18.db",
    "optuna_efficientnet_b0.db",
    "optuna_convnext_tiny.db",
]

# Study names
study_names = [
    "resnet18_hpo",
    "efficientnet_b0_hpo",
    "convnext_tiny_hpo",
]

print("Initializing Optuna databases...")

for db_file, study_name in zip(databases, study_names):
    db_path = Path(db_file)

    # Remove existing database if it exists
    if db_path.exists():
        print(f"Removing existing database: {db_file}")
        db_path.unlink()

    # Create new database with proper initialization
    print(f"Creating database: {db_file} with study: {study_name}")

    try:
        # Create study with file storage
        # This will properly initialize the database schema
        study = optuna.create_study(
            storage=f"sqlite:///{db_file}",
            study_name=study_name,
            direction="maximize",
            load_if_exists=False
        )
        print(f"✓ Successfully created: {db_file}")
    except Exception as e:
        print(f"✗ Failed to create {db_file}: {e}")
        # Try alternative method
        try:
            import sqlite3
            conn = sqlite3.connect(db_file)
            conn.close()
            study = optuna.create_study(
                storage=f"sqlite:///{db_file}",
                study_name=study_name,
                direction="maximize",
                load_if_exists=True
            )
            print(f"✓ Successfully created with fallback: {db_file}")
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")

print("\nDatabase initialization complete!")
print("\nYou can now run HPO experiments:")
print("  ./scripts/run_hpo.sh")

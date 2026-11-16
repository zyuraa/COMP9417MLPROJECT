"""Export a reproducible demand-only candidate using Bryan's original live interface."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .lasso import BRYAN_DIR
from .package_train import save_artifacts


def export_demand(output, data, model, settings):
    here = Path(__file__).resolve().parent
    locked = {key: data.config[key] for key in
              ("selected_model_type", "alpha", "l1_ratio", "selected_features")}
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        save_artifacts(data, model, settings, stage / "artifacts")
        bundle = stage / "demand_candidate.zip"
        with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
            for name in ("model.py", "features.py"):
                archive.write(BRYAN_DIR / name, name)
            archive.write(here / "lasso.py", "lasso.py")
            archive.write(here / "package_train.py", "train.py")
            archive.writestr("settings.json", json.dumps({**locked, **settings}, indent=2) + "\n")
            for name in ("config.json", "demand_model.joblib"):
                archive.write(stage / "artifacts" / name, "artifacts/" + name)
        bundle.replace(output / bundle.name)

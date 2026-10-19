"""H2 acceptance checks using fixtures and temporary storage."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from foxtrot.config.checkpoint import CheckpointReference
from foxtrot.config.dataset import DatasetMetadata
from foxtrot.config.evaluation import EvaluationResult
from foxtrot.config.event import ProgressEvent
from foxtrot.config.job import JobSpec
from foxtrot.config.result import JobResult
from foxtrot.config.snapshot import resolve_job


class ConfigurationContractTests(unittest.TestCase):
    def request(self):
        return {
            "dataset_id": "dataset-demo",
            "tracking_destination_id": "local-mlflow",
            "training": {"epochs": 5, "seed": 42},
        }

    def test_configuration_files_round_trip(self):
        request = self.request()
        snapshots = resolve_job(request)

        with tempfile.TemporaryDirectory(prefix="foxtrot-h2-") as directory:
            submitted_path = Path(directory) / "submitted.json"
            resolved_path = Path(directory) / "resolved.json"

            submitted_path.write_text(
                snapshots.submitted_json, encoding="utf-8"
            )
            resolved_path.write_text(
                snapshots.resolved_json, encoding="utf-8"
            )

            submitted = json.loads(
                submitted_path.read_text(encoding="utf-8")
            )
            restored = JobSpec.model_validate_json(
                resolved_path.read_text(encoding="utf-8")
            )

            self.assertEqual(submitted, request)
            self.assertNotIn("batch_size", submitted["training"])
            self.assertEqual(restored, JobSpec.model_validate(request))
            self.assertEqual(restored.training.batch_size, 16)
            self.assertEqual(restored.schema_version, 1)

    def test_invalid_requests_report_fields(self):
        cases = [
            ({"training": {"epochs": "5"}}, ("training", "epochs")),
            ({"training": {"epochs": True}}, ("training", "epochs")),
            ({"training": {"batch_size": 0}}, ("training", "batch_size")),
            ({"training": {"seed": 2**32}}, ("training", "seed")),
            ({"training": {"device": "unknown"}}, ("training", "device")),
            ({"model_id": "unknown"}, ("model_id",)),
            ({"dataset_id": "/tmp/data"}, ("dataset_id",)),
            ({"schema_version": 2}, ("schema_version",)),
            ({"output_dir": "/tmp/shared"}, ("output_dir",)),
            ({"tracking_password": "fixture-only"}, ("tracking_password",)),
        ]

        for changes, expected_field in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError) as caught:
                    resolve_job({**self.request(), **changes})

                locations = {
                    error["loc"] for error in caught.exception.errors()
                }
                self.assertIn(expected_field, locations)

    def test_configurations_are_frozen_and_independent(self):
        request = self.request()
        snapshots = resolve_job(request)
        job = JobSpec.model_validate(request)

        request["training"]["epochs"] = 99

        self.assertEqual(
            json.loads(snapshots.submitted_json)["training"]["epochs"], 5
        )
        self.assertEqual(job.training.epochs, 5)

        with self.assertRaises(FrozenInstanceError):
            snapshots.resolved_json = "{}"

        with self.assertRaises(ValidationError):
            job.training.epochs = 99

    def test_related_records_round_trip(self):
        dataset = DatasetMetadata(
            dataset_id="dataset-demo",
            class_names=("Hexagon", "Triangle"),
            train_images_ref="train/images",
            validation_images_ref="val/images",
            content_sha256="1" * 64,
        )
        checkpoint = CheckpointReference(
            relative_path="weights/best.pt",
            sha256="2" * 64,
        )
        evaluation = EvaluationResult(
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.content_sha256,
            checkpoint=checkpoint,
            precision=0.82,
            recall=0.76,
            map50=0.80,
            map50_95=0.54,
        )
        result = JobResult(
            job_id="job-demo",
            outcome="finished",
            checkpoint=checkpoint,
            evaluation=evaluation,
            tracking_destination_id="local-mlflow",
        )
        event = ProgressEvent(
            job_id=result.job_id,
            sequence=1,
            timestamp=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
            event_type="state_changed",
            payload={"outcome": result.outcome},
        )

        for record in (dataset, checkpoint, evaluation, result, event):
            with self.subTest(contract=type(record).__name__):
                restored = type(record).model_validate_json(
                    record.model_dump_json()
                )
                self.assertEqual(restored, record)

        mismatched = result.model_dump()
        mismatched["checkpoint"]["sha256"] = "3" * 64
        with self.assertRaises(ValidationError):
            JobResult.model_validate(mismatched)

        invalid_event = event.model_dump(mode="json")
        invalid_event["payload"] = {"score": float("nan")}
        with self.assertRaises(ValidationError):
            ProgressEvent.model_validate_json(json.dumps(invalid_event))


if __name__ == "__main__":
    unittest.main()
"""Job configuration persistence checks use temporary fixtures."""

import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from foxtrot.config.job import JobSpec
from foxtrot.persistence.jobs import create_configured_job


class JobConfigurationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(
            prefix="foxtrot-job-configs-"
        )
        self.addCleanup(self.fixture.cleanup)

        self.root = Path(self.fixture.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

        self.workspace = self.root / "workspace"

    def request(self):
        return {
            "dataset_id": "dataset-demo",
            "tracking_destination_id": "local-mlflow",
            "training": {"epochs": 5},
        }

    def test_saves_submitted_and_resolved_settings(self):
        request = self.request()

        job_directory = create_configured_job(
            self.workspace,
            self.checkout,
            request,
        )

        submitted = json.loads(
            (job_directory / "submitted.json").read_text(
                encoding="utf-8"
            )
        )
        resolved = JobSpec.model_validate_json(
            (job_directory / "resolved.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            job_directory.parent,
            self.workspace / "jobs",
        )
        self.assertEqual(submitted, request)
        self.assertNotIn("batch_size", submitted["training"])
        self.assertEqual(resolved, JobSpec.model_validate(request))
        self.assertEqual(resolved.training.batch_size, 16)

    def test_new_job_preserves_previous_configuration(self):
        request = self.request()

        first = create_configured_job(
            self.workspace,
            self.checkout,
            request,
        )
        first_saved = (first / "resolved.json").read_text(
            encoding="utf-8"
        )

        request["training"]["epochs"] = 9

        second = create_configured_job(
            self.workspace,
            self.checkout,
            request,
        )
        second_saved = (second / "resolved.json").read_text(
            encoding="utf-8"
        )

        self.assertNotEqual(first, second)
        self.assertEqual(
            (first / "resolved.json").read_text(encoding="utf-8"),
            first_saved,
        )
        self.assertEqual(
            JobSpec.model_validate_json(first_saved).training.epochs,
            5,
        )
        self.assertEqual(
            JobSpec.model_validate_json(second_saved).training.epochs,
            9,
        )

    def test_invalid_request_creates_no_workspace(self):
        request = self.request()
        request["training"]["epochs"] = 0

        with self.assertRaises(ValidationError):
            create_configured_job(
                self.workspace,
                self.checkout,
                request,
            )

        self.assertFalse(self.workspace.exists())


if __name__ == "__main__":
    unittest.main()
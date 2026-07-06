from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import company_data_intake_validator as validator


EXPECTED_TABLES = {
    "orders",
    "dispatch_attempts",
    "deliveries",
    "routes",
    "fleet",
    "inventory",
    "costs",
}


class CompanyDataIntakeValidatorTests(unittest.TestCase):
    def test_valid_minimal_builtin_fixture_is_ready(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="built_in:valid_minimal",
            synthetic_data=True,
        )

        self.assertEqual(EXPECTED_TABLES, set(tables))
        self.assertEqual(
            {
                "report_metadata",
                "table_summaries",
                "join_key_checks",
                "data_quality_checks",
                "warnings",
                "decision",
            },
            set(report),
        )
        self.assertTrue(report["report_metadata"]["synthetic_data"])
        self.assertEqual("built_in:valid_minimal", report["report_metadata"]["source_kind"])
        self.assertEqual(7, report["report_metadata"]["table_count"])
        self.assertEqual("COMPANY_DATA_SCHEMA_READY", report["decision"]["classification"])
        self.assertEqual([], _blocking_checks(report))
        self.assertEqual([], _warnings(report))

    def test_missing_required_columns_need_fixes(self) -> None:
        report = _report_from_fixture("missing_required_columns")

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("fail", checks["orders_required_columns"]["status"])
        self.assertIn("promised_window_end", checks["orders_required_columns"]["missing_columns"])
        self.assertEqual("fail", checks["dispatch_attempts_required_columns"]["status"])
        self.assertIn("dispatch_attempt_id", checks["dispatch_attempts_required_columns"]["missing_columns"])

    def test_bad_timestamps_need_fixes(self) -> None:
        report = _report_from_fixture("bad_timestamps")

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        check_ids = {check["check_id"] for check in _blocking_checks(report)}
        self.assertIn("orders_timestamp_parse", check_ids)
        self.assertIn("orders_promised_window_order", check_ids)

    def test_blank_required_join_values_need_fixes(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")
        tables["inventory"][0]["sku_id"] = ""
        tables["inventory"][0]["site_id"] = ""

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="fixture:blank_inventory_keys",
            synthetic_data=True,
        )

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("fail", checks["inventory_required_values"]["status"])
        self.assertIn("sku_id", checks["inventory_required_values"]["blank_fields"])
        self.assertIn("site_id", checks["inventory_required_values"]["blank_fields"])

    def test_missing_required_table_blocks_report(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")
        del tables["fleet"]

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="fixture:missing_fleet",
            synthetic_data=True,
        )

        self.assertEqual("COMPANY_DATA_SCHEMA_BLOCKED", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("blocked", checks["fleet_table_present"]["status"])

    def test_duplicate_primary_keys_need_fixes(self) -> None:
        report = _report_from_fixture("duplicate_keys")

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("fail", checks["orders_duplicate_primary_key"]["status"])
        self.assertEqual("fail", checks["dispatch_attempts_duplicate_primary_key"]["status"])

    def test_invalid_fleet_and_reorder_labels_need_fixes(self) -> None:
        report = _report_from_fixture("invalid_labels")

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        check_ids = {check["check_id"] for check in _blocking_checks(report)}
        self.assertIn("fleet_fleet_type_controlled_label", check_ids)
        self.assertIn("inventory_reorder_type_controlled_label", check_ids)

    def test_missing_required_join_key_needs_fixes(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")
        tables["dispatch_attempts"][0]["order_id"] = "missing_order"

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="fixture:missing_join",
            synthetic_data=True,
        )

        self.assertEqual("COMPANY_DATA_SCHEMA_NEEDS_FIXES", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("fail", checks["dispatch_attempts_order_id_join"]["status"])

    def test_carrier_only_join_mismatch_is_warnings_only_when_vehicle_join_is_valid(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")
        tables["dispatch_attempts"][0]["carrier_id"] = "missing_carrier"

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="fixture:carrier_warning",
            synthetic_data=True,
        )

        self.assertEqual("COMPANY_DATA_SCHEMA_WARNINGS_ONLY", report["decision"]["classification"])
        checks = _checks_by_id(report)
        self.assertEqual("warning", checks["dispatch_attempts_carrier_id_join"]["status"])

    def test_optional_fields_absent_do_not_block_readiness(self) -> None:
        tables = validator.built_in_synthetic_fixture("valid_minimal")
        for table_name, optional_fields in validator.TABLE_SCHEMAS.items():
            for field in optional_fields["optional"]:
                for row in tables[table_name]:
                    row.pop(field, None)

        report = validator.build_company_data_schema_report(
            tables,
            source_kind="fixture:optional_absent",
            synthetic_data=True,
        )

        self.assertEqual("COMPANY_DATA_SCHEMA_READY", report["decision"]["classification"])
        self.assertEqual([], _blocking_checks(report))

    def test_synthetic_fixture_root_csv_loader_is_ready(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tables = validator.built_in_synthetic_fixture("valid_minimal")
            _write_fixture_root(root, tables)

            loaded = validator.load_fixture_root(root)

            self.assertEqual(tables, loaded)
            report = validator.build_company_data_schema_report(
                loaded,
                source_kind=f"synthetic_fixture_root:{root}",
                synthetic_data=True,
            )
            self.assertEqual("COMPANY_DATA_SCHEMA_READY", report["decision"]["classification"])

    def test_cli_writes_only_requested_output_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "schema_report.json"
            before = _relative_files(root)

            exit_code, stdout, stderr = _run_main(
                "--use-built-in-synthetic-fixture",
                "valid_minimal",
                "--output",
                str(output),
            )

            after = _relative_files(root)
            self.assertEqual(0, exit_code)
            self.assertEqual("COMPANY_DATA_SCHEMA_READY", stdout.strip())
            self.assertEqual("", stderr)
            self.assertEqual(before | {Path("schema_report.json")}, after)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("COMPANY_DATA_SCHEMA_READY", report["decision"]["classification"])

    def test_cli_refuses_missing_synthetic_input(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "schema_report.json"

            with self.assertRaises(SystemExit) as raised:
                _run_main("--output", str(output))

            self.assertEqual(2, raised.exception.code)
            self.assertFalse(output.exists())

    def test_cli_refuses_conflicting_synthetic_input_modes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_fixture_root(root / "fixtures", validator.built_in_synthetic_fixture("valid_minimal"))
            output = root / "schema_report.json"

            with self.assertRaises(SystemExit) as raised:
                _run_main(
                    "--use-built-in-synthetic-fixture",
                    "valid_minimal",
                    "--synthetic-fixture-root",
                    str(root / "fixtures"),
                    "--output",
                    str(output),
                )

            self.assertEqual(2, raised.exception.code)
            self.assertFalse(output.exists())


def _report_from_fixture(name: str) -> dict:
    return validator.build_company_data_schema_report(
        validator.built_in_synthetic_fixture(name),
        source_kind=f"built_in:{name}",
        synthetic_data=True,
    )


def _checks_by_id(report: dict) -> dict[str, dict]:
    checks = {}
    for section in ("data_quality_checks", "join_key_checks"):
        for check in report[section]:
            checks[check["check_id"]] = check
    return checks


def _blocking_checks(report: dict) -> list[dict]:
    return [
        check
        for section in ("data_quality_checks", "join_key_checks")
        for check in report[section]
        if check["status"] == "fail"
    ]


def _warnings(report: dict) -> list[dict]:
    return [check for check in report["warnings"] if check["status"] == "warning"]


def _write_fixture_root(root: Path, tables: dict[str, list[dict[str, str]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for table_name, rows in tables.items():
        path = root / f"{table_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def _run_main(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = validator.main(list(args))
    return exit_code, stdout.getvalue(), stderr.getvalue()

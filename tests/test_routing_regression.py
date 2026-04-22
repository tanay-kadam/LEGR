"""
test_routing_regression.py -- Structural Regression Tests
=========================================================

Pure structural assertions on taxonomy design and validation logic.
No LLM calls required -- these run instantly and catch regressions
in branch layout, balance, tier distribution, and hallucination blocking.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vocab_config as vc
from routers import _validate_tool
from routing_tiers import EXPLICIT_ROUTING_TOOL_NAMES_15
from routing_tiers import EXPLICIT_ROUTING_TOOL_NAMES_30


def _load_taxonomies(tool_count: int):
    """Reload taxonomies module with a specific tool count."""
    vc.ACTIVE_TOOL_COUNT = tool_count
    import taxonomies

    importlib.reload(taxonomies)
    return taxonomies.SEMANTIC_TAXONOMY, taxonomies.TOOL_BOUND_TAXONOMY, taxonomies.ALL_TOOLS


def _load_router_and_taxonomies(tool_count: int):
    """Reload both taxonomies and routers for a specific tool count."""
    vc.ACTIVE_TOOL_COUNT = tool_count
    import routers
    import taxonomies

    importlib.reload(taxonomies)
    importlib.reload(routers)
    return taxonomies, routers


def _dataset_label_tools(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        fieldnames = rows.fieldnames or []
        label_col = "ground_truth" if "ground_truth" in fieldnames else "label"
        return sorted({row[label_col] for row in rows})


@pytest.fixture(params=[15, 30, 45], ids=["15tools", "30tools", "45tools"])
def tier(request):
    """Parametrized fixture that yields (sem_tax, tb_tax, all_tools, tool_count)."""
    sem, tb, tools = _load_taxonomies(request.param)
    yield sem, tb, tools, request.param
    vc.ACTIVE_TOOL_COUNT = 45


@pytest.fixture
def tb45():
    """Tool-Bound taxonomy at 45 tools specifically."""
    _, tb, tools = _load_taxonomies(45)
    yield tb, tools
    vc.ACTIVE_TOOL_COUNT = 45


class TestBranchExclusivity:
    def test_no_tool_in_multiple_tb_branches(self, tier):
        _, tb, _, _ = tier
        seen: dict[str, str] = {}
        for branch_name, branch_info in tb["branches"].items():
            for tool in branch_info["tools"]:
                assert tool not in seen, (
                    f"Tool '{tool}' appears in both '{seen[tool]}' and "
                    f"'{branch_name}'"
                )
                seen[tool] = branch_name

    def test_all_active_tools_assigned_tb(self, tier):
        _, tb, all_tools, _ = tier
        assigned = set()
        for branch_info in tb["branches"].values():
            assigned.update(branch_info["tools"])
        missing = set(all_tools) - assigned
        assert not missing, f"Tools not assigned to any TB branch: {missing}"

    def test_no_tool_in_multiple_sem_branches(self, tier):
        sem, _, _, _ = tier
        seen: dict[str, str] = {}
        for branch_name, branch_info in sem["branches"].items():
            for tool in branch_info["tools"]:
                assert tool not in seen, (
                    f"Semantic: '{tool}' in both '{seen[tool]}' and "
                    f"'{branch_name}'"
                )
                seen[tool] = branch_name

    def test_all_active_tools_assigned_sem(self, tier):
        sem, _, all_tools, _ = tier
        assigned = set()
        for branch_info in sem["branches"].values():
            assigned.update(branch_info["tools"])
        missing = set(all_tools) - assigned
        assert not missing, f"Tools not assigned to any Semantic branch: {missing}"


class TestTaxonomyParity:
    def test_same_tool_sets(self, tier):
        sem, tb, _, _ = tier
        sem_tools = set()
        for branch_info in sem["branches"].values():
            sem_tools.update(branch_info["tools"])
        tb_tools = set()
        for branch_info in tb["branches"].values():
            tb_tools.update(branch_info["tools"])
        assert sem_tools == tb_tools, (
            f"Taxonomies cover different tools. "
            f"Only in Semantic: {sem_tools - tb_tools}, "
            f"Only in Tool-Bound: {tb_tools - sem_tools}"
        )


EXPECTED_BRANCH_COUNT = {15: 3, 30: 5, 45: 5}
EXPECTED_BRANCH_SIZE = {15: 5, 30: 6, 45: 9}


class TestBranchCountParity:
    def test_equal_branch_counts(self, tier):
        sem, tb, _, tool_count = tier
        sem_count = len(sem["branches"])
        tb_count = len(tb["branches"])
        assert sem_count == tb_count, (
            f"At {tool_count} tools: Semantic has {sem_count} branches, "
            f"Tool-Bound has {tb_count} -- must be equal"
        )

    def test_expected_branch_count_at_each_tier(self, tier):
        sem, tb, _, tool_count = tier
        expected = EXPECTED_BRANCH_COUNT[tool_count]
        assert len(sem["branches"]) == expected, (
            f"Semantic has {len(sem['branches'])} branches at {tool_count}, "
            f"expected {expected}"
        )
        assert len(tb["branches"]) == expected, (
            f"Tool-Bound has {len(tb['branches'])} branches at {tool_count}, "
            f"expected {expected}"
        )


class TestBranchBalance:
    def test_sem_branches_equal_size(self, tier):
        sem, _, _, tool_count = tier
        expected = EXPECTED_BRANCH_SIZE[tool_count]
        for name, info in sem["branches"].items():
            assert len(info["tools"]) == expected, (
                f"Semantic '{name}' has {len(info['tools'])} tools at "
                f"{tool_count}, expected {expected}"
            )

    def test_tb_branches_equal_size(self, tier):
        _, tb, _, tool_count = tier
        expected = EXPECTED_BRANCH_SIZE[tool_count]
        for name, info in tb["branches"].items():
            assert len(info["tools"]) == expected, (
                f"Tool-Bound '{name}' has {len(info['tools'])} tools at "
                f"{tool_count}, expected {expected}"
            )


EXPECTED_TB_BRANCHES_45 = {
    "Data Retrieval & Monitoring",
    "State Modification & Provisioning",
    "Communication & Orchestration",
    "Infrastructure Lifecycle",
    "Access Control & Configuration",
}

EXPECTED_SEM_BRANCHES_45 = {
    "Infrastructure Operations",
    "Security & Data Protection",
    "User & Account Management",
    "Deployment & Observability",
    "Communication & Data Services",
}

EXPECTED_TB_BRANCHES_15 = {
    "Data Retrieval & Monitoring",
    "State Modification & Provisioning",
    "Communication & Orchestration",
}

EXPECTED_SEM_BRANCHES_15 = {
    "IT Support",
    "Security & Compliance",
    "Billing & Data Management",
}

EXPECTED_TB_BRANCHES_30 = {
    "Observe, Review & Assess",
    "Access Decisions & Credential Actions",
    "Live System & Record Changes",
    "Recovery, Reversal & Continuity Actions",
    "Case, Escalation & Durable Records",
}

EXPECTED_SEM_BRANCHES_30 = {
    "Service Support & Systems Administration",
    "Security, Risk & Compliance Response",
    "Identity, Access & Governance Services",
    "Release & Platform Engineering",
    "Service Continuity & Customer Operations",
}

EXPECTED_ACTIVE_TOOLS_30 = {
    "check_service_status",
    "provision_workspace",
    "reset_user_password",
    "restart_service",
    "create_support_ticket",
    "inspect_security_alerts",
    "revoke_system_access",
    "quarantine_endpoint",
    "escalate_security_incident",
    "log_compliance_event",
    "generate_access_report",
    "assign_access_role",
    "approve_access_request",
    "notify_access_change",
    "update_identity_record",
    "validate_release_readiness",
    "deploy_service_release",
    "enable_feature_flag",
    "rollback_service_release",
    "record_release_note",
    "run_load_test",
    "scale_service_capacity",
    "trigger_failover",
    "schedule_maintenance_window",
    "snapshot_system_state",
    "update_customer_record",
    "authorize_data_export",
    "process_refund",
    "send_customer_notification",
    "archive_customer_data",
}

EXPECTED_SEMANTIC_BRANCHES_30_TOOLS = {
    "Service Support & Systems Administration": {
        "check_service_status",
        "provision_workspace",
        "reset_user_password",
        "restart_service",
        "create_support_ticket",
        "schedule_maintenance_window",
    },
    "Security, Risk & Compliance Response": {
        "inspect_security_alerts",
        "revoke_system_access",
        "quarantine_endpoint",
        "escalate_security_incident",
        "log_compliance_event",
        "snapshot_system_state",
    },
    "Identity, Access & Governance Services": {
        "generate_access_report",
        "assign_access_role",
        "approve_access_request",
        "notify_access_change",
        "update_identity_record",
        "authorize_data_export",
    },
    "Release & Platform Engineering": {
        "validate_release_readiness",
        "deploy_service_release",
        "enable_feature_flag",
        "rollback_service_release",
        "record_release_note",
        "run_load_test",
    },
    "Service Continuity & Customer Operations": {
        "scale_service_capacity",
        "trigger_failover",
        "update_customer_record",
        "process_refund",
        "send_customer_notification",
        "archive_customer_data",
    },
}

EXPECTED_TB_BRANCHES_30_TOOLS = {
    "Observe, Review & Assess": {
        "check_service_status",
        "inspect_security_alerts",
        "generate_access_report",
        "validate_release_readiness",
        "run_load_test",
        "snapshot_system_state",
    },
    "Access Decisions & Credential Actions": {
        "reset_user_password",
        "revoke_system_access",
        "assign_access_role",
        "approve_access_request",
        "notify_access_change",
        "authorize_data_export",
    },
    "Live System & Record Changes": {
        "provision_workspace",
        "deploy_service_release",
        "enable_feature_flag",
        "scale_service_capacity",
        "update_identity_record",
        "update_customer_record",
    },
    "Recovery, Reversal & Continuity Actions": {
        "restart_service",
        "quarantine_endpoint",
        "rollback_service_release",
        "trigger_failover",
        "schedule_maintenance_window",
        "process_refund",
    },
    "Case, Escalation & Durable Records": {
        "create_support_ticket",
        "escalate_security_incident",
        "log_compliance_event",
        "record_release_note",
        "send_customer_notification",
        "archive_customer_data",
    },
}

EXPECTED_SEMANTIC_DESCRIPTIONS_30 = {
    "Service Support & Systems Administration": (
        "Requests involving service support, workspace setup, account recovery, and routine systems administration tasks."
    ),
    "Security, Risk & Compliance Response": (
        "Requests involving security review, incident handling, risk containment, and compliance response activities work."
    ),
    "Identity, Access & Governance Services": (
        "Requests involving identity records, access permissions, approval workflows, and governance decisions for accounts."
    ),
    "Release & Platform Engineering": (
        "Requests involving release readiness, production rollouts, reliability testing, and platform delivery changes work."
    ),
    "Service Continuity & Customer Operations": (
        "Requests involving customer records, billing actions, service continuity, and ongoing operational maintenance work."
    ),
}

EXPECTED_TOOL_BOUND_DESCRIPTIONS_30 = {
    "Observe, Review & Assess": (
        "Requests involving observation, inspection, evaluation, and evidence capture before taking follow-on actions elsewhere."
    ),
    "Access Decisions & Credential Actions": (
        "Requests involving access decisions, credential changes, and account-state updates affecting user permissions directly."
    ),
    "Live System & Record Changes": (
        "Requests involving provisioning, deployment, scaling, and direct updates to live records or systems."
    ),
    "Recovery, Reversal & Continuity Actions": (
        "Requests involving containment, rollback, restart, reversal, and continuity actions on existing systems safely."
    ),
    "Case, Escalation & Durable Records": (
        "Requests involving logging, notification, escalation, scheduling, and routing work to people or records."
    ),
}


class TestBranchNames:
    def test_tb_branch_names_at_15(self):
        _, tb, _ = _load_taxonomies(15)
        actual = set(tb["branches"].keys())
        assert actual == EXPECTED_TB_BRANCHES_15, (
            f"Expected 15-tool TB branches {EXPECTED_TB_BRANCHES_15}, got {actual}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_sem_branch_names_at_15(self):
        sem, _, _ = _load_taxonomies(15)
        actual = set(sem["branches"].keys())
        assert actual == EXPECTED_SEM_BRANCHES_15, (
            f"Expected 15-tool Semantic branches {EXPECTED_SEM_BRANCHES_15}, got {actual}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_tb_branch_names_at_30(self):
        _, tb, _ = _load_taxonomies(30)
        actual = set(tb["branches"].keys())
        assert actual == EXPECTED_TB_BRANCHES_30, (
            f"Expected 30-tool TB branches {EXPECTED_TB_BRANCHES_30}, got {actual}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_sem_branch_names_at_30(self):
        sem, _, _ = _load_taxonomies(30)
        actual = set(sem["branches"].keys())
        assert actual == EXPECTED_SEM_BRANCHES_30, (
            f"Expected 30-tool Semantic branches {EXPECTED_SEM_BRANCHES_30}, got {actual}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_tb_branch_names_at_45(self, tb45):
        tb, _ = tb45
        actual = set(tb["branches"].keys())
        assert actual == EXPECTED_TB_BRANCHES_45, (
            f"Expected TB branches {EXPECTED_TB_BRANCHES_45}, got {actual}"
        )

    def test_sem_branch_names_at_45(self):
        sem, _, _ = _load_taxonomies(45)
        actual = set(sem["branches"].keys())
        assert actual == EXPECTED_SEM_BRANCHES_45, (
            f"Expected Semantic branches {EXPECTED_SEM_BRANCHES_45}, got {actual}"
        )
        vc.ACTIVE_TOOL_COUNT = 45


class TestTierVocabularyAlignment:
    def test_15_active_tools_match_upgraded_dataset(self):
        _, _, all_tools = _load_taxonomies(15)
        dataset_tools = _dataset_label_tools(
            ROOT / "upgraded_data" / "routing_15tools" / "base_cleaned.csv"
        )
        assert sorted(all_tools) == dataset_tools, (
            f"15-tool taxonomy set does not match upgraded dataset. "
            f"Only in taxonomy: {sorted(set(all_tools) - set(dataset_tools))}, "
            f"Only in dataset: {sorted(set(dataset_tools) - set(all_tools))}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    @pytest.mark.parametrize(
        "split_name",
        sorted(
            path.name
            for path in (ROOT / "upgraded_data" / "routing_30tools").glob("*.csv")
        ),
    )
    def test_30_active_tools_match_every_upgraded_split(self, split_name):
        _, _, all_tools = _load_taxonomies(30)
        dataset_tools = _dataset_label_tools(
            ROOT / "upgraded_data" / "routing_30tools" / split_name
        )
        assert sorted(all_tools) == dataset_tools, (
            f"30-tool taxonomy set does not match {split_name}. "
            f"Only in taxonomy: {sorted(set(all_tools) - set(dataset_tools))}, "
            f"Only in dataset: {sorted(set(dataset_tools) - set(all_tools))}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_active_tools_match_explicit_routing_contract(self):
        _, _, all_tools = _load_taxonomies(30)
        assert sorted(all_tools) == sorted(EXPLICIT_ROUTING_TOOL_NAMES_30)
        vc.ACTIVE_TOOL_COUNT = 45


class TestBranchPromptShape:
    def test_30_tool_branch_prompt_uses_full_taxonomy_formatter_for_both_taxonomies(self):
        taxonomies, routers = _load_router_and_taxonomies(30)
        for taxonomy in (taxonomies.SEMANTIC_TAXONOMY, taxonomies.TOOL_BOUND_TAXONOMY):
            branch_text = routers._build_branch_text(taxonomy)
            assert branch_text == taxonomies.format_taxonomy_prompt(taxonomy)
            for branch_name, branch_info in taxonomy["branches"].items():
                assert branch_name in branch_text
                assert branch_info["description"] in branch_text
                for tool in branch_info["tools"]:
                    assert tool in branch_text
                    assert taxonomies.TOOL_DESCRIPTIONS[tool] in branch_text
        vc.ACTIVE_TOOL_COUNT = 45


class TestThirtyToolContracts:
    def test_30_tool_semantic_branches_match_expected_contract(self):
        sem30, _, _ = _load_taxonomies(30)
        for branch_name, expected_tools in EXPECTED_SEMANTIC_BRANCHES_30_TOOLS.items():
            actual_tools = set(sem30["branches"][branch_name]["tools"])
            assert actual_tools == expected_tools
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_tool_bound_branches_match_expected_contract(self):
        _, tb30, _ = _load_taxonomies(30)
        for branch_name, expected_tools in EXPECTED_TB_BRANCHES_30_TOOLS.items():
            actual_tools = set(tb30["branches"][branch_name]["tools"])
            assert actual_tools == expected_tools
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_branch_sets_cover_expected_vocabulary(self):
        sem30, tb30, _ = _load_taxonomies(30)
        sem_tools = {
            tool
            for branch_info in sem30["branches"].values()
            for tool in branch_info["tools"]
        }
        tb_tools = {
            tool
            for branch_info in tb30["branches"].values()
            for tool in branch_info["tools"]
        }
        assert sem_tools == EXPECTED_ACTIVE_TOOLS_30
        assert tb_tools == EXPECTED_ACTIVE_TOOLS_30
        vc.ACTIVE_TOOL_COUNT = 45

    def test_no_semantic_branch_equals_any_tool_bound_branch(self):
        sem30, tb30, _ = _load_taxonomies(30)
        for sem_branch, sem_info in sem30["branches"].items():
            sem_tools = set(sem_info["tools"])
            for tb_branch, tb_info in tb30["branches"].items():
                tb_tools = set(tb_info["tools"])
                assert sem_tools != tb_tools, (
                    f"Semantic branch '{sem_branch}' should not match tool-bound "
                    f"branch '{tb_branch}' exactly"
                )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_branch_descriptions_match_expected_contract(self):
        sem30, tb30, _ = _load_taxonomies(30)
        for branch_name, description in EXPECTED_SEMANTIC_DESCRIPTIONS_30.items():
            assert sem30["branches"][branch_name]["description"] == description
        for branch_name, description in EXPECTED_TOOL_BOUND_DESCRIPTIONS_30.items():
            assert tb30["branches"][branch_name]["description"] == description
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_branch_descriptions_are_single_sentence_and_no_tool_names(self):
        sem30, tb30, _ = _load_taxonomies(30)
        sem_lengths = []
        for branch_name in EXPECTED_SEMANTIC_BRANCHES_30_TOOLS:
            description = sem30["branches"][branch_name]["description"]
            sem_lengths.append(len(description.split()))
            assert description.startswith("Requests involving ")
            assert description.endswith(".")
            assert description.count(".") == 1
            for tool in EXPLICIT_ROUTING_TOOL_NAMES_30:
                assert tool not in description.lower()

        tb_lengths = []
        for branch_name in EXPECTED_TB_BRANCHES_30_TOOLS:
            description = tb30["branches"][branch_name]["description"]
            tb_lengths.append(len(description.split()))
            assert description.startswith("Requests involving ")
            assert description.endswith(".")
            assert description.count(".") == 1
            for tool in EXPLICIT_ROUTING_TOOL_NAMES_30:
                assert tool not in description.lower()

        assert len(set(sem_lengths)) == 1
        assert len(set(tb_lengths)) == 1
        assert sem_lengths == tb_lengths
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_description_word_counts_match_branch_for_branch(self):
        sem30, tb30, _ = _load_taxonomies(30)
        sem_lengths = [
            len(sem30["branches"][branch_name]["description"].split())
            for branch_name in EXPECTED_SEMANTIC_DESCRIPTIONS_30
        ]
        tb_lengths = [
            len(tb30["branches"][branch_name]["description"].split())
            for branch_name in EXPECTED_TOOL_BOUND_DESCRIPTIONS_30
        ]
        assert sem_lengths == tb_lengths, (
            f"Expected matched description word counts at 30 tools, got "
            f"semantic={sem_lengths}, tool_bound={tb_lengths}"
        )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_semantic_descriptions_are_domain_based(self):
        sem30, _, _ = _load_taxonomies(30)
        domain_keywords = {
            "service",
            "security",
            "identity",
            "release",
            "customer",
        }
        for branch_name, branch_info in sem30["branches"].items():
            text = f"{branch_name} {branch_info['description']}".lower()
            assert any(keyword in text for keyword in domain_keywords), (
                f"Semantic branch '{branch_name}' should read as domain-based"
            )
        vc.ACTIVE_TOOL_COUNT = 45

    def test_30_tool_tool_bound_descriptions_are_action_based(self):
        _, tb30, _ = _load_taxonomies(30)
        action_keywords = {
            "observation",
            "access",
            "provisioning",
            "containment",
            "logging",
        }
        for branch_name, branch_info in tb30["branches"].items():
            text = f"{branch_name} {branch_info['description']}".lower()
            assert any(keyword in text for keyword in action_keywords), (
                f"Tool-bound branch '{branch_name}' should read as action-based"
            )
        vc.ACTIVE_TOOL_COUNT = 45


FAILURE_CASE_TOOLS = [
    "deploy_container",
    "check_status",
    "archive_data",
    "transfer_ownership",
    "update_database",
]


class TestFailureCaseToolsExist:
    @pytest.mark.parametrize("tool", FAILURE_CASE_TOOLS)
    def test_tool_has_branch(self, tb45, tool):
        tb, all_tools = tb45
        if tool not in all_tools:
            pytest.skip(f"{tool} not active at this tier")
        from taxonomies import get_branch_for_tool
        branch = get_branch_for_tool(tb, tool)
        assert branch is not None, f"{tool} not assigned to any branch"


class TestHallucinationBlocking:
    def test_invalid_tool_is_blocked(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool(
            "Airtable", all_tools, all_tools,
        )
        assert validated is None
        assert hallucinated is True

    def test_completely_random_name_is_blocked(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool(
            "xyzzy_nonexistent_tool_99", all_tools, all_tools,
        )
        assert validated is None
        assert hallucinated is True

    def test_valid_tool_passes(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool(
            "query_database", all_tools, all_tools,
        )
        assert validated == "query_database"
        assert hallucinated is False

    def test_close_match_recovers(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool(
            "querry_database", all_tools, all_tools,
        )
        assert validated == "query_database"
        assert hallucinated is False

    def test_none_input_is_safe(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool(None, all_tools, all_tools)
        assert validated is None
        assert hallucinated is False

    def test_empty_string_is_safe(self, tb45):
        _, all_tools = tb45
        validated, hallucinated = _validate_tool("", all_tools, all_tools)
        assert validated is None
        assert hallucinated is False


REGRESSION_QUERIES = [
    ("Run the pipeline and push the build to staging-db-02", "deploy_container"),
    ("Scan prod-web-01 -- is it healthy?", "check_status"),
    ("Back up the old records and remove them", "archive_data"),
    ("Grant Charlie full control over the resource", "transfer_ownership"),
    ("Query payment-api-03 to see if it's still alive", "check_status"),
    ("Do me a favor and persist bob's data to the database", "update_database"),
]


class TestRegressionQueryToolsReachable:
    @pytest.mark.parametrize(
        "query,expected_tool",
        REGRESSION_QUERIES,
        ids=[query[:40] for query, _ in REGRESSION_QUERIES],
    )
    def test_expected_tool_in_taxonomy(self, tb45, query, expected_tool):
        tb, all_tools = tb45
        assert expected_tool in all_tools, (
            f"Expected tool '{expected_tool}' not in active tools"
        )
        from taxonomies import get_branch_for_tool
        branch = get_branch_for_tool(tb, expected_tool)
        assert branch is not None, (
            f"Expected tool '{expected_tool}' has no branch assignment"
        )

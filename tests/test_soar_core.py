from pathlib import Path

from generators.csv_parser import CSVParser
from src.engine.control import ControlState
from src.engine.goals import GoalState
from src.engine.impasse import ImpasseManager
from src.engine.operator_handler import OperatorHandler
from src.engine.policy import CognitivePolicy
from src.engine.productions import PreferenceResolver, ProductionCompiler, ProductionMatcher
from src.engine.runtime_model import RuntimeModel
from src.engine.types import OperatorProposal, Preference
from src.engine.tracing import TraceRecorder
from src.engine.working_memory import WorkingMemory


def _build_parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()
    return parser


class _HandlerEncoder:
    def generate(self, template_id: str, slot_values: dict) -> str:
        return f"tmpl:{template_id}"


class _HandlerToolExecutor:
    def execute_from_operator_spec(self, operator_spec, slot_values, metadata):
        class _Result:
            data = {"task-complete": "yes", "agent-message": "done"}
            metadata = {"matched_rule_id": "rule-1"}

        return _Result()


def test_working_memory_projects_slots_into_wmes_and_back():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)

    memory.set_slot("lighting-params.room", "bedroom", provenance="test", parent_slot="lighting-params")

    assert memory.get_slot("lighting-params.room") == "bedroom"
    wme = memory.get_wme("lighting-params.room")
    assert wme is not None
    assert wme.attribute == "room"
    assert wme.parent_identifier == "wme:lighting-params"
    assert memory.snapshot_slots()["lighting-params.room"] == "bedroom"


def test_control_state_projects_pending_and_interrupted_operator_ids_into_memory():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)
    control = ControlState(memory, unknown_sentinel=parser.unknown_sentinel)

    pending = next(op for op in parser.operators if op.operator_id == "confirm-understanding")
    interrupted = next(op for op in parser.operators if op.operator_id == "ask-lighting-room")

    control.pending_operator = pending
    control.interrupted_operator = interrupted

    assert memory.get_slot("control.pending-operator-id") == "confirm-understanding"
    assert memory.get_slot("control.interrupted-operator-id") == "ask-lighting-room"

    control.clear_pending()
    control.clear_interrupted()

    assert memory.get_slot("control.pending-operator-id") == parser.unknown_sentinel
    assert memory.get_slot("control.interrupted-operator-id") == parser.unknown_sentinel


def test_goal_state_projects_goal_subgoal_and_impasse_depth_into_memory():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)
    control = ControlState(memory, unknown_sentinel=parser.unknown_sentinel)
    runtime_model = RuntimeModel.from_parser(parser)
    goals = GoalState(
        memory,
        unknown_sentinel=parser.unknown_sentinel,
        phase_goals=runtime_model.phase_goals,
    )
    manager = ImpasseManager()

    memory.set_slot("dialogue-phase", "parameter-collection", provenance="test")
    pending = next(op for op in parser.operators if op.operator_id == "ask-lighting-room")
    control.pending_operator = pending

    manager.open_impasse(
        kind="missing-required-parameter",
        goal="collect-task-parameters",
        context={"dialogue-phase": "parameter-collection"},
        cycle_index=2,
    )
    goals.sync(control.pending_operator, manager)

    assert memory.get_slot("control.current-goal") == "parameter collection"
    assert memory.get_slot("control.current-subgoal") == "ask-lighting-room"
    assert memory.get_slot("control.substate-depth") == "1"
    assert memory.get_slot("control.current-impasse-kind") == "missing-required-parameter"


def test_operator_rows_compile_into_proposals_and_resolve_by_preference():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)
    memory.set_slot("dialogue-phase", "parameter-collection", provenance="test")
    memory.set_slot("intent", "configure-lighting", provenance="test")
    memory.set_slot("lighting-params.room", "bedroom", provenance="test")
    memory.set_slot("lighting-params.budget", "low", provenance="test")

    compiler = ProductionCompiler()
    productions = compiler.compile_action_space(parser.operators)
    matcher = ProductionMatcher(unknown_sentinel=parser.unknown_sentinel)
    proposals = matcher.propose(productions, memory)

    proposal_ids = {proposal.operator_id for proposal in proposals}
    assert "ask-lighting-automation" in proposal_ids

    outcome = PreferenceResolver().select(proposals)
    assert outcome.status == "selected"
    assert outcome.selected_proposal is not None
    assert outcome.selected_proposal.operator_id == "ask-lighting-automation"
    assert outcome.selected_proposal.supporting_productions == ["operator:ask-lighting-automation"]


def test_impasse_and_trace_core_track_explicit_substates():
    manager = ImpasseManager()
    recorder = TraceRecorder()

    impasse = manager.open_impasse(
        kind="no-operator",
        goal="parameter-collection",
        context={"dialogue-phase": "parameter-collection"},
        cycle_index=3,
    )
    recorder.record("impasse_opened", 3, impasse_id=impasse.impasse_id, kind=impasse.kind)

    assert manager.current_depth() == 1
    latest = recorder.latest("impasse_opened")
    assert latest is not None
    assert latest.payload["impasse_id"] == impasse.impasse_id

    manager.resolve_latest()
    assert manager.current_depth() == 0


def test_impasse_manager_returns_recovery_directives():
    manager = ImpasseManager()

    _, help_directive = manager.handle_interrupt(
        routing_category="help_request",
        goal="classify-intent",
        context={"dialogue-phase": "init"},
        cycle_index=1,
        has_interrupted_operator=False,
    )
    assert help_directive.next_operator_id == "classify-intent"
    assert help_directive.slot_updates["dialogue-phase"] == "intent-collection"

    _, ambiguous_directive = manager.handle_ambiguous_confirmation(
        goal="confirm-understanding",
        context={"intent": "configure-security"},
        cycle_index=2,
    )
    assert ambiguous_directive.suppress_impasse_increment is True

    _, no_operator_directive = manager.handle_no_operator(
        goal="parameter-collection",
        context={"intent": "configure-lighting"},
        cycle_index=3,
    )
    assert no_operator_directive.next_operator_id == "impasse-recovery"

    _, tie_directive = manager.handle_tie(
        goal="parameter-collection",
        context={"intent": "configure-lighting"},
        cycle_index=4,
    )
    assert tie_directive.next_operator_id == "impasse-recovery"

    _, failure_directive = manager.handle_interpretation_failure(
        goal="confirm-understanding",
        context={"intent": "configure-lighting"},
        cycle_index=5,
    )
    assert failure_directive.next_operator_id == "impasse-recovery"


def test_preference_resolver_handles_require_reject_and_tie():
    resolver = PreferenceResolver()

    selected_outcome = resolver.select(
        [
            OperatorProposal(
                operator_id="a",
                operator_type="nlu",
                phase="init",
                operator_spec=None,
                preferences=[Preference(kind="acceptable", weight=5, source="p1")],
            ),
            OperatorProposal(
                operator_id="b",
                operator_type="nlu",
                phase="init",
                operator_spec=None,
                preferences=[Preference(kind="require", weight=1, source="p2")],
            ),
            OperatorProposal(
                operator_id="c",
                operator_type="nlu",
                phase="init",
                operator_spec=None,
                preferences=[Preference(kind="reject", weight=1, source="p3")],
            ),
        ]
    )
    assert selected_outcome.status == "selected"
    assert selected_outcome.selected_proposal is not None
    assert selected_outcome.selected_proposal.operator_id == "b"

    tie_outcome = resolver.select(
        [
            OperatorProposal(
                operator_id="x",
                operator_type="nlu",
                phase="init",
                operator_spec=None,
                preferences=[Preference(kind="acceptable", weight=10, source="px")],
            ),
            OperatorProposal(
                operator_id="y",
                operator_type="nlu",
                phase="init",
                operator_spec=None,
                preferences=[Preference(kind="acceptable", weight=10, source="py")],
            ),
        ]
    )
    assert tie_outcome.status == "tie"
    assert "top score" in tie_outcome.reason
    assert {proposal.operator_id for proposal in tie_outcome.competing_proposals} == {"x", "y"}


def test_operator_proposal_preference_summary_is_traceable():
    proposal = OperatorProposal(
        operator_id="x",
        operator_type="nlu",
        phase="init",
        operator_spec=None,
        preferences=[
            Preference(kind="acceptable", weight=10, source="p1"),
            Preference(kind="require", weight=1, source="p2"),
        ],
    )

    assert proposal.score > 10
    assert proposal.preference_summary() == [
        {"kind": "acceptable", "weight": 10, "source": "p1"},
        {"kind": "require", "weight": 1, "source": "p2"},
    ]


def test_cognitive_policy_covers_chaining_interrupts_and_affirmation_mapping():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)
    policy = CognitivePolicy(
        parser=parser,
        unknown_sentinel=parser.unknown_sentinel,
        runtime_model=RuntimeModel.from_parser(parser),
    )

    memory.set_slot("conversation-category", "task_related", provenance="test")
    operator = next(op for op in parser.operators if op.operator_id == "greet-and-classify")
    chaining_eval = policy.evaluate_chaining(memory, operator, {"intent": "configure-lighting"})
    assert chaining_eval.decision is True
    assert chaining_eval.matched_rule_ids == ("chain-followup-routing",)

    interrupt_eval = policy.evaluate_routing_interrupt("help_request")
    assert interrupt_eval.decision is True
    assert interrupt_eval.matched_rule_ids == ("interrupt-category-help_request",)

    capture_eval = policy.evaluate_interrupt_capture("off_topic")
    assert capture_eval.decision is True
    assert capture_eval.matched_rule_ids == ("capture-interrupted-off_topic",)

    memory.set_slot("affirmation", "confirmed", provenance="test")
    confirm_operator = next(op for op in parser.operators if op.operator_id == "confirm-understanding")
    affirmation_eval = policy.evaluate_affirmation_mapping(memory, confirm_operator)
    assert affirmation_eval.decision is True
    assert affirmation_eval.matched_rule_ids == ("affirmation-map",)
    updates = policy.affirmation_updates(memory, confirm_operator)
    assert updates[0][:2] == ("confirmation-status", "confirmed")

    ambiguous_eval = policy.evaluate_ambiguous_confirmation(memory, confirm_operator, state_changed=False)
    assert ambiguous_eval.decision is True
    assert ambiguous_eval.matched_rule_ids == ("ambiguous-confirmation-routing",)


def test_cognitive_policy_maps_intents_and_interrupt_reseed_rules():
    parser = _build_parser()
    memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=parser.unknown_sentinel)
    runtime_model = RuntimeModel.from_parser(parser)
    policy = CognitivePolicy(
        parser=parser,
        unknown_sentinel=parser.unknown_sentinel,
        runtime_model=runtime_model,
    )

    climate_eval = policy.evaluate_intent_domain("configure-climate")
    assert climate_eval.decision is True
    assert climate_eval.matched_rule_ids == ("intent-domain-configure-climate",)
    assert policy.intent_utterance_type("configure-security") == "security_related"

    reseed_eval = policy.evaluate_interrupt_reseed(
        memory,
        operator_id="respond-help",
        has_interrupted_operator=False,
    )
    assert reseed_eval.decision is True
    assert reseed_eval.matched_rule_ids == ("interrupt-reseed-help",)
    assert policy.interrupt_reseed_category(
        memory,
        operator_id="respond-off-topic",
        has_interrupted_operator=False,
    ) == "off_topic"

    resume_eval = policy.evaluate_interrupt_resume(memory, has_interrupted_operator=True)
    assert resume_eval.decision is True
    assert resume_eval.matched_rule_ids == ("resume-interrupted-operator",)


def test_runtime_model_derives_phases_goals_and_intent_domains_from_config():
    parser = _build_parser()
    runtime_model = RuntimeModel.from_parser(parser)

    assert runtime_model.phase_order["init"] == 0
    assert runtime_model.phase_order["pipeline"] == 4
    assert runtime_model.phase_goals["intent-collection"] == "intent collection"
    assert runtime_model.intent_utterance_types["configure-lighting"] == "task_related"
    assert runtime_model.intent_utterance_types["configure-climate"] == "climate_related"


def test_operator_handler_supports_multi_slot_orchestration_updates():
    parser = _build_parser()
    handler = OperatorHandler(parser=parser, encoder=_HandlerEncoder(), tool_executor=_HandlerToolExecutor())
    operator = next(op for op in parser.operators if op.operator_id == "evaluate-and-select-rule")
    operator = type(operator)(
        operator_id="multi-orch",
        operator_type="orchestration",
        phase="init",
        priority=1,
        conditions="",
        utterance_template_id="greeting",
        classifier_tool="",
        write_method="single",
        tool_name="",
        requires_slot="",
        affected_slot="conversation-category,greeting-issued",
        expected_value="task_related,true",
    )

    result = handler.execute(
        operator_spec=operator,
        slot_values={},
        interrupted_operator=None,
        metadata={},
        on_seed_followup=lambda _operator_id: [],
        on_nlu_pending=lambda _operator: None,
    )

    assert result.slot_updates == [
        ("conversation-category", "task_related", "orchestration"),
        ("greeting-issued", "true", "orchestration"),
    ]
    assert result.message == "tmpl:greeting"

import contextlib
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("check_research_gates",ROOT/"scripts/check_research_gates.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ResearchGateTests(unittest.TestCase):
    def test_current_repository_satisfies_gates(self):
        self.assertEqual(mod.main(),0)

    def test_v4_no_go_decisions_are_bound_to_their_complete_ci_artifacts(self):
        for exp_id in mod.V4_NO_GO:
            with self.subTest(experiment=exp_id):
                self.assertEqual(mod.v4_no_go_errors(exp_id,ROOT),[])

    def test_m009_is_locked_before_a_completed_m002_v5_pass(self):
        for status in ("prepared", "running", "completed"):
            experiments={"M009":{"status":status},"M002-v5":{"status":"planned"}}
            self.assertEqual(
                mod.m009_lock_errors(ROOT,experiments,{}),
                ["M009: locked until M002-v5 is completed with PASS"],
            )
        self.assertEqual(mod.m009_lock_errors(ROOT,{"M009":{"status":"planned"}},{}),[])

    def test_unlisted_m009_cannot_bypass_the_m002_v5_dependency_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            experiments.mkdir()
            (experiments / "preregistration.toml").write_text("[experiment]\n", encoding="utf-8")
            (experiments / "registry.toml").write_text("experiment = []\n", encoding="utf-8")
            self.assertEqual(
                mod.launch_errors(root, "M009"),
                ["M009: locked until M002-v5 is completed with PASS"],
            )

    def test_m009_recomputes_the_bound_decision_instead_of_trusting_a_pass_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            experiment=root/"experiments/model/M002-v5-factorized-typed-attention"
            results=experiment/"results"
            scripts=root/"scripts"
            results.mkdir(parents=True)
            scripts.mkdir()
            sources=[]
            for seed in (17,29,43,71,101):
                path=results/f"run-{seed}.json"
                record={"seed":seed,"value":"bound"}
                path.write_text(json.dumps(record),encoding="utf-8",newline="\n")
                relative=path.relative_to(root).as_posix()
                sources.append({
                    "path":relative,
                    "canonical_sha256":hashlib.sha256(mod.canonical_json(record).encode("utf-8")).hexdigest(),
                })
            expected={
                "decision":"PASS",
                "gates":{"all":True},
                "provenance":{"source_records":sources},
            }
            (root/"expected.json").write_text(json.dumps(expected),encoding="utf-8",newline="\n")
            (scripts/"aggregate_m002_v5.py").write_text(
                "import json\n"
                "def decide(records, root):\n"
                "    return json.loads((root / 'expected.json').read_text(encoding='utf-8'))\n",
                encoding="utf-8",newline="\n",
            )
            decision_path=results/"m002-v5-decision.json"
            decision_path.write_text(json.dumps(expected),encoding="utf-8",newline="\n")
            visible=lambda checkout,path: checkout/path if (checkout/path).is_file() else None
            with (
                mock.patch.object(mod,"committed_regular_file",side_effect=visible),
                mock.patch.object(mod,"m002_v5_archived_evidence_errors",return_value=[]),
            ):
                self.assertEqual(mod.bound_m002_v5_pass_errors(root,experiment),[])
                forged_source = copy.deepcopy(expected)
                forged_source["provenance"]["source_records"][0]["path"] = "forged/evidence-17.json"
                decision_path.write_text(json.dumps(forged_source),encoding="utf-8",newline="\n")
                self.assertEqual(
                    mod.bound_m002_v5_pass_errors(root,experiment),
                    ["M009: M002-v5 decision does not bind exactly the immutable five-seed run archive"],
                )
                forged=dict(expected)
                forged["claim"]="self-asserted"
                decision_path.write_text(json.dumps(forged),encoding="utf-8",newline="\n")
                self.assertEqual(
                    mod.bound_m002_v5_pass_errors(root,experiment),
                    ["M009: M002-v5 decision does not equal the canonical recomputation"],
                )

    def test_m009_refuses_a_pass_when_immutable_record_history_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experiments/model/M002-v5-factorized-typed-attention"
            path.mkdir(parents=True)
            with mock.patch.object(mod, "m002_v5_archived_evidence_errors", return_value=["forged git_sha"]):
                self.assertEqual(
                    mod.bound_m002_v5_pass_errors(root, path),
                    ["M009: locked because M002-v5 archived evidence is invalid: forged git_sha"],
                )

    def test_m009_requires_the_exact_factorized_config_and_checkpoint_contract(self):
        contract = {
            "source_experiment": "M002-v5",
            "attention_mode": "factorized-v2",
            "d_model": 48,
            "rank": 16,
            "bias_limit": 2.0,
            "metadata_dropout": 0.1,
            "router": {
                "mode": "calibrated-cosine-v2",
                "logit_scale": 5.0,
                "label_smoothing": 0.05,
                "consistency_weight": 0.1,
            },
            "architecture_contract_sha256": "a" * 64,
            "contract_path": "experiments/model/M002-v5-factorized-typed-attention/factorized-v2-contract.txt",
        }
        digest = contract["architecture_contract_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m009 = root / "experiments/model/M009"
            m009.mkdir(parents=True)
            (m009 / "config.toml").write_text(
                "[m002_v5_binding]\n"
                "source_experiment = \"M002-v5\"\n"
                "attention_mode = \"factorized-v2\"\n"
                "d_model = 48\nrank = 16\nbias_limit = 2.0\nmetadata_dropout = 0.1\n"
                f"contract_path = \"{contract['contract_path']}\"\n"
                f"architecture_contract_sha256 = \"{digest}\"\n"
                f"runtime_contract = \"{contract['contract_path']}\"\n"
                f"runtime_contract_sha256 = \"{digest}\"\n"
                f"checkpoint_architecture_contract_sha256 = \"{digest}\"\n"
                "[m002_v5_binding.router]\n"
                "mode = \"calibrated-cosine-v2\"\nlogit_scale = 5.0\nlabel_smoothing = 0.05\nconsistency_weight = 0.1\n",
                encoding="utf-8",
            )
            (m009 / "experiment.toml").write_text(
                "entrypoint = \"cargo +1.95.0-x86_64-pc-windows-gnu run --release --locked --quiet --jobs 1 --manifest-path model/burn-a0/Cargo.toml --example m009_learned_backend -- --m002-v5-contract <runtime_contract> --m002-v5-contract-sha256 <runtime_contract_sha256>\"\n",
                encoding="utf-8",
            )
            with mock.patch.object(mod, "m002_v5_factorized_contract", return_value=(contract, [])):
                self.assertEqual(mod.m009_architecture_binding_errors(root, root / "M002", m009), [])
                (m009 / "experiment.toml").write_text(
                    (m009 / "experiment.toml").read_text(encoding="utf-8").replace(
                        "--example m009_learned_backend", "--example unrelated_program"
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    mod.m009_architecture_binding_errors(root, root / "M002", m009),
                    ["M009: entrypoint does not pass the frozen runtime contract to model construction"],
                )
                (m009 / "experiment.toml").write_text(
                    (m009 / "experiment.toml").read_text(encoding="utf-8").replace(
                        "unrelated_program", "m009_learned_backend"
                    ),
                    encoding="utf-8",
                )
                (m009 / "config.toml").write_text(
                    (m009 / "config.toml").read_text(encoding="utf-8").replace("rank = 16", "rank = 8"),
                    encoding="utf-8",
                )
                self.assertEqual(
                    mod.m009_architecture_binding_errors(root, root / "M002", m009),
                    ["M009: rank does not bind the frozen M002-v5 FactorizedV2 contract"],
                )

    def test_bound_m002_v2_no_go_marker_accepts_only_the_two_exact_freezes(self):
        experiment=ROOT/"experiments/model/M002-v2-typed-attention"
        manifest=mod.load(experiment/"experiment.toml")
        marker,errors=mod.no_go_marker("M002-v2",experiment,ROOT,manifest)
        self.assertEqual(errors,[])
        self.assertIsNotNone(marker)
        current=manifest
        for commit in mod.M002_V2_NO_GO_COMMITS:
            then=mod.toml_at(ROOT,commit,"experiments/model/M002-v2-typed-attention/experiment.toml")
            self.assertTrue(mod.accepts_m002_v2_no_go(marker,"M002-v2",commit,then,current,["entrypoint"]))
            self.assertFalse(mod.accepts_m002_v2_no_go(marker,"M002-v2",commit,then,current,["entrypoint","hypothesis"]))
        then=mod.toml_at(ROOT,mod.M002_V2_NO_GO_COMMITS[0],"experiments/model/M002-v2-typed-attention/experiment.toml")
        self.assertFalse(mod.accepts_m002_v2_no_go(marker,"M002-v2","0"*40,then,current,["entrypoint"]))

    def test_m002_v2_no_go_marker_rejects_tampering_and_never_completes_study(self):
        experiment=ROOT/"experiments/model/M002-v2-typed-attention"
        manifest=mod.load(experiment/"experiment.toml")
        marker_path=experiment/"results/NO-GO.toml"
        original=mod.load(marker_path)
        for field,value in (
            ("experiment_id","M001-v4"),
            ("decision","completed"),
            ("field","hypothesis"),
            ("current_entrypoint_sha256","0"*64),
        ):
            tampered=dict(original)
            tampered[field]=value
            with self.subTest(field=field), mock.patch.object(mod,"load",side_effect=lambda path, tampered=tampered: tampered if Path(path)==marker_path else original):
                accepted,errors=mod.no_go_marker("M002-v2",experiment,ROOT,manifest)
                self.assertIsNone(accepted)
                self.assertTrue(errors)
        self.assertNotEqual(manifest.get("status"),"completed")

    def test_stale_results_of_a_completed_experiment_fail_the_gate(self):
        completed=[
            item["id"]
            for item in mod.load(ROOT/"experiments/registry.toml").get("experiment",[])
            if mod.load(ROOT/"experiments"/item["path"]/"experiment.toml").get("status")=="completed"
        ]
        self.assertTrue(completed)
        checked=[]

        def stale(exp_id,experiment,results,root,listed_experiment):
            checked.append(exp_id)
            self.assertEqual((root,results.parent),(mod.ROOT,experiment))
            self.assertEqual(listed_experiment,exp_id in mod.listed_now(mod.ROOT))
            return [f"{exp_id}: results/run.json ran at other code"]

        output=io.StringIO()
        with mock.patch.object(mod.experiment_records,"staleness_errors",side_effect=stale),contextlib.redirect_stdout(output):
            self.assertEqual(mod.main(),1)
        self.assertEqual(checked,completed)
        for exp_id in completed:
            self.assertIn(f"ERROR: {exp_id}: results/run.json ran at other code",output.getvalue())

    def test_artifacts_of_a_completed_experiment_that_are_not_one_aggregate_fail_the_gate(self):
        # Both required paths can exist and run.json's provenance hold while
        # metrics.json or mutations.json is not what run.json was aggregated
        # with.
        completed=[
            item["id"]
            for item in mod.load(ROOT/"experiments/registry.toml").get("experiment",[])
            if mod.load(ROOT/"experiments"/item["path"]/"experiment.toml").get("status")=="completed"
        ]
        checked=[]

        def unbound(exp_id,experiment,results,root):
            checked.append(exp_id)
            self.assertEqual((root,results.parent),(mod.ROOT,experiment))
            return [f"{exp_id}: results/metrics.json is not the metrics run.json was aggregated with"]

        output=io.StringIO()
        with mock.patch.object(mod.experiment_records,"aggregate_errors",side_effect=unbound),contextlib.redirect_stdout(output):
            self.assertEqual(mod.main(),1)
        self.assertEqual(checked,completed)
        for exp_id in completed:
            self.assertIn(f"ERROR: {exp_id}: results/metrics.json is not the metrics",output.getvalue())

def toml_value(value) -> str:
    """`value` as a TOML value (integers, booleans, strings, floats, lists and
    inline tables are all the fixtures need)."""
    if isinstance(value,bool):
        return "true" if value else "false"
    if isinstance(value,(int,float)):
        return repr(value)
    if isinstance(value,str):
        return json.dumps(value)
    if isinstance(value,list):
        return "["+", ".join(toml_value(element) for element in value)+"]"
    if isinstance(value,dict):
        return "{ "+", ".join(f"{key} = {toml_value(item)}" for key,item in value.items())+" }"
    raise TypeError(value)

def toml_key(key: str) -> str:
    """`key` bare when TOML allows it, quoted otherwise."""
    return key if key and all(character.isascii() and (character.isalnum() or character in "_-") for character in key) else json.dumps(key)

def toml_table(name: str, table: dict) -> str:
    """`table` as the TOML table `[name]`."""
    return f"[{name}]\n"+"".join(f"{toml_key(key)} = {toml_value(value)}\n" for key,value in table.items())

def git(root: Path, *args: str) -> str:
    """Run git in `root` as the tests' committer, without the automatic
    maintenance that can outlive the temporary repository."""
    command = [
        "git",
        "-c", "user.name=PTR tests",
        "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
        "-c", "maintenance.auto=false",
        "-c", "gc.auto=0",
        *args,
    ]
    return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()

def commit_all(root: Path, message: str = "tree") -> str:
    """Commit everything under `root`, making it a repository first, and
    return the commit."""
    if not (root/".git").exists():
        git(root,"init","-q")
    git(root,"add","-A")
    git(root,"commit","-q","--no-verify","--allow-empty","-m",message)
    return git(root,"rev-parse","HEAD")

def write(root: Path, relative: str, text: str) -> None:
    """Write `text` to `relative` under `root`, creating its directories."""
    path=root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding="utf-8")

PLAIN_MODEL='version = 1\n[model]\nbackend = "must-be-pinned-before-run"\nmodel = "must-be-pinned-before-run"\n'
CONFIG_HEADER='version = 1\nkind = "workspace-area"\nname = "X900-fixture"\npath = "experiments/semdb/X900-fixture"\ntests_dir = "tests"\n'
CARGO_OUTSIDE=("X900: entrypoint gives Cargo {} {}, outside what the repository's watch reads, whose sources no watch or "
               "record binds; name a path the repository holds")
CARGO_FIRST=("X900: entrypoint gives Cargo {} where one of its built-in commands bench, build, check, run, test comes "
             "first (after at most a +toolchain): an alias, an option before the command or an external command could "
             "have Cargo build or run what no record binds")
CARGO_DIRECTORY=("X900: entrypoint gives Cargo a directory to run in (-C), whose Cargo configuration no check reads; a "
                 "listed experiment's Cargo runs from the root")
CARGO_CONFIGURATION="X900: entrypoint gives Cargo configuration on its command line (--config), which can name a rustc wrapper, flags or sources outside the commit; set what the build needs in the repository's .cargo/config.toml"
TABLE={"schema":1,"harness":"fixture","seeds":[17,29],"programs":["rmw","set_op"],"see_intent":False}
REQUIRED={"schema":"int","harness":"str","seeds":"int-list","programs":"str-list","see_intent":"bool"}
BASELINE_PATH="research/baselines/fixture/config.toml"
BASELINE={"name":"fixture","path":BASELINE_PATH,"keys":["model.revision","answer.generators"],"status_key":"status"}
BASELINE_CONFIG={"status":"reference-implemented","model":{"revision":"0123abc"},"answer":{"generators":["g1","g2"]}}
DIGEST=object()

def baseline_digest(files: dict[str, str], executable=()) -> str:
    """The digest the gate freezes a baseline by: the canonical digest of
    each file of its directory, by path within it, mapped to its git mode
    (100755 for those named in `executable`, 100644 otherwise) and the
    SHA-256 of its bytes, line endings included."""
    return mod.experiment_records.canonical_digest({
        name:("100755" if name in executable else "100644")+" "+hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name,text in files.items()
    })

def gate(root: Path) -> tuple[int, list[str]]:
    """Run the gate on the tree at `root`, with the checks of completed
    experiments' archived results (git history the fixtures do not have)
    passing once they are asked about that tree, and return its exit code
    and error lines."""
    def archived(exp_id,experiment,results,given_root,listed_experiment=False):
        """Pass the archived-results checks, once they are asked about `root`."""
        if given_root!=root or not experiment.is_relative_to(root):
            raise AssertionError(f"{exp_id}: archived results checked in {given_root}, not {root}")
        return []

    output=io.StringIO()
    with (
        mock.patch.object(mod.experiment_records,"staleness_errors",side_effect=archived),
        mock.patch.object(mod.experiment_records,"aggregate_errors",side_effect=archived),
        contextlib.redirect_stdout(output),
    ):
        code=mod.main(root)
    lines=output.getvalue().splitlines()
    return code,[line.removeprefix("ERROR: ") for line in lines if line.startswith("ERROR: ")]

def unarchived(status: str) -> list[str]:
    """What a fixture at `status` that holds no archive adds to the gate's
    errors: a listed experiment that is completed needs its aggregate,
    run.json, and the metrics.json beside it, whatever its manifest lists."""
    if status!="completed":
        return []
    return [f"X900: completed experiment missing {name}" for name in ("run.json","metrics.json")]

def superseded_unarchived(completed: str, now: str = "superseded") -> list[str]:
    """What a fixture that was completed at `completed`, holds no archive and
    has the status `now` adds to the gate's errors."""
    at=f"experiment, completed at {completed[:12]} and now '{now}',"
    return [f"X900: {at} missing {name}" for name in ("run.json","metrics.json")]

ONCE_COMPLETED=re.compile(r"X900: experiment, completed at [0-9a-f]{12} and now '\w+', missing (run|metrics)\.json")

class PreregistrationGateTests(unittest.TestCase):
    """The preregistration gate on fixture trees holding one listed
    experiment, X900, and one baseline."""

    def tree(
        self,
        status="prepared",
        table=TABLE,
        required=REQUIRED,
        digest=DIGEST,
        rules=DIGEST,
        seeds=(17,29),
        baselines=(BASELINE,),
        baseline_config=BASELINE_CONFIG,
        entry_extra=None,
        listed_text=None,
        freeze_baselines=True,
    ) -> Path:
        """A fixture tree; `digest` is the manifest's preregistration_sha256,
        the table's own digest unless given (None leaves it out), `rules` its
        preregistration_rules_sha256, the digest of the list's entry unless
        given (None leaves it out), and
        `listed_text`, when given, the whole of the list. With
        `freeze_baselines`, the table also holds each baseline's
        `baseline_<name>_sha256`, the digest of its directory, which holds
        only its configuration, unless the table sets it itself. The tree is
        a git repository with nothing committed yet."""
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        baseline_text=None
        if baseline_config is not None:
            values={name:value for name,value in baseline_config.items() if not isinstance(value,dict)}
            baseline_text="".join(f"{name} = {toml_value(value)}\n" for name,value in values.items())
            for name,section in baseline_config.items():
                if name not in values:
                    baseline_text+="\n"+toml_table(name,section)
        if table is not None and freeze_baselines and baseline_text is not None:
            table=dict(table)
            for baseline in baselines:
                name=baseline.get("name") if isinstance(baseline,dict) else None
                if isinstance(name,str) and f"baseline_{name}_sha256" not in table:
                    table[f"baseline_{name}_sha256"]=baseline_digest({"config.toml":baseline_text})
        write(root,"experiments/registry.toml",f'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/X900-fixture"\nstatus = "{status}"\n')
        config=CONFIG_HEADER+("\n"+toml_table("preregistration",table) if table is not None else "")
        write(root,"experiments/semdb/X900-fixture/config.toml",config)
        listed="version = 1\n\n"+toml_table("experiment.X900.required",required)
        for baseline in baselines:
            listed+="\n[[experiment.X900.baseline]]\n"+"".join(f"{key} = {toml_value(value)}\n" for key,value in baseline.items())
        if entry_extra:
            listed+="\n"+entry_extra
        listed=listed if listed_text is None else listed_text
        write(root,"experiments/preregistration.toml",listed)
        manifest={"version":1,"id":"X900","status":status,"seeds":list(seeds),"entrypoint":"bench <seed>","required_artifacts":[]}
        if digest is DIGEST:
            digest=mod.experiment_records.preregistration_digest(table) if table is not None else None
        if digest is not None:
            manifest["preregistration_sha256"]=digest
        if rules is DIGEST:
            entries=tomllib.loads(listed).get("experiment",{})
            entry=entries.get("X900") if isinstance(entries,dict) else None
            try:
                rules=mod.experiment_records.canonical_digest(entry) if isinstance(entry,dict) else None
            except (ValueError,UnicodeEncodeError):
                rules=None
        if rules is not None:
            manifest["preregistration_rules_sha256"]=rules
        write(root,"experiments/semdb/X900-fixture/experiment.toml","".join(f"{key} = {toml_value(value)}\n" for key,value in manifest.items()))
        write(root,"research/baselines/plain_model/config.toml",PLAIN_MODEL)
        if baseline_text is not None:
            write(root,BASELINE_PATH,baseline_text)
        git(root,"init","-q")
        return root

    def assert_blocked(self, root: Path, *errors: str) -> None:
        """The gate on `root` fails with exactly `errors`, in any order."""
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertEqual(sorted(lines),sorted(errors))

    def test_planned_experiments_are_not_gated(self):
        unset={**TABLE,"harness":"must-be-pinned-before-prepared","see_intent":"must-be-signed-by-owner"}
        for table in (TABLE,unset,None):
            with self.subTest(table=table):
                root=self.tree(status="planned",table=table,digest=None,baseline_config={"status":"blocked-unpinned"})
                self.assertEqual(gate(root),(0,[]))
        # Nor is a superseded one: another experiment replaced it, and that
        # one's preregistration counts. A failed one ran, and is gated.
        self.assertEqual(gate(self.tree(status="superseded",table=None,digest=None)),(0,[]))

    def test_a_complete_preregistration_passes(self):
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                code,lines=gate(self.tree(status=status))
                self.assertEqual((code,lines),(1 if unarchived(status) else 0,unarchived(status)))
        # A key the list does not require is preregistered too, empty lists
        # included; only required lists must hold something.
        self.assertEqual(gate(self.tree(table={**TABLE,"low_cells":[],"note":"x"})),(0,[]))
        # Without a baseline, only the table is checked.
        self.assertEqual(gate(self.tree(baselines=(),baseline_config=None)),(0,[]))

    def test_a_missing_required_key_keeps_an_experiment_from_leaving_planned(self):
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                without={key:value for key,value in TABLE.items() if key!="harness"}
                self.assert_blocked(self.tree(status=status,table=without),"X900: preregistration key harness is missing",
                                    *unarchived(status))
                self.assert_blocked(
                    self.tree(status=status,table=None,digest="0"*64),
                    "X900: config.toml has no [preregistration] table",
                    *unarchived(status),
                )
                root=self.tree(status=status)
                (root/"experiments/semdb/X900-fixture/config.toml").unlink()
                self.assert_blocked(root,"X900: config.toml does not exist",*unarchived(status))

    def test_a_placeholder_or_wrongly_typed_value_blocks(self):
        cases=[
            ("harness","must-be-pinned-before-prepared","is a placeholder ('must-be-pinned-before-prepared')"),
            ("harness","","is a placeholder ('')"),
            ("harness"," None ","is a placeholder (' None ')"),
            ("harness","unconfigured","is a placeholder ('unconfigured')"),
            # An owner decision passes as no type until the owner signs it.
            ("see_intent","must-be-signed-by-owner","is a placeholder ('must-be-signed-by-owner')"),
            ("see_intent",0,"must be a bool, not an integer"),
            ("see_intent","false","must be a bool, not a string"),
            ("schema",True,"must be an int, not a boolean"),
            ("schema","1","must be an int, not a string"),
            ("harness",1,"must be a str, not an integer"),
            ("harness",["fixture"],"must be a str, not a list"),
            ("seeds",17,"must be an int-list, not an integer"),
            ("seeds",["17","29"],"has element 0 that must be an int, not a string"),
            ("programs",["rmw","must-be-pinned-x"],"has element 1 that is a placeholder ('must-be-pinned-x')"),
            ("programs",[1,2],"has element 0 that must be a str, not an integer"),
            # Case and surrounding space do not hide a placeholder.
            ("harness","Must-Be-Signed-by-owner","is a placeholder ('Must-Be-Signed-by-owner')"),
            ("harness"," MUST-BE-PINNED ","is a placeholder (' MUST-BE-PINNED ')"),
            # Nor does leaving the key out of the list: every preregistered
            # value is pinned, though a list the list does not require may be
            # empty.
            ("reviewer_pool","must-be-signed-by-owner","is a placeholder ('must-be-signed-by-owner')"),
            ("margin","must-be-pinned-before-prepared","is a placeholder ('must-be-pinned-before-prepared')"),
            ("note","",  "is a placeholder ('')"),
            ("cells",["L0N2","must-be-pinned"],"has element 1 that is a placeholder ('must-be-pinned')"),
        ]
        for key,value,problem in cases:
            with self.subTest(key=key,value=value):
                table={**TABLE,key:value}
                expected=[f"X900: preregistration key {key} {problem}"]
                if key=="seeds":
                    expected.append(f"X900: preregistered seeds {value!r} are not the manifest's seeds [17, 29]")
                self.assert_blocked(self.tree(table=table),*expected)
        # A value without a canonical text, required or not, has no digest.
        outside="holds a character outside printable ASCII, whose escape depends on who serializes it"
        for key,value,problem in (
            ("rate",0.5,"is a float, which has no canonical text"),
            ("seeds",[17,"29"],"is a list that holds anything but only integers or only strings"),
            ("note","first segment \u2014 descriptive",f"is a string that {outside}"),
            ("programs",["rmw","set\top"],f"has element 1 that {outside}"),
        ):
            with self.subTest(key=key,value=value):
                table={**TABLE,key:value}
                expected=[f"X900: preregistration key {key} {problem}"]
                if key=="seeds":
                    expected+=[
                        "X900: preregistration key seeds has element 1 that must be an int, not a string",
                        "X900: preregistered seeds [17, '29'] are not the manifest's seeds [17, 29]",
                    ]
                self.assert_blocked(self.tree(table=table,digest="0"*64),*expected)

    def test_an_empty_required_list_blocks(self):
        self.assert_blocked(self.tree(table={**TABLE,"programs":[]}),"X900: preregistration key programs is an empty list")
        self.assert_blocked(
            self.tree(table={**TABLE,"seeds":[]}),
            "X900: preregistration key seeds is an empty list",
            "X900: preregistered seeds [] are not the manifest's seeds [17, 29]",
        )

    def test_a_digest_mismatch_blocks(self):
        # Without a baseline, the table is exactly TABLE.
        tree=lambda **changes:self.tree(baselines=(),baseline_config=None,**changes)
        digest=mod.experiment_records.preregistration_digest(TABLE)
        self.assert_blocked(tree(digest=None),"X900: experiment.toml names no preregistration_sha256")
        # The table changed after the manifest named its digest.
        changed={**TABLE,"schema":2}
        self.assert_blocked(
            tree(table=changed,digest=digest),
            f"X900: preregistration_sha256 {digest!r} is not {mod.experiment_records.preregistration_digest(changed)}, "
            "the digest of config.toml's [preregistration]",
        )
        for recorded in (digest.upper(),digest[:63],"",1):
            with self.subTest(recorded=recorded):
                self.assert_blocked(
                    tree(digest=recorded),
                    f"X900: preregistration_sha256 {recorded!r} is not {digest}, the digest of config.toml's [preregistration]",
                )

    def test_preregistered_seeds_must_be_the_manifests_seeds(self):
        # A float seed is not the integer the table froze, whatever == says.
        for seeds in ((29,17),(17,),(17,29,43),(17.0,29)):
            with self.subTest(seeds=seeds):
                self.assert_blocked(
                    self.tree(seeds=seeds),
                    f"X900: preregistered seeds [17, 29] are not the manifest's seeds {list(seeds)!r}",
                )
        # Every listed experiment preregisters its seeds, so none can be
        # added to the manifest alone after an outcome is seen.
        without={key:value for key,value in TABLE.items() if key!="seeds"}
        self.assert_blocked(self.tree(table=without,seeds=(17,29,43)),"X900: preregistration key seeds is missing")

    def test_an_unpinned_or_blocked_baseline_blocks(self):
        where=f"X900: baseline {BASELINE_PATH}"
        cases=[
            ({**BASELINE_CONFIG,"model":{"revision":""}},[f"{where}: model.revision is a placeholder ('')"]),
            ({**BASELINE_CONFIG,"model":{"revision":"must-be-pinned-before-run"}},[f"{where}: model.revision is a placeholder ('must-be-pinned-before-run')"]),
            ({**BASELINE_CONFIG,"model":{}},[f"{where}: model.revision is missing"]),
            ({**BASELINE_CONFIG,"model":{"revision":{"sha":"x"}}},[f"{where}: model.revision is a table, not a value"]),
            ({**BASELINE_CONFIG,"answer":{"generators":[]}},[f"{where}: answer.generators is an empty list"]),
            ({**BASELINE_CONFIG,"answer":{"generators":["g1","none"]}},[f"{where}: answer.generators has element 1 that is a placeholder ('none')"]),
            ({**BASELINE_CONFIG,"status":"blocked-unpinned-model-revisions"},[f"{where} is blocked-unpinned-model-revisions"]),
            ({**BASELINE_CONFIG,"status":"Blocked-by-licence"},[f"{where} is Blocked-by-licence"]),
            ({**BASELINE_CONFIG,"status":" blocked-x"},[f"{where} is  blocked-x"]),
            # A value where the key path needs a table has no such key.
            ({**BASELINE_CONFIG,"model":"revision-x"},[f"{where}: model.revision is missing"]),
            ({**BASELINE_CONFIG,"status":"must-be-pinned"},[f"{where}: status is not pinned ('must-be-pinned')"]),
            ({**BASELINE_CONFIG,"status":1},[f"{where}: status is not pinned (1)"]),
            ({key:value for key,value in BASELINE_CONFIG.items() if key!="status"},[f"{where}: status is missing"]),
            (None,[f"{where} is not a file in the repository"]),
        ]
        for config,errors in cases:
            with self.subTest(config=config):
                self.assert_blocked(self.tree(baseline_config=config),*errors)
        # A planned experiment's baseline is not checked.
        self.assertEqual(gate(self.tree(status="planned",baseline_config=None)),(0,[]))

    def test_the_list_names_only_registered_experiments_known_types_and_well_formed_baselines(self):
        where="experiments/preregistration.toml: X900"
        cases=[
            ({"required":{**REQUIRED,"seeds":"list"}},{},[f"{where}: key seeds has unknown type 'list'",f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs"]),
            ({"required":{**REQUIRED,"seeds":"int"}},{},[f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs"]),
            ({"required":{key:kind for key,kind in REQUIRED.items() if key!="seeds"}},{},[f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs"]),
            ({"required":{}},{},[f"{where} requires no keys"]),
            ({"baselines":({**BASELINE,"path":"/etc/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"path":"../outside.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            # git would read these as pathspec magic or globs, not as the path.
            ({"baselines":({**BASELINE,"path":":research/fixture/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"path":"research/fix*/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"path":"research/fixture[1]/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            # Its directory, which freezes with it, would be the repository.
            ({"baselines":({**BASELINE,"path":"config.toml"},)},{},[f"{where}: baseline 0 needs a configuration in a directory below the repository's root"]),
            ({"baselines":({**BASELINE,"keys":[]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({**BASELINE,"keys":["model.revision",""]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({key:value for key,value in BASELINE.items() if key!="status_key"},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({**BASELINE,"revision":"x"},)},{},[f"{where}: baseline 0 has unknown field revision"]),
            ({},{"entry_extra":"[experiment.X901.required]\nseeds = \"int-list\"\n"},["experiments/preregistration.toml: X901 is not a registered experiment"]),
            ({},{"entry_extra":"[experiment.X900.optional]\nschema = \"int\"\n"},[f"{where} has unknown field optional"]),
            ({"required":{**REQUIRED,"gr\u00f6\u00dfe":"int"}},{},[f"{where}: key 'gr\u00f6\u00dfe' is not printable ASCII"]),
            ({"required":{**REQUIRED,"seeds":["int"]}},{},[f"{where}: key seeds has unknown type ['int']",f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs"]),
            ({"required":{**REQUIRED,"seeds":1}},{},[f"{where}: key seeds has unknown type 1",f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs"]),
            ({"baselines":({**BASELINE,"keys":"model.revision"},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({**BASELINE,"keys":[1]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({**BASELINE,"keys":["model.r\u00e9vision"]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({**BASELINE,"status_key":""},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({**BASELINE,"status_key":1},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({key:value for key,value in BASELINE.items() if key!="name"},)},{},[f"{where}: baseline 0 needs a name of lowercase letters, digits and underscores"]),
            ({"baselines":({**BASELINE,"name":"Strong-RAG"},)},{},[f"{where}: baseline 0 needs a name of lowercase letters, digits and underscores"]),
            ({"baselines":({**BASELINE,"name":["fixture"]},)},{},[f"{where}: baseline 0 needs a name of lowercase letters, digits and underscores"]),
            ({"baselines":({**BASELINE,"name":{"a":1}},)},{},[f"{where}: baseline 0 needs a name of lowercase letters, digits and underscores"]),
            ({"baselines":(BASELINE,BASELINE)},{},[f"{where}: baseline 1 repeats the name 'fixture'"]),
        ]
        required=toml_table("experiment.X900.required",REQUIRED)
        baseline="".join(f"{key} = {toml_value(value)}\n" for key,value in BASELINE.items())
        listed=[
            # A misspelt top-level table would take its experiments out of the gate.
            ("version = 1\n"+required+'\n[experiments.X900.required]\nschema = "int"\n',["experiments/preregistration.toml has unknown field experiments"]),
            ('version = 1\n[experiments.X900.required]\nschema = "int"\n',["experiments/preregistration.toml has unknown field experiments"]),
            ('version = 1\n[[experiment]]\nid = "X900"\n',["experiments/preregistration.toml: experiment must be a table of experiments"]),
            ("version = 1\n[experiment]\nX900 = 1\n",["experiments/preregistration.toml: X900 is not a table"]),
            ('version = 1\n[experiment.X900]\nrequired = "int"\n',[f"{where}: required must be a table of keys and their types"]),
            ("version = 1\n"+required+"\n[experiment.X900.baseline]\n"+baseline,[f"{where}: baseline must be an array of tables"]),
            ("version = 1\n[experiment.X900]\nbaseline = [1]\n\n"+required,[f"{where}: baseline 0 is not a table"]),
        ]
        for status in ("planned","prepared"):
            for changes,extra,errors in cases:
                with self.subTest(status=status,changes=changes,extra=extra):
                    self.assert_blocked(self.tree(status=status,**changes,**extra),*errors)
            for text,errors in listed:
                with self.subTest(status=status,listed=text):
                    # X900 holds a complete preregistration, so only the list is wrong.
                    self.assert_blocked(self.tree(status=status,listed_text=text),*errors)
        # With the whole list misspelt, an experiment with nothing frozen
        # still fails the gate.
        self.assert_blocked(
            self.tree(table=None,digest=None,listed_text=listed[1][0]),
            "experiments/preregistration.toml has unknown field experiments",
        )

    def test_a_file_the_preregistration_names_is_frozen_by_its_content(self):
        required={**REQUIRED,"protocol":"file"}
        path="experiments/semdb/X900-fixture/PROTOCOL.md"
        text="# Protocol\nJudge the delta, the state before and after, and the verifier codes.\n"
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()

        def tree(table, content=text):
            root=self.tree(table=table,required=required)
            if content is not None:
                (root/path).write_bytes(content.encode("utf-8"))
            return root

        good={**TABLE,"protocol":path,"protocol_sha256":digest}
        self.assertEqual(gate(tree(good)),(0,[]))
        # Line endings are content: a copy that a converting checkout wrote
        # with CRLF ones is another file than the LF text frozen, which a
        # command reading it tells apart.
        crlf_text=text.replace("\n","\r\n")
        crlf_digest=hashlib.sha256(crlf_text.encode("utf-8")).hexdigest()
        self.assertNotEqual(crlf_digest,digest)
        root=tree(good)
        committed=commit_all(root)
        (root/path).write_bytes(crlf_text.encode("utf-8"))
        self.assert_blocked(
            root,
            f"X900: preregistration key protocol_sha256 {digest!r} is not {crlf_digest}, the digest of {path}",
        )
        # The same copy is no more the file the commit froze, and the commit
        # still freezes the LF text.
        self.assertEqual(mod.frozen_file_at(root,committed,path),text.encode("utf-8"))
        self.assertIsNotNone(mod.launchable_at(root,committed,"X900","experiments/semdb/X900-fixture"))
        # Committed with CRLF, the text is frozen with them, byte for byte: by
        # their own digest, and not by the digest of the LF text.
        crlf_table={**good,"protocol_sha256":crlf_digest}
        root=tree(crlf_table,content=crlf_text)
        self.assertEqual(gate(root),(0,[]))
        committed=commit_all(root)
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(mod.frozen_file_at(root,committed,path),crlf_text.encode("utf-8"))
        self.assertIsNotNone(mod.launchable_at(root,committed,"X900","experiments/semdb/X900-fixture"))
        (root/path).write_bytes(text.encode("utf-8"))
        self.assert_blocked(
            root,
            f"X900: preregistration key protocol_sha256 {crlf_digest!r} is not {digest}, the digest of {path}",
        )
        # A file is frozen by its content, and never as an executable one;
        # its carriage returns, in text or not, are content.
        self.assertIsNone(mod.file_form_problem("100644"))
        self.assertIsNotNone(mod.file_form_problem("100755"))
        # A protocol edited after the freeze no longer is the frozen one.
        edited=text+"Or do not.\n"
        self.assert_blocked(
            tree(good,content=edited),
            f"X900: preregistration key protocol_sha256 {digest!r} is not "
            f"{hashlib.sha256(edited.encode('utf-8')).hexdigest()}, the digest of {path}",
        )
        without={key:value for key,value in good.items() if key!="protocol_sha256"}
        self.assert_blocked(tree(without),f"X900: preregistration key protocol_sha256 is missing; it must be {digest}, the digest of {path}")
        self.assert_blocked(
            tree({**good,"protocol_sha256":"must-be-pinned-before-prepared"}),
            "X900: preregistration key protocol_sha256 is a placeholder ('must-be-pinned-before-prepared')",
        )
        # The path must name a file inside the repository.
        # Nor below git's own directory, whose files no commit holds.
        for named in (path,"../PROTOCOL.md","/etc/hostname","experiments/semdb/X900-fixture",".git/config",path+"/"):
            with self.subTest(named=named):
                content=None if named==path else text
                self.assert_blocked(
                    tree({**good,"protocol":named},content=content),
                    f"X900: preregistration key protocol names {named!r}, which is not a file in the repository",
                )
        self.assert_blocked(
            tree({**good,"protocol":"must-be-signed-by-owner"}),
            "X900: preregistration key protocol is a placeholder ('must-be-signed-by-owner')",
        )
        self.assert_blocked(tree({**good,"protocol":1}),"X900: preregistration key protocol must be a file, not an integer")

    def test_a_baseline_is_frozen_by_every_file_of_its_directory(self):
        directory=BASELINE_PATH.removesuffix("/config.toml")
        text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        frozen=baseline_digest({"config.toml":text})
        self.assertEqual(gate(self.tree(table={**TABLE,"baseline_fixture_sha256":frozen})),(0,[]))
        self.assert_blocked(
            self.tree(freeze_baselines=False),
            f"X900: preregistration key baseline_fixture_sha256 is missing; it must be {frozen}, the digest of every file in {directory}/",
        )
        # A baseline changed after the freeze, still pinned and unblocked, is
        # no longer the baseline the experiment froze, whether a pinned key or
        # a setting the list does not name changed.
        for config in (
            {**BASELINE_CONFIG,"model":{"revision":"4567def"}},
            {**BASELINE_CONFIG,"answer":{"generators":["g2","g1"]}},
            {**BASELINE_CONFIG,"retrieval":{"top_k":50}},
            {**BASELINE_CONFIG,"status":"reference-implemented-v2"},
        ):
            with self.subTest(config=config):
                root=self.tree(table={**TABLE,"baseline_fixture_sha256":frozen},baseline_config=config)
                moved=baseline_digest({"config.toml":(root/BASELINE_PATH).read_text(encoding="utf-8")})
                self.assert_blocked(
                    root,
                    f"X900: preregistration key baseline_fixture_sha256 {frozen!r} is not {moved}, the digest of every file in {directory}/",
                )
        # The implementation beside the configuration is frozen with it:
        # a file added to the directory, committed or not, or one edited after
        # the freeze, is another baseline.
        code="def rank(query):\n    return []\n"
        with_code=baseline_digest({"config.toml":text,"runner.py":code})
        root=self.tree(table={**TABLE,"baseline_fixture_sha256":frozen})
        write(root,f"{directory}/runner.py",code)
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {frozen!r} is not {with_code}, the digest of every file in {directory}/",
        )
        commit_all(root)
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {frozen!r} is not {with_code}, the digest of every file in {directory}/",
        )
        root=self.tree(table={**TABLE,"baseline_fixture_sha256":with_code})
        write(root,f"{directory}/runner.py",code)
        self.assertEqual(gate(root),(0,[]))
        write(root,f"{directory}/lib/hybrid.py","WEIGHT = 1\n")
        nested=baseline_digest({"config.toml":text,"runner.py":code,"lib/hybrid.py":"WEIGHT = 1\n"})
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {with_code!r} is not {nested}, the digest of every file in {directory}/",
        )
        (root/directory/"lib/hybrid.py").unlink()
        self.assertEqual(gate(root),(0,[]))
        # A committed file removed from the tree is gone from the baseline.
        commit_all(root)
        (root/directory/"runner.py").unlink()
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {with_code!r} is not {frozen}, the digest of every file in {directory}/",
        )
        write(root,f"{directory}/runner.py",code)
        # What the .gitignore files ignore is refused rather than left out:
        # an interpreter can run a bytecode cache in place of runner.py, and
        # git holds none. A file of a directory beside it is no file of it.
        write(root,".gitignore","__pycache__/\n")
        write(root,f"{directory}/__pycache__/runner.cpython-313.pyc","cached")
        write(root,f"{directory}-other/config.toml","x = 1\n")
        self.assert_blocked(
            root,
            f"X900: baseline {directory}/ holds {directory}/__pycache__/runner.cpython-313.pyc, which git ignores; "
            "a baseline's directory holds only what git tracks or would track",
        )
        shutil.rmtree(root/directory/"__pycache__")
        self.assertEqual(gate(root),(0,[]))
        # An ignored link is refused once, as what git ignores: the listing of
        # what the baseline holds leaves out what its .gitignore files ignore.
        write(root,".gitignore","__pycache__/\ncache.lnk\n")
        self.link(Path("config.toml"),root/directory/"cache.lnk")
        self.assert_blocked(
            root,
            f"X900: baseline {directory}/ holds {directory}/cache.lnk, which git ignores; "
            "a baseline's directory holds only what git tracks or would track",
        )
        (root/directory/"cache.lnk").unlink()
        self.assertEqual(gate(root),(0,[]))
        # A file's mode is frozen with its content, as git holds it: made
        # executable, runner.py makes another baseline, in the tree and in a
        # commit.
        # Git's mode counts for a tracked file, as a checkout that cannot
        # show modes still holds it.
        git(root,"update-index","--chmod=+x",f"{directory}/runner.py")
        executable=baseline_digest({"config.toml":text,"runner.py":code},executable={"runner.py"})
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {with_code!r} is not {executable}, the digest of every file in {directory}/",
        )
        (root/directory/"runner.py").chmod(0o755)
        chmodded=commit_all(root,"runner.py executable")
        self.assertEqual(mod.directory_digest_at(root,chmodded,directory),executable)
        # Line endings are content: a copy a converting checkout wrote with
        # CRLF ones is another baseline than the LF one frozen, in the tree
        # and not at the commit, which still holds the LF one.
        crlf_text=text.replace("\n","\r\n")
        crlf_frozen=baseline_digest({"config.toml":crlf_text})
        self.assertNotEqual(crlf_frozen,frozen)
        root=self.tree(table={**TABLE,"baseline_fixture_sha256":frozen})
        lf=commit_all(root,"LF")
        (root/BASELINE_PATH).write_bytes(crlf_text.encode("utf-8"))
        self.assert_blocked(
            root,
            f"X900: preregistration key baseline_fixture_sha256 {frozen!r} is not {crlf_frozen}, the digest of every file in {directory}/",
        )
        self.assertEqual(mod.directory_digest_at(root,lf,directory),frozen)
        (root/BASELINE_PATH).write_bytes(text.encode("utf-8"))
        self.assertEqual(gate(root),(0,[]))
        # Committed with CRLF, a text file of it is frozen with them, byte for
        # byte: by their own digest, and not by the digest of the LF text.
        crlf_root=self.tree(table={**TABLE,"baseline_fixture_sha256":crlf_frozen})
        (crlf_root/BASELINE_PATH).write_bytes(crlf_text.encode("utf-8"))
        self.assertEqual(gate(crlf_root),(0,[]))
        crlf_commit=commit_all(crlf_root,"CRLF")
        self.assertEqual(gate(crlf_root),(0,[]))
        self.assertEqual(mod.directory_digest_at(crlf_root,crlf_commit,directory),crlf_frozen)
        (crlf_root/BASELINE_PATH).write_bytes(text.encode("utf-8"))
        self.assert_blocked(
            crlf_root,
            f"X900: preregistration key baseline_fixture_sha256 {crlf_frozen!r} is not {frozen}, the digest of every file in {directory}/",
        )
        # A file that is not text keeps its carriage returns as content.
        write(root,f"{directory}/table.bin","")
        (root/directory/"table.bin").write_bytes(b"\x00\r\n")
        binary=commit_all(root,"binary")
        self.assertIsNotNone(mod.directory_digest_at(root,binary,directory))
        self.assert_blocked(
            self.tree(table={**TABLE,"baseline_fixture_sha256":"must-be-pinned-before-prepared"}),
            "X900: preregistration key baseline_fixture_sha256 is a placeholder ('must-be-pinned-before-prepared')",
        )

    RECORD="experiments/semdb/X900-fixture/results/run-20260101T000000.000000Z-seed-17.json"
    AGGREGATE="experiments/semdb/X900-fixture/results/run.json"
    METRICS="experiments/semdb/X900-fixture/results/metrics.json"
    MANIFEST="experiments/semdb/X900-fixture/experiment.toml"

    def frozen(self, root: Path) -> tuple[str, str]:
        """The digests the fixture's manifest names: the table's and the rules'."""
        manifest=mod.load(root/self.MANIFEST)
        return manifest["preregistration_sha256"],manifest["preregistration_rules_sha256"]

    def record(self, root: Path, commit: str, **changes) -> dict:
        """A run record as run_experiment.py writes one at `commit`; a
        `completed` or `failed` one holds the outcome such a run writes
        (`experiment_records.finished_run`) unless `changes` name another."""
        digest,rules=self.frozen(root)
        record={
            "git_sha":commit,
            "manifest":{"id":"X900","preregistration_sha256":digest,"preregistration_rules_sha256":rules},
            "manifest_sha256":hashlib.sha256(git(root,"show",f"{commit}:{self.MANIFEST}").encode("utf-8")+b"\n").hexdigest(),
        }
        record.update(changes)
        seed=record.get("seed")
        if not isinstance(seed,bool) and isinstance(seed,int):
            # What the runner builds from the fixture's entrypoint, `bench <seed>`.
            record.setdefault("command",["bench",str(seed)])
            record.setdefault("parameters",{})
        if record.get("status") in ("completed","failed"):
            record.setdefault("exit_code",0 if record["status"]=="completed" else 1)
            record.setdefault("finished_at","2026-01-01T00:00:00+00:00")
            record.setdefault("stdout","")
            record.setdefault("stderr","")
        return record

    def aggregate(self, root: Path, commit: str, **changes) -> dict:
        """An aggregate run.json of the runs at `commit`, naming the SHA-256
        of the record of each seed the results directory holds a run of and
        of the `{}` its metrics.json holds (`METRICS`)."""
        digest,rules=self.frozen(root)
        results=root/"experiments/semdb/X900-fixture/results"
        changes.setdefault("metrics_sha256",hashlib.sha256(b"{}").hexdigest())
        if "seed_records" not in changes:
            changes["seed_records"]=mod.experiment_records.seed_record_digests(results) if results.is_dir() else {}
        return {"git_sha":commit,"preregistration_sha256":digest,"preregistration_rules_sha256":rules,**changes}

    def publish(self, root: Path, report: dict) -> None:
        """Write `report`, an aggregate, and the `{}` its metrics.json holds,
        as one aggregate (`aggregate` names that content's digest)."""
        write(root,self.METRICS,"{}")
        write(root,self.AGGREGATE,json.dumps(report))

    def seeded(self, root: Path, commit: str) -> list[str]:
        """Write the run record of each preregistered seed at `commit` into
        the results directory, as the runner writes them, for an aggregate to
        bind; their names, as the gate shows them."""
        names=[]
        for seed in TABLE["seeds"]:
            name=f"results/run-2026010{seed % 9}T000000.000000Z-seed-{seed}.json"
            write(root,f"experiments/semdb/X900-fixture/{name}",
                  json.dumps(self.record(root,commit,seed=seed,status="completed")))
            names.append(name)
        return names

    def test_archived_runs_must_name_the_frozen_digests_and_a_commit_that_holds_them(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                root=self.tree(status=status)
                commit=commit_all(root)
                short=commit[:12]
                digest,rules=self.frozen(root)
                write(root,self.RECORD,json.dumps(self.record(root,commit)))
                self.seeded(root,commit)
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,commit)))
                # Files other than run records are not run records.
                write(root,"experiments/semdb/X900-fixture/results/metrics.json","{}")
                write(root,"experiments/semdb/X900-fixture/results/run_notes.json","[]")
                self.assertEqual(gate(root),(0,[]))
                # A manifest hashed with CRLF line endings, as a converting
                # checkout would read it, is not the committed manifest.
                crlf=hashlib.sha256(git(root,"show",f"{commit}:{self.MANIFEST}").replace("\n","\r\n").encode("utf-8")).hexdigest()
                write(root,self.RECORD,json.dumps(self.record(root,commit,manifest_sha256=crlf)))
                self.assert_blocked(
                    root,
                    f"X900: {name} names manifest_sha256 {crlf!r}, not the SHA-256 of experiment.toml at {short}",
                )
                write(root,self.RECORD,json.dumps(self.record(root,commit)))
                self.assertEqual(gate(root),(0,[]))
                write(root,self.RECORD,json.dumps(self.record(root,commit),indent=1))
                commit_all(root,"records")
                self.assertEqual(gate(root),(0,[]))
                cases=[
                    (self.record(root,commit,manifest={"preregistration_sha256":"0"*64,"preregistration_rules_sha256":rules}),
                     [f"X900: {name} names preregistration_sha256 {'0'*64!r}, not {digest}, the digest the experiment is frozen at"]),
                    (self.record(root,commit,manifest={"preregistration_sha256":digest}),
                     [f"X900: {name} names preregistration_rules_sha256 None, not {rules}, the digest the experiment is frozen at"]),
                    (self.record(root,commit,git_sha=None),
                     [f"X900: {name} names no commit this repository holds (None)"]),
                    (self.record(root,commit,git_sha="f"*40),
                     [f"X900: {name} names no commit this repository holds ({'f'*40!r})"]),
                    (self.record(root,commit,manifest_sha256="0"*64),
                     [f"X900: {name} names manifest_sha256 {'0'*64!r}, not the SHA-256 of experiment.toml at {short}"]),
                ]
                for record,errors in cases:
                    with self.subTest(status=status,record=record):
                        # Edited in the working tree, the committed record differs too.
                        write(root,self.RECORD,json.dumps(record))
                        self.assert_blocked(root,*errors,f"X900: {name} differs from the record committed as it")
                write(root,self.RECORD,json.dumps(self.record(root,commit),indent=1))
                self.assertEqual(gate(root),(0,[]))
                # A record the index tells git to take as HEAD's, which git
                # diff and status then do not look at, is read from disk.
                kept=(root/self.RECORD).read_bytes()
                for flag in ("skip-worktree","assume-unchanged"):
                    with self.subTest(status=status,flag=flag):
                        git(root,"update-index",f"--{flag}",self.RECORD)
                        write(root,self.RECORD,json.dumps({**self.record(root,commit),"stdout":"another outcome\n"}))
                        self.assert_blocked(root,f"X900: {name} differs from the record committed as it")
                        (root/self.RECORD).write_bytes(kept)
                        git(root,"update-index",f"--no-{flag}",self.RECORD)
                self.assertEqual(gate(root),(0,[]))
                # Line endings are content: a copy a converting checkout wrote
                # with CRLF ones, or with any other carriage return, is
                # another record than the one committed with LF ones.
                for other in (b"\r\n",b"\r"):
                    with self.subTest(status=status,line_ending=other):
                        (root/self.RECORD).write_bytes(kept.replace(b"\n",other))
                        self.assert_blocked(root,f"X900: {name} differs from the record committed as it")
                (root/self.RECORD).write_bytes(kept)
                for text,errors in (
                    (json.dumps({**self.aggregate(root,commit),"preregistration_sha256":None}),
                     [f"X900: results/run.json names preregistration_sha256 None, not {digest}, the digest the experiment is frozen at"]),
                    (json.dumps(self.aggregate(root,commit,git_sha="f"*40)),
                     [f"X900: results/run.json names no commit this repository holds ({'f'*40!r})"]),
                    ("[]",["X900: results/run.json is not a JSON object"]),
                ):
                    with self.subTest(status=status,aggregate=text):
                        write(root,self.AGGREGATE,text)
                        self.assert_blocked(root,*errors)
                write(root,self.AGGREGATE,"{")
                code,lines=gate(root)
                self.assertEqual((code,len(lines)),(1,1))
                self.assertTrue(lines[0].startswith("X900: results/run.json cannot be read: "),lines)

    def test_a_record_names_the_digest_of_the_manifest_bytes_as_committed_whatever_their_line_endings(self):
        # `git()` reads text with universal newlines, so the committed bytes
        # are read here as they are: a manifest committed with CRLF line
        # endings is hashed with them, one committed with LF endings without.
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        for ending,other in ((b"\r\n",b"\n"),(b"\n",b"\r\n")):
            with self.subTest(committed=ending):
                root=self.tree(status="running")
                converted=(root/self.MANIFEST).read_bytes().replace(b"\r\n",b"\n").replace(b"\n",ending)
                (root/self.MANIFEST).write_bytes(converted)
                commit=commit_all(root)
                short=commit[:12]
                held=subprocess.run(
                    ["git","-C",str(root),"show",f"{commit}:{self.MANIFEST}"],check=True,capture_output=True
                ).stdout
                self.assertEqual(held,converted)
                self.assertEqual(held.count(b"\n"),held.count(ending))
                named=hashlib.sha256(held).hexdigest()
                write(root,self.RECORD,json.dumps(self.record(root,commit,manifest_sha256=named)))
                self.assertEqual(gate(root),(0,[]))
                # The same text with the other line endings, whole and
                # including its last line ending, is another file.
                unlike=held.replace(b"\r\n",b"\n").replace(b"\n",other)
                self.assertNotEqual(hashlib.sha256(unlike).hexdigest(),named)
                wrong=hashlib.sha256(unlike).hexdigest()
                write(root,self.RECORD,json.dumps(self.record(root,commit,manifest_sha256=wrong)))
                self.assert_blocked(
                    root,
                    f"X900: {name} names manifest_sha256 {wrong!r}, not the SHA-256 of experiment.toml at {short}",
                )

    def test_a_preregistration_rewritten_after_its_runs_fails_whatever_else_is_rewritten(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        self.publish(root,self.aggregate(root,ran))
        recorded=commit_all(root,"records")
        before=self.frozen(root)
        # The table, the manifest and both records are rewritten to a new
        # preregistration, and committed.
        rewritten=self.tree(status="running",table={**TABLE,"schema":2})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        after=self.frozen(root)
        self.assertNotEqual(before,after)
        record=self.record(root,ran)
        record["manifest"]={**record["manifest"],"preregistration_sha256":after[0]}
        write(root,self.RECORD,json.dumps(record))
        self.publish(root,{**self.aggregate(root,ran),"preregistration_sha256":after[0]})
        rewrite=commit_all(root,"rewrite")
        short=ran[:12]
        self.assert_blocked(
            root,
            f"X900: {name} ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: {name} ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900: {name} was changed after it was committed (2 versions of it were committed)",
            # The commit the runs were made at froze the old preregistration.
            f"X900 was frozen at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900: results/run.json ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: results/run.json ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
            # It reports on seeds no run record holds.
            "X900: results/run.json reports on seed 17, which has no run record",
            "X900: results/run.json reports on seed 29, which has no run record",
            # The version now on disk saw no run record in its own commit either.
            f"X900: results/run.json as committed at {rewrite[:12]} reports on seed 17, which has no run record",
            f"X900: results/run.json as committed at {rewrite[:12]} reports on seed 29, which has no run record",
            # The aggregate as it was first committed saw the outcome under
            # the old preregistration.
            f"X900: results/run.json as committed at {recorded[:12]} names preregistration_sha256 {before[0]!r}, not {after[0]}, "
            "the digest the experiment is frozen at",
            f"X900: results/run.json as committed at {recorded[:12]} ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: results/run.json as committed at {recorded[:12]} ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
            # And it reported on the same seeds, which no run record holds.
            f"X900: results/run.json as committed at {recorded[:12]} reports on seed 17, which has no run record",
            f"X900: results/run.json as committed at {recorded[:12]} reports on seed 29, which has no run record",
        )

    def test_a_run_record_deleted_or_renamed_after_it_was_committed_fails(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        self.assertEqual(gate(root),(0,[]))
        # Renamed: the record is removed and put back under another name.
        renamed=self.RECORD.replace("seed-17","seed-17-again")
        git(root,"mv",self.RECORD,renamed)
        commit_all(root,"renamed")
        self.assert_blocked(root,f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded")
        # Deleted with the whole results directory, the record still fails.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        git(root,"rm","-q","-r","experiments/semdb/X900-fixture/results")
        commit_all(root,"deleted")
        self.assert_blocked(root,f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded")
        # Committed on a side branch and dropped by the merge that brings the
        # branch in, the record was committed all the same.
        root=self.tree(status="running")
        ran=commit_all(root)
        git(root,"checkout","-q","-b","side")
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        git(root,"checkout","-q","main")
        git(root,"merge","-q","-s","ours","--no-edit","side")
        self.assertFalse((root/self.RECORD).exists())
        self.assert_blocked(root,f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded")
        # Added by one merge and dropped by another, with no commit of its own,
        # the record was committed all the same.
        root=self.tree(status="running")
        ran=commit_all(root)
        for branch,change in (("adds",lambda: write(root,self.RECORD,json.dumps(self.record(root,ran)))),
                              ("drops",lambda: git(root,"rm","-q",self.RECORD))):
            git(root,"checkout","-q","-b",branch)
            write(root,f"notes-{branch}.md",f"{branch}\n")
            commit_all(root,branch)
            git(root,"checkout","-q","main")
            git(root,"merge","-q","--no-ff","--no-commit",branch)
            change()
            commit_all(root,f"merge {branch}")
        self.assertFalse((root/self.RECORD).exists())
        self.assert_blocked(root,f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded")
        # A record that was never committed may be discarded: it holds no
        # outcome the history saw.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        (root/self.RECORD).unlink()
        self.assertEqual(gate(root),(0,[]))

    def ran(self, status="running", seed=None, **tree) -> tuple[Path, str]:
        """A fixture tree at `status` with one run record committed (of
        `seed`, when given, whose run finished), and the commit the record
        names."""
        root=self.tree(status=status,**tree)
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran,**({} if seed is None else {"seed":seed,"status":"completed"}))))
        commit_all(root,"records")
        self.assertEqual(gate(root),(1 if unarchived(status) else 0,unarchived(status)))
        return root,ran

    def edit(self, root: Path, relative: str, old: str, new: str) -> None:
        """Replace `old`, which `relative` holds once, with `new`."""
        text=(root/relative).read_text(encoding="utf-8")
        self.assertEqual(text.count(old),1,(relative,old))
        write(root,relative,text.replace(old,new))

    def test_after_a_run_the_manifest_and_configuration_change_only_in_status(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        config="experiments/semdb/X900-fixture/config.toml"
        # A manifest key outside the preregistration, such as the criterion
        # the outcome is judged by, changed after the run.
        root,ran=self.ran()
        at=f"X900: {name} ran at {ran[:12]}"
        froze=f"X900 was frozen at {ran[:12]}"
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'falsification = "any outcome"\n')
        changed=[
            f"{at}, whose experiment.toml differs from the current one in falsification; after a run only its status changes",
            f"{froze}, whose experiment.toml differs from the current one in falsification; after a run only its status changes",
        ]
        self.assert_blocked(root,*changed)
        commit_all(root,"criterion")
        self.assert_blocked(root,*changed)
        # So is a setting of config.toml outside its [preregistration] table.
        root,ran=self.ran()
        at=f"X900: {name} ran at {ran[:12]}"
        self.edit(root,config,'tests_dir = "tests"','tests_dir = "checks"')
        self.assert_blocked(
            root,
            f"{at}, whose config.toml differs from the current one in tests_dir; after a run the configuration stays as it ran",
            f"X900 was frozen at {ran[:12]}, whose config.toml differs from the current one in tests_dir; after a run the configuration stays as it ran",
        )
        # The aggregate is bound the same way; it names the record of the
        # one seed the experiment preregisters.
        root,ran=self.ran(table={**TABLE,"seeds":[17]},seeds=(17,),seed=17)
        at=f"X900: {name} ran at {ran[:12]}"
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate")
        self.assertEqual(gate(root),(0,[]))
        self.edit(root,self.MANIFEST,"required_artifacts = []","required_artifacts = []\nhypothesis = \"another\"")
        self.assert_blocked(
            root,
            f"{at}, whose experiment.toml differs from the current one in hypothesis; after a run only its status changes",
            f"X900: results/run.json ran at {ran[:12]}, whose experiment.toml differs from the current one in hypothesis; after a run only its status changes",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml differs from the current one in hypothesis; after a run only its status changes",
        )

    def test_after_a_run_the_status_only_moves_forward(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        forward={
            "prepared":("prepared","running","completed","failed"),
            "running":("running","completed","failed"),
            "completed":("completed",),
            "failed":("failed",),
        }
        for then,allowed in forward.items():
            for now in ("prepared","running","completed","failed"):
                with self.subTest(then=then,now=now):
                    root,ran=self.ran(status=then)
                    self.edit(root,self.MANIFEST,f'status = "{then}"',f'status = "{now}"')
                    self.edit(root,"experiments/registry.toml",f'status = "{then}"',f'status = "{now}"') if now!=then else None
                    if now in allowed:
                        self.assertEqual(gate(root),(1 if unarchived(now) else 0,unarchived(now)))
                    else:
                        self.assert_blocked(
                            root,
                            f"X900: {name} ran at {ran[:12]}, where it was {then!r}; it cannot be {now!r} after that, "
                            "since a status moves only from prepared to running to completed or failed, or to superseded",
                            f"X900 was frozen at {ran[:12]}, where it was {then!r}; it cannot be {now!r} after that, "
                            "since a status moves only from prepared to running to completed or failed, or to superseded",
                            *unarchived(now),
                            *(superseded_unarchived(ran,now) if then=="completed" and now!="completed" else []),
                        )
        # Back at planned, the experiment is not gated as frozen, but it has
        # run: its committed records keep it from hiding as never run.
        root,ran=self.ran()
        self.edit(root,self.MANIFEST,'status = "running"','status = "planned"')
        self.assert_blocked(
            root,
            f"X900 is 'planned', but run records of it were committed ({self.RECORD}); a listed experiment that has run "
            "stays prepared, running, completed or failed, or is superseded",
        )
        # Superseded, another experiment replaced it: it is not launched, and
        # it stays as it ran.
        self.edit(root,self.MANIFEST,'status = "planned"','status = "superseded"')
        self.edit(root,"experiments/registry.toml",'status = "running"','status = "superseded"')
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(mod.launch_errors(root,"X900"),[
            "X900 preregisters (experiments/preregistration.toml) and is 'superseded': it runs only once its "
            "preregistration is frozen and it is prepared, running, completed or failed",
        ])
        # A record whose commit held the experiment before it was prepared
        # did not come from the runner, which refuses to launch it there.
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'status = "running"','status = "planned"')
        self.edit(root,"experiments/registry.toml",'status = "running"','status = "planned"')
        early=commit_all(root)
        self.edit(root,self.MANIFEST,'status = "planned"','status = "running"')
        self.edit(root,"experiments/registry.toml",'status = "planned"','status = "running"')
        commit_all(root,"prepared")
        write(root,self.RECORD,json.dumps(self.record(root,early)))
        self.assert_blocked(root,f"X900: {name} ran at {early[:12]}, where it was 'planned'; a listed experiment runs only once it is prepared")

    def test_a_superseded_experiment_that_ran_stays_as_it_ran(self):
        # Superseding stops the runner, not the binding: an experiment that
        # ran, superseded, keeps its records and its preregistration, so no
        # outcome can be erased by superseding it.
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        for then in ("running","completed","failed"):
            with self.subTest(then=then):
                root,ran=self.ran(status=then)
                self.edit(root,self.MANIFEST,f'status = "{then}"','status = "superseded"')
                self.edit(root,"experiments/registry.toml",f'status = "{then}"','status = "superseded"')
                commit_all(root,"superseded")
                # One that was completed keeps the aggregate it was completed
                # with, which this fixture never had.
                lost=superseded_unarchived(ran) if then=="completed" else []
                self.assertEqual(gate(root),(1 if lost else 0,lost))
                (root/self.RECORD).unlink()
                rewritten=self.tree(status="superseded",table={**TABLE,"schema":2})
                for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
                    shutil.copyfile(rewritten/relative,root/relative)
                commit_all(root,"record deleted and criteria rewritten")
                code,lines=gate(root)
                self.assertEqual(code,1)
                for line in (
                    f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded",
                    f"X900 was frozen at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
                ):
                    self.assertIn(line,lines)
        # One superseded before it ever left planned was never bound.
        root=self.tree(status="planned",table={**TABLE,"harness":"must-be-pinned-before-prepared"})
        commit_all(root)
        self.edit(root,self.MANIFEST,'status = "planned"','status = "superseded"')
        commit_all(root,"superseded")
        self.assertEqual(gate(root),(0,[]))

    def test_only_a_results_directory_holds_run_records(self):
        # A file named like a record elsewhere in the experiment's directory,
        # such as a test's fixture, is none: the runner writes records only
        # into a results directory a committed manifest declared.
        root,ran=self.ran()
        for name in ("tests/run-example.json","tests/run.json"):
            write(root,f"experiments/semdb/X900-fixture/{name}",'{"fixture": true}\n')
        commit_all(root,"fixtures")
        self.assertEqual(gate(root),(0,[]))
        (root/"experiments/semdb/X900-fixture/tests/run-example.json").unlink()
        commit_all(root,"a fixture removed")
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(mod.committed_records(root,["experiments/semdb/X900-fixture"]),[self.RECORD])

    def test_run_records_are_found_wherever_the_experiment_kept_them(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        # A record kept in a results directory a committed manifest named is
        # found once the manifest names another, and deleted, it is missed.
        root=self.tree(status="running")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "out"\n')
        ran=commit_all(root)
        kept=self.RECORD.replace("/results/","/out/")
        write(root,kept,json.dumps(self.record(root,ran)))
        commit_all(root,"record")
        self.assertIn(kept,mod.committed_records(root,["experiments/semdb/X900-fixture"]))
        (root/kept).unlink()
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8").replace('results_dir = "out"\n',""))
        commit_all(root,"record deleted, results_dir dropped")
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(f"X900: out/{name.removeprefix('results/')} was committed and has since been deleted or renamed; "
                      "a run record stays as it was recorded",lines)
        # The results directory moved after the run, and a rewritten
        # preregistration recorded in the new one: the old record is still
        # checked, where it was committed.
        root,ran=self.ran()
        before=self.frozen(root)
        rewritten=self.tree(status="running",table={**TABLE,"schema":2})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "new-results"\n')
        after=self.frozen(root)
        rewrite=commit_all(root,"rewrite")
        write(root,self.RECORD.replace("/results/","/new-results/"),json.dumps(self.record(root,rewrite)))
        commit_all(root,"new record")
        at=f"X900: {name} ran at {ran[:12]}"
        self.assert_blocked(
            root,
            f"X900: {name} names preregistration_sha256 {before[0]!r}, not {after[0]}, the digest the experiment is frozen at",
            f"{at}, whose config.toml holds another [preregistration] than the frozen one",
            f"{at}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"{at}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
            f"X900 was frozen at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
        )
        # Moved into the new results directory, the record is gone from
        # where it was committed.
        root,ran=self.ran()
        git(root,"mv","experiments/semdb/X900-fixture/results","experiments/semdb/X900-fixture/new-results")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "new-results"\n')
        commit_all(root,"moved")
        self.assert_blocked(
            root,
            f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded",
            f"X900: new-results/{name.removeprefix('results/')} ran at {ran[:12]}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
        )
        # The experiment's directory moved in the registry, its records with
        # it: the records are gone from where they were committed, and the
        # commit they ran at held no experiment where it is now.
        root,ran=self.ran()
        recorded=json.loads((root/self.RECORD).read_text(encoding="utf-8"))["manifest_sha256"]
        git(root,"mv","experiments/semdb/X900-fixture","experiments/semdb/X900-moved")
        self.edit(root,"experiments/registry.toml",'path = "semdb/X900-fixture"','path = "semdb/X900-moved"')
        commit_all(root,"moved")
        at=f"X900: {name} ran at {ran[:12]}"
        self.assert_blocked(
            root,
            f"X900: {self.RECORD} was committed and has since been deleted or renamed; a run record stays as it was recorded",
            f"{at}, whose config.toml holds another [preregistration] than the frozen one",
            f"{at}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900: {name} names manifest_sha256 {recorded!r}, not the SHA-256 of experiment.toml at {ran[:12]}",
            # Where it was frozen, the commit held no experiment where it is now.
            f"X900 was frozen at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"{at}, where the registry placed it in experiments/semdb/X900-fixture, not in experiments/semdb/X900-moved",
            f"X900 was frozen at {ran[:12]}, where the registry placed it in experiments/semdb/X900-fixture, not in experiments/semdb/X900-moved",
        )
        # Back at planned there, the records committed where the experiment
        # was are still its records.
        self.edit(root,"experiments/semdb/X900-moved/experiment.toml",'status = "running"','status = "planned"')
        self.assert_blocked(
            root,
            f"X900 is 'planned', but run records of it were committed ({self.RECORD}); a listed experiment that has run "
            "stays prepared, running, completed or failed, or is superseded",
        )
        # Moved in the registry to a twin of its directory, planted before the
        # run with another setting, the experiment ran elsewhere all the same.
        root=self.tree(status="running")
        shutil.copytree(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900-twin")
        self.edit(root,"experiments/semdb/X900-twin/config.toml",'tests_dir = "tests"','tests_dir = "checks"')
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        self.assertEqual(gate(root),(0,[]))
        self.edit(root,"experiments/registry.toml",'path = "semdb/X900-fixture"','path = "semdb/X900-twin"')
        commit_all(root,"moved to the twin")
        at=f"X900: experiments/semdb/X900-fixture/results/{name.removeprefix('results/')} ran at {ran[:12]}"
        self.assert_blocked(
            root,
            f"{at}, where the registry placed it in experiments/semdb/X900-fixture, not in experiments/semdb/X900-twin",
            f"X900 was frozen at {ran[:12]}, where the registry placed it in experiments/semdb/X900-fixture, not in experiments/semdb/X900-twin",
        )
        # A results directory outside the experiment's directory, or the
        # experiment's directory itself, is refused: the runner holds the
        # rest of that directory to HEAD.
        # Nor one that begins with a drive letter and a colon, which Windows
        # reads as a path on that drive.
        for results_dir in ("../shared",".","./","C:/out","c:out","C:"):
            with self.subTest(results_dir=results_dir):
                root=self.tree()
                write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+f'results_dir = "{results_dir}"\n')
                self.assert_blocked(root,f"X900: results_dir {results_dir!r} is not a directory below the experiment's directory")

    def test_a_run_record_names_a_commit_on_heads_history(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        commit_all(root)
        git(root,"checkout","-q","-b","side")
        write(root,"notes.md","side\n")
        side=commit_all(root,"side")
        git(root,"checkout","-q","main")
        write(root,self.RECORD,json.dumps(self.record(root,side)))
        commit_all(root,"records")
        self.assert_blocked(root,f"X900: {name} ran at {side[:12]}, which is not on HEAD's history")
        # Merged, it is.
        git(root,"merge","-q","--no-edit","side")
        self.assertEqual(gate(root),(0,[]))

    def test_an_aggregate_is_bound_in_every_version_it_was_committed_in(self):
        # The aggregate was committed after the runs; it shows the outcome.
        root=self.tree(status="running")
        ran=commit_all(root)
        runs=self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        recorded=commit_all(root,"aggregate")
        self.assertEqual(gate(root),(0,[]))
        before=self.frozen(root)
        # The preregistration and manifest are rewritten and the aggregate
        # written again, in place, from a run at the rewritten commit.
        rewritten=self.tree(status="running",table={**TABLE,"schema":2})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        after=self.frozen(root)
        rewrite=commit_all(root,"rewrite")
        self.publish(root,self.aggregate(root,rewrite))
        commit_all(root,"aggregate again")
        at=f"X900: results/run.json as committed at {recorded[:12]}"
        earlier=[
            f"{at} names preregistration_sha256 {before[0]!r}, not {after[0]}, the digest the experiment is frozen at",
            f"{at} ran at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"{at} ran at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900 was frozen at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        ]
        # The runs it reports on were made under the old preregistration too.
        for run in runs:
            earlier+=[
                f"X900: {run} names preregistration_sha256 {before[0]!r}, not {after[0]}, the digest the experiment is frozen at",
                f"X900: {run} ran at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
                f"X900: {run} ran at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
            ]
        self.assert_blocked(root,*earlier)
        # Written again in a new results directory instead, the old
        # aggregate is still checked where it was committed.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.publish(root,self.aggregate(root,ran))
        old_aggregate=commit_all(root,"aggregate")
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "new-results"\n')
        rewrite=commit_all(root,"rewrite")
        write(root,self.AGGREGATE.replace("/results/","/new-results/"),json.dumps(self.aggregate(root,rewrite)))
        write(root,self.METRICS.replace("/results/","/new-results/"),"{}")
        new_aggregate=commit_all(root,"new aggregate")
        at=f"X900: results/run.json ran at {ran[:12]}"
        froze=f"X900 was frozen at {ran[:12]}"
        self.assert_blocked(
            root,
            f"X900: results/run.json names preregistration_sha256 {before[0]!r}, not {after[0]}, the digest the experiment is frozen at",
            f"{at}, whose config.toml holds another [preregistration] than the frozen one",
            f"{at}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"{at}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
            f"{froze}, whose config.toml holds another [preregistration] than the frozen one",
            f"{froze}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"{froze}, whose experiment.toml differs from the current one in results_dir; after a run only its status changes",
            # Neither aggregate has a run record to name, in the commit it was
            # committed in either.
            *(f"X900: {aggregate} reports on seed {seed}, which has no run record"
              for aggregate in ("results/run.json","new-results/run.json") for seed in TABLE["seeds"]),
            *(f"X900: {aggregate} as committed at {commit[:12]} reports on seed {seed}, which has no run record"
              for aggregate,commit in (("results/run.json",old_aggregate),("new-results/run.json",new_aggregate))
              for seed in TABLE["seeds"]),
        )
        # And a committed aggregate stays.
        (root/self.AGGREGATE).unlink()
        commit_all(root,"deleted")
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn("X900: results/run.json was committed and has since been deleted or renamed; a run record stays as it was recorded",lines)
        # Aggregated again after the run, under the same preregistration,
        # the aggregate passes in every version.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate")
        self.publish(root,self.aggregate(root,ran,note="again"))
        commit_all(root,"aggregate again")
        self.assertEqual(gate(root),(0,[]))
        # Deleted and put back as it was, it is the aggregate committed.
        (root/self.AGGREGATE).unlink()
        commit_all(root,"aggregate deleted")
        self.publish(root,self.aggregate(root,ran,note="again"))
        commit_all(root,"aggregate restored")
        self.assertEqual(gate(root),(0,[]))
        # A version committed as a link was another file's content, read
        # through it; git holds only the link's target path.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        write(root,"experiments/semdb/X900-fixture/elsewhere.json",json.dumps(self.aggregate(root,ran)))
        self.link(Path("../elsewhere.json"),root/self.AGGREGATE)
        linked=commit_all(root,"aggregate linked")
        (root/self.AGGREGATE).unlink()
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate")
        self.assert_blocked(root,f"X900: results/run.json as committed at {linked[:12]} is not a regular file")

    def test_an_aggregate_names_the_run_record_of_each_preregistered_seed(self):
        # Fabricated outcomes, committed beside no run or beside records
        # rewritten since, would pass on the digests an aggregate carries
        # alone: it binds each preregistered seed's record by its SHA-256.
        def missing(seed):
            return f"X900: results/run.json reports on seed {seed}, which has no run record"

        def refused(seed,named,held):
            return (f"X900: results/run.json names seed_records[{seed}] {named!r}, not {held}, the SHA-256 of the run "
                    f"record of seed {seed}")

        # A completed experiment whose aggregate is all there is: no run
        # record holds any seed the aggregate reports on.
        root=self.tree(status="completed")
        ran=commit_all(root)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        write(root,"experiments/semdb/X900-fixture/results/metrics.json","{}")
        self.assert_blocked(root,missing(17),missing(29))
        # Named by the runs that happened, it passes, whatever the status.
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                root=self.tree(status=status)
                ran=commit_all(root)
                runs=self.seeded(root,ran)
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
                write(root,"experiments/semdb/X900-fixture/results/metrics.json","{}")
                self.assertEqual(gate(root),(0,[]))
        digest_17=hashlib.sha256((root/f"experiments/semdb/X900-fixture/{runs[0]}").read_bytes()).hexdigest()
        digest_29=hashlib.sha256((root/f"experiments/semdb/X900-fixture/{runs[1]}").read_bytes()).hexdigest()
        held={"17":digest_17,"29":digest_29}
        # The aggregate may not leave the binding out, or write it as
        # anything but an object.
        names_none=("X900: results/run.json names no seed_records, the SHA-256 of the run record of each "
                    "preregistered seed, so nothing binds the outcome it reports to the runs")
        for report in ({"git_sha":ran,**dict(zip(("preregistration_sha256","preregistration_rules_sha256"),self.frozen(root)))},
                       {**self.aggregate(root,ran),"seed_records":[digest_17,digest_29]},
                       {**self.aggregate(root,ran),"seed_records":"none"},
                       {**self.aggregate(root,ran),"seed_records":None}):
            with self.subTest(report=report):
                write(root,self.AGGREGATE,json.dumps(report))
                self.assert_blocked(root,names_none)
        # Each seed's digest is that of its record; a wrong one, one left out
        # and one for a seed that was not preregistered are each named.
        wrong="0"*64
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records={**held,"17":wrong})))
        self.assert_blocked(root,refused(17,wrong,digest_17))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records={"29":digest_29})))
        self.assert_blocked(root,refused(17,None,digest_17))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records={**held,"41":digest_17})))
        self.assert_blocked(root,"X900: results/run.json names a record for seed 41, which is not preregistered")
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records={**held,"17":digest_29,"29":digest_17})))
        self.assert_blocked(root,refused(17,digest_29,digest_17),refused(29,digest_17,digest_29))
        # A record rewritten after the aggregate named it is not the one it
        # named, and the record itself is one committed as it was.
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        commit_all(root,"aggregate")
        self.assertEqual(gate(root),(0,[]))
        rewritten=self.record(root,ran,seed=17,status="completed")
        rewritten["stdout"]="another outcome"
        write(root,f"experiments/semdb/X900-fixture/{runs[0]}",json.dumps(rewritten))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(f"X900: {runs[0]} differs from the record committed as it",lines)
        self.assertTrue(any(line.startswith("X900: results/run.json names seed_records[17] ") for line in lines),lines)
        # A run whose command failed to launch saw no outcome, and so is no
        # run record to name; two records of one seed are named as that, not
        # as a record no aggregate could name.
        root=self.tree(status="running")
        ran=commit_all(root)
        runs=self.seeded(root,ran)
        results="experiments/semdb/X900-fixture/results"
        write(root,f"{results}/{runs[1].removeprefix('results/')}",json.dumps(self.record(root,ran,seed=29,status="failed-to-launch")))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assert_blocked(root,missing(29))
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        write(root,f"{results}/run-20260201T000000.000000Z-seed-17.json",json.dumps(self.record(root,ran,seed=17,status="completed")))
        digest,rules=self.frozen(root)
        write(root,self.AGGREGATE,json.dumps({"git_sha":ran,"preregistration_sha256":digest,"preregistration_rules_sha256":rules,
                                              "seed_records":{"17":"0"*64,"29":"0"*64}}))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertTrue(any("seed 17 ran more than once" in line for line in lines),lines)
        self.assertFalse(any("names seed_records[17]" in line for line in lines),lines)
        self.assertTrue(any(line.startswith("X900: results/run.json names seed_records[29] ") for line in lines),lines)

    def test_a_reservation_nothing_finished_is_no_run_an_aggregate_reports_on(self):
        # A runner that died or was refused after it reserved a seed leaves
        # its record at `started`, with no exit code and no output: it is a
        # run of its seed, which no other may follow, but holds no outcome
        # for an aggregate to bind, however its digest is named.
        record="results/run-20260101T000000.000000Z-seed-17.json"
        other="results/run-20260108T000000.000000Z-seed-29.json"
        directory="experiments/semdb/X900-fixture"

        def unfinished(seed,held):
            return (f"X900: results/run.json reports on seed {seed}, whose run record ({held}) holds no outcome; a run that "
                    "finished has status completed or failed with its exit_code, finished_at, stdout and stderr, and a "
                    "reservation nothing finished saw none")

        def named(root):
            """The aggregate a hand would write: the SHA-256 of each file."""
            return {str(seed):hashlib.sha256((root/directory/name).read_bytes()).hexdigest()
                    for seed,name in zip(TABLE["seeds"],(record,other))}

        root=self.tree(status="running")
        ran=commit_all(root)
        for seed,name in zip(TABLE["seeds"],(record,other)):
            write(root,f"{directory}/{name}",json.dumps(self.record(root,ran,seed=seed,status="started")))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records=named(root))))
        self.assert_blocked(root,unfinished(17,record),unfinished(29,other))
        # The aggregate the aggregator would write names none of them.
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assertEqual(json.loads((root/self.AGGREGATE).read_text(encoding="utf-8"))["seed_records"],{})
        self.assert_blocked(root,unfinished(17,record),unfinished(29,other))
        # One seed finished and the other reserved: only the reservation is named.
        finished=self.record(root,ran,seed=17,status="completed")
        write(root,f"{directory}/{record}",json.dumps(finished))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records=named(root))))
        self.assert_blocked(root,unfinished(29,other))
        # Both finished, the aggregate passes.
        write(root,f"{directory}/{other}",json.dumps(self.record(root,ran,seed=29,status="failed")))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assertEqual(gate(root),(0,[]))

        # A status is not an outcome: the record must hold the exit code, the
        # time it finished and the output, and the status must agree with the
        # code.
        def without(key):
            return lambda held:held.pop(key)

        def setting(**changes):
            return lambda held:held.update(changes)

        cases=[
            ("no exit code",without("exit_code")),
            ("no finish time",without("finished_at")),
            ("no stdout",without("stdout")),
            ("no stderr",without("stderr")),
            ("a null exit code",setting(exit_code=None)),
            ("a text exit code",setting(exit_code="0")),
            ("a boolean exit code",setting(exit_code=False)),
            ("a float exit code",setting(exit_code=0.0)),
            ("a null finish time",setting(finished_at=None)),
            ("a number for stdout",setting(stdout=0)),
            ("a list for stderr",setting(stderr=[])),
            ("completed with a failing code",setting(exit_code=1)),
            ("failed with a passing code",setting(status="failed",exit_code=0)),
            ("running",setting(status="running")),
            ("an unknown status",setting(status="unknown")),
            ("another case",setting(status="Completed")),
        ]
        for label,change in cases:
            with self.subTest(label=label):
                held=self.record(root,ran,seed=17,status="completed")
                change(held)
                write(root,f"{directory}/{record}",json.dumps(held))
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records=named(root))))
                self.assert_blocked(root,unfinished(17,record))
        # A reservation beside a finished record of its seed is two runs of
        # one seed, named as that; the finished record is the outcome, so the
        # seed is not also named as one with none.
        write(root,f"{directory}/{record}",json.dumps(finished))
        again="results/run-20260201T000000.000000Z-seed-17.json"
        write(root,f"{directory}/{again}",json.dumps(self.record(root,ran,seed=17,status="started")))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records=named(root))))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertTrue(any(f"seed 17 ran more than once ({record}, {again})" in line for line in lines),lines)
        self.assertFalse(any("holds no outcome" in line for line in lines),lines)
        # Two reservations of one seed hold no outcome either.
        write(root,f"{directory}/{record}",json.dumps(self.record(root,ran,seed=17,status="started")))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertTrue(any("seed 17 ran more than once" in line for line in lines),lines)
        self.assertIn(unfinished(17,f"{record}, {again}"),lines)

    def test_an_aggregate_replaced_by_a_bound_one_is_bound_in_the_version_it_was_first_committed_in(self):
        # An aggregate written again keeps every version it was committed in:
        # a fabricated one replaced by an aggregate that names the records is
        # still the outcome its commit showed.
        def refused(at,seed,named,held):
            return (f"X900: results/run.json as committed at {at[:12]} names seed_records[{seed}] {named!r}, not {held}, "
                    f"the SHA-256 of the run record of seed {seed}")

        def names_none(at):
            return (f"X900: results/run.json as committed at {at[:12]} names no seed_records, the SHA-256 of the run record "
                    "of each preregistered seed, so nothing binds the outcome it reports to the runs")

        wrong="0"*64
        # A version that names the records, and one written over it, pass.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        self.publish(root,bound)
        commit_all(root,"aggregate")
        self.publish(root,{**bound,"note":"again"})
        commit_all(root,"aggregate again")
        self.assertEqual(gate(root),(0,[]))
        # Each of these, committed first and replaced by a bound aggregate,
        # is found where it was committed.
        cases=[
            ("no binding",
             lambda bound,held:{key:value for key,value in bound.items() if key!="seed_records"},
             lambda first,held:[names_none(first)]),
            ("a list",
             lambda bound,held:{**bound,"seed_records":list(held.values())},
             lambda first,held:[names_none(first)]),
            ("wrong digests",
             lambda bound,held:{**bound,"seed_records":{"17":wrong,"29":wrong}},
             lambda first,held:[refused(first,17,wrong,held["17"]),refused(first,29,wrong,held["29"])]),
            # Committed before the last seed ran, the aggregate showed the
            # outcome of the others, by which what ran next could be chosen.
            ("one seed only",
             lambda bound,held:{**bound,"seed_records":{"17":held["17"]}},
             lambda first,held:[refused(first,29,None,held["29"])]),
            ("a record for a seed not preregistered",
             lambda bound,held:{**bound,"seed_records":{**held,"41":wrong}},
             lambda first,held:[f"X900: results/run.json as committed at {first[:12]} names a record for seed 41, which is "
                                "not preregistered"]),
        ]
        for label,fabricate,expected in cases:
            with self.subTest(label=label):
                root=self.tree(status="running")
                ran=commit_all(root)
                self.seeded(root,ran)
                bound=self.aggregate(root,ran)
                held=bound["seed_records"]
                self.assertEqual(sorted(held),["17","29"])
                self.publish(root,fabricate(bound,held))
                first=commit_all(root,"fabricated aggregate")
                self.publish(root,bound)
                commit_all(root,"aggregate")
                self.assert_blocked(root,*expected(first,held))
        # A version between the first and the last is bound the same.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        self.publish(root,bound)
        commit_all(root,"aggregate")
        self.publish(root,{**bound,"seed_records":{}})
        middle=commit_all(root,"emptied")
        self.publish(root,{**bound,"note":"again"})
        commit_all(root,"aggregate again")
        self.assert_blocked(root,*(refused(middle,int(seed),None,held) for seed,held in bound["seed_records"].items()))

    def test_a_completed_listed_experiment_holds_its_aggregate_whatever_its_manifest_lists(self):
        # `required_artifacts` is written by the experiment's author: a
        # completed experiment that lists none, with no run record and no
        # aggregate, would otherwise pass though none of its seeds ran.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="completed")
        commit_all(root)
        self.assert_blocked(root,*unarchived("completed"))
        # Naming them in the manifest names them once.
        root=self.tree(status="completed")
        self.edit(root,self.MANIFEST,"required_artifacts = []",'required_artifacts = ["run.json", "metrics.json"]')
        commit_all(root)
        self.assert_blocked(root,*unarchived("completed"))
        # What the manifest lists beyond them is needed as well.
        root=self.tree(status="completed")
        self.edit(root,self.MANIFEST,"required_artifacts = []",'required_artifacts = ["plot.png", "run.json"]')
        commit_all(root)
        self.assert_blocked(root,"X900: completed experiment missing plot.png",*unarchived("completed"))
        # The aggregate, with a run of every seed and its metrics, is enough.
        root=self.tree(status="completed")
        ran=commit_all(root)
        self.seeded(root,ran)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assert_blocked(root,"X900: completed experiment missing metrics.json")
        write(root,f"{directory}/results/metrics.json","{}")
        self.assertEqual(gate(root),(0,[]))
        # Metrics without the aggregate leave the runs unbound.
        (root/self.AGGREGATE).unlink()
        self.assert_blocked(root,"X900: completed experiment missing run.json")
        # An experiment that has not completed needs neither.
        for status in ("prepared","running","failed"):
            with self.subTest(status=status):
                self.assertEqual(gate(self.tree(status=status)),(0,[]))

    def test_an_experiment_that_was_completed_keeps_the_aggregate_it_was_completed_with(self):
        # A commit that held it completed declared it complete: superseding
        # it afterwards does not excuse the aggregate, whose absence the gate
        # asks of a completed one, or the run of every preregistered seed.
        directory="experiments/semdb/X900-fixture"

        def supersede(root):
            for relative in (self.MANIFEST,"experiments/registry.toml"):
                self.edit(root,relative,'status = "completed"','status = "superseded"')
            return commit_all(root,"superseded")

        root=self.tree(status="completed")
        completed=commit_all(root)
        supersede(root)
        self.assert_blocked(root,*superseded_unarchived(completed))
        # Kept, with a run of every seed, it passes.
        root=self.tree(status="completed")
        ran=commit_all(root)
        self.seeded(root,ran)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        write(root,f"{directory}/results/metrics.json","{}")
        commit_all(root,"archive")
        self.assertEqual(gate(root),(0,[]))
        supersede(root)
        self.assertEqual(gate(root),(0,[]))
        # Deleted afterwards, the archive is missing again.
        (root/self.AGGREGATE).unlink()
        (root/directory/"results/metrics.json").unlink()
        code,lines=gate(root)
        self.assertEqual(code,1)
        for name in ("run.json","metrics.json"):
            self.assertIn(f"X900: experiment, completed at {ran[:12]} and now 'superseded', missing {name}",lines)
        # An experiment that was never completed needs none, whatever it was
        # before it was superseded.
        for then in ("prepared","running","failed"):
            with self.subTest(then=then):
                root=self.tree(status=then)
                commit_all(root)
                for relative in (self.MANIFEST,"experiments/registry.toml"):
                    self.edit(root,relative,f'status = "{then}"','status = "superseded"')
                commit_all(root,"superseded")
                self.assertEqual(gate(root),(0,[]))
        # A results directory that is no repository path is named as that by
        # its own check, and no artifact is looked for beyond the repository
        # (a status that went back from completed is named too).
        root=self.tree(status="completed")
        self.edit(root,self.MANIFEST,"required_artifacts = []",'required_artifacts = []\nresults_dir = "../elsewhere"')
        commit_all(root)
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "completed"','status = "running"')
        commit_all(root,"back to running")
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertTrue(any("results_dir '../elsewhere' is not a directory below the experiment's directory" in line for line in lines),lines)
        self.assertFalse(any(" missing " in line for line in lines),lines)
        # The commit that completed it is the one the error names.
        root=self.tree(status="running")
        commit_all(root)
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "running"','status = "completed"')
        first=commit_all(root,"completed")
        supersede(root)
        self.assert_blocked(root,*superseded_unarchived(first))

    def test_an_artifact_of_a_completed_experiment_is_a_file_no_symlink_stands_for(self):
        # A link reads the content of another file, which no commit holds at
        # the artifact's own path.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="completed")
        ran=commit_all(root)
        self.seeded(root,ran)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        write(root,f"{directory}/results/metrics.json","{}")
        self.assertEqual(gate(root),(0,[]))
        message=lambda name:(f"X900: completed experiment {name} is a symlink or lies below one; an artifact is the file a "
                             "commit holds at its own path")
        for name in ("run.json","metrics.json"):
            with self.subTest(name=name):
                artifact=root/directory/"results"/name
                kept=artifact.read_bytes()
                artifact.unlink()
                write(root,f"{directory}/elsewhere.json",kept.decode("utf-8"))
                self.link(Path("../elsewhere.json"),artifact)
                code,lines=gate(root)
                self.assertEqual(code,1)
                self.assertIn(message(name),lines)
                artifact.unlink()
                artifact.write_bytes(kept)
                (root/directory/"elsewhere.json").unlink()
                self.assertEqual(gate(root),(0,[]))
        # So is an artifact below a linked directory.
        results=root/directory/"results"
        moved=root/directory/"moved"
        results.rename(moved)
        self.link(Path("moved"),results)
        code,lines=gate(root)
        self.assertEqual(code,1)
        for name in ("run.json","metrics.json"):
            self.assertIn(message(name),lines)

    def test_a_path_is_reached_through_a_symlink_at_any_step_from_the_root_down(self):
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))/"root"
        (root/"real/deep").mkdir(parents=True)
        (root/"real/deep/file").write_text("x",encoding="utf-8")
        self.assertFalse(mod.reached_through_symlink(root,root/"real/deep/file"))
        # The first step, one between, and the file itself.
        self.link(Path("real"),root/"top")
        self.assertTrue(mod.reached_through_symlink(root,root/"top/deep/file"))
        self.link(Path("deep"),root/"real/mid")
        self.assertTrue(mod.reached_through_symlink(root,root/"real/mid/file"))
        self.link(Path("file"),root/"real/deep/alias")
        self.assertTrue(mod.reached_through_symlink(root,root/"real/deep/alias"))
        # The first step of a path is checked as the others are.
        self.assertFalse(mod.reached_through_symlink(root,root/"real/deep"))
        self.assertTrue(mod.reached_through_symlink(root,root/"top"))
        # A path outside the root is not one the commit holds.
        self.assertTrue(mod.reached_through_symlink(root,root.parent/"elsewhere"))

    def test_a_run_record_names_the_command_its_commits_frozen_values_give_its_seed(self):
        # The runner builds the command from the manifest and the table as
        # the commit holds them: a record of another program, or of other
        # values, is no run of what was frozen, whatever the launch read.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="running")
        ran=commit_all(root)
        names=self.seeded(root,ran)
        self.assertEqual(gate(root),(0,[]))

        def wrong(name,seed,command,parameters):
            return [f"X900: {name} names command {command!r}, not {['bench',str(seed)]!r}, the command the manifest and "
                    f"[preregistration] table at {ran[:12]} give seed {seed}",
                    f"X900: {name} names parameters {parameters!r}, not {{}}, the values the [preregistration] table at "
                    f"{ran[:12]} freezes for the command's placeholders"]

        first=names[0]
        held=self.record(root,ran,seed=17,status="completed")
        # A program the table did not freeze, another seed's value, no command.
        for changes,expected in (
            ({"command":["/tmp/attacker","17"]},wrong(first,17,["/tmp/attacker","17"],{})[:1]),
            ({"command":["bench","29"]},wrong(first,17,["bench","29"],{})[:1]),
            ({"command":["bench","17","--x"]},wrong(first,17,["bench","17","--x"],{})[:1]),
            ({"command":"bench 17"},wrong(first,17,"bench 17",{})[:1]),
            ({"parameters":{"iterations":"30"}},wrong(first,17,["bench","17"],{"iterations":"30"})[1:]),
            ({"parameters":None},wrong(first,17,["bench","17"],None)[1:]),
        ):
            with self.subTest(changes=changes):
                write(root,f"{directory}/{first}",json.dumps({**held,**changes}))
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
                self.assert_blocked(root,*expected)
        # Nor may either be left out.
        without={key:value for key,value in held.items() if key not in ("command","parameters")}
        write(root,f"{directory}/{first}",json.dumps(without))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assert_blocked(root,*wrong(first,17,None,None))
        # A command that failed to launch was built the same way.
        write(root,f"{directory}/{first}",json.dumps({**held,"status":"failed-to-launch","exit_code":None,
                                                     "command":["/tmp/attacker","17"]}))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,seed_records={"29":"0"*64})))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(wrong(first,17,["/tmp/attacker","17"],{})[0],lines)
        # A record naming no seed is a prepared one: no command was built.
        write(root,f"{directory}/{first}",json.dumps({key:value for key,value in without.items() if key!="seed"}))
        code,lines=gate(root)
        self.assertFalse(any("names command" in line for line in lines),lines)
        # A seed that is no integer is refused as a seed the table did not
        # freeze, and no command is derived from it.
        for seed in (True,17.0,"17",None,[17]):
            with self.subTest(seed=seed):
                write(root,f"{directory}/{first}",json.dumps({**held,"seed":seed,"command":["bench","17"]}))
                code,lines=gate(root)
                self.assertFalse(any("names command" in line for line in lines),lines)

    def test_the_values_a_records_command_takes_are_those_its_commits_table_freezes(self):
        directory="experiments/semdb/X900-fixture"
        table={**TABLE,"iterations":30,"verbose":True,"name":"alpha beta"}
        root=self.tree(status="running",table=table)
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n',
                  'entrypoint = "bench <seed> <iterations> <verbose> \'<name>\'"\n')
        ran=commit_all(root)
        command=lambda seed:["bench",str(seed),"30","true","alpha beta"]
        values={"iterations":"30","verbose":"true","name":"alpha beta"}
        for seed in TABLE["seeds"]:
            name=f"results/run-2026010{seed % 9}T000000.000000Z-seed-{seed}.json"
            write(root,f"{directory}/{name}",json.dumps(
                self.record(root,ran,seed=seed,status="completed",command=command(seed),parameters=values)))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        self.assertEqual(gate(root),(0,[]))
        # Another value of any placeholder is another command.
        seventeen=f"{directory}/results/run-20260108T000000.000000Z-seed-17.json"
        for label,changes in (
            ("integer",{"command":["bench","17","31","true","alpha beta"],"parameters":{**values,"iterations":"31"}}),
            ("boolean",{"command":["bench","17","30","false","alpha beta"],"parameters":{**values,"verbose":"false"}}),
            ("string",{"command":["bench","17","30","true","other"],"parameters":{**values,"name":"other"}}),
            ("parameters alone",{"parameters":{**values,"iterations":"31"}}),
        ):
            with self.subTest(label=label):
                write(root,seventeen,json.dumps(self.record(root,ran,seed=17,status="completed",
                                                           **{"command":command(17),"parameters":values,**changes})))
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
                code,lines=gate(root)
                self.assertEqual(code,1)
                self.assertTrue(any(" names command " in line or " names parameters " in line for line in lines),lines)

    def test_an_aggregate_version_is_bound_to_the_metrics_and_evidence_committed_beside_it(self):
        # An aggregate written again keeps every version it was committed
        # in: one that named other metrics, or none, replaced by a valid pair
        # later, still saw what it was committed with.
        directory="experiments/semdb/X900-fixture"
        right=hashlib.sha256(b"{}").hexdigest()
        other=hashlib.sha256(b"[]").hexdigest()
        evidence=b'{"killed": 2, "total": 2}\n'
        held=hashlib.sha256(evidence).hexdigest()

        def problem(first,field,detail):
            return f"X900: results/run.json as committed at {first[:12]} names {detail}"

        wrong_metrics=lambda first:problem(first,"metrics_sha256",f"metrics_sha256 {other}, not {right}, the SHA-256 of the metrics.json committed beside it")
        cases=[
            ("other metrics",{"metrics_sha256":other},None,lambda first:[wrong_metrics(first)]),
            ("no metrics digest",{"metrics_sha256":None},None,lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names no metrics_sha256, so nothing binds it to the "
                "metrics.json committed beside it"]),
            ("a digest that is no text",{"metrics_sha256":5},None,lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names no metrics_sha256, so nothing binds it to the "
                "metrics.json committed beside it"]),
            ("no metrics file",{},"no-metrics",lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names metrics_sha256 {right}, but that commit holds "
                "no metrics.json beside it"]),
            ("other evidence",{"mutation_checks":{"sha256":other,"killed":2,"total":2}},"evidence",lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names mutation_checks.sha256 {other}, not {held}, "
                "the SHA-256 of the mutations.json committed beside it"]),
            ("evidence with no digest",{"mutation_checks":{"killed":2,"total":2}},"evidence",lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names no mutation_checks.sha256, so nothing binds it "
                "to the mutations.json committed beside it"]),
            ("evidence that is not there",{"mutation_checks":{"sha256":held,"killed":2,"total":2}},None,lambda first:[
                f"X900: results/run.json as committed at {first[:12]} names mutation_checks.sha256 {held}, but that commit "
                "holds no mutations.json beside it"]),
        ]
        for label,changes,files,expected in cases:
            with self.subTest(label=label):
                root=self.tree(status="running")
                ran=commit_all(root)
                self.seeded(root,ran)
                bound=self.aggregate(root,ran)
                write(root,self.METRICS,"{}")
                if files=="no-metrics":
                    (root/self.METRICS).unlink()
                if files=="evidence":
                    (root/f"{directory}/results/mutations.json").write_bytes(evidence)
                report={**bound,"metrics_sha256":right,**changes}
                if report["metrics_sha256"] is None:
                    del report["metrics_sha256"]
                write(root,self.AGGREGATE,json.dumps(report))
                first=commit_all(root,"aggregate that names what it should not")
                # Replaced by a valid pair, it is still named where it was committed.
                (root/f"{directory}/results/mutations.json").unlink(missing_ok=True)
                self.publish(root,{**bound,"note":"valid pair"})
                commit_all(root,"valid pair")
                self.assert_blocked(root,*expected(first))
        # A version whose artifacts are the ones it names passes, mutation
        # evidence included.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        (root/f"{directory}/results/mutations.json").parent.mkdir(parents=True,exist_ok=True)
        (root/f"{directory}/results/mutations.json").write_bytes(evidence)
        checked=self.aggregate(root,ran,mutation_checks={"sha256":held,"killed":2,"total":2,"git_sha":None})
        self.publish(root,checked)
        commit_all(root,"aggregate with evidence")
        self.publish(root,{**checked,"note":"again"})
        commit_all(root,"aggregate again")
        self.assertEqual(gate(root),(0,[]))

    def test_the_aggregate_of_an_experiment_once_completed_stays_the_one_it_was(self):
        # The metrics and the mutation evidence an archived aggregate names
        # are checked while the experiment is completed; superseding it does
        # not free them to be replaced.
        directory="experiments/semdb/X900-fixture"

        def gate_with_aggregates(root):
            def archived(exp_id,experiment,results,given_root,listed_experiment=False):
                return []

            output=io.StringIO()
            with mock.patch.object(mod.experiment_records,"staleness_errors",side_effect=archived),contextlib.redirect_stdout(output):
                code=mod.main(root)
            return code,[line.removeprefix("ERROR: ") for line in output.getvalue().splitlines() if line.startswith("ERROR: ")]

        root=self.tree(status="completed")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"archive")
        self.assertEqual(gate_with_aggregates(root),(0,[]))
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "completed"','status = "superseded"')
        commit_all(root,"superseded")
        self.assertEqual(gate_with_aggregates(root),(0,[]))
        # Metrics replaced after the supersession are not the ones the
        # aggregate names.
        write(root,self.METRICS,"[]")
        replaced=commit_all(root,"metrics replaced")
        code,lines=gate_with_aggregates(root)
        self.assertEqual(code,1)
        # The metrics on disk, and the state the aggregate was in at the
        # commit that replaced them (a commit that changes one artifact of
        # the pair is a version of it).
        self.assertEqual(len(lines),2,lines)
        self.assertEqual(len([line for line in lines if line.startswith(
            f"X900: {directory}/results/metrics.json is not the metrics run.json was aggregated with")]),1,lines)
        self.assertEqual(len([line for line in lines if line.startswith(
            f"X900: results/run.json as committed at {replaced[:12]} names metrics_sha256 ")]),1,lines)
        # An experiment that was never completed is not asked for the
        # integrity of the files on disk, but every state its aggregate was
        # in stays a version of it: the commit that replaced the metrics is
        # named, and nothing else.
        for then in ("prepared","running","failed"):
            with self.subTest(then=then):
                root=self.tree(status=then)
                ran=commit_all(root)
                self.seeded(root,ran)
                self.publish(root,self.aggregate(root,ran))
                commit_all(root,"aggregate")
                for relative in (self.MANIFEST,"experiments/registry.toml"):
                    self.edit(root,relative,f'status = "{then}"','status = "superseded"')
                commit_all(root,"superseded")
                self.assertEqual(gate_with_aggregates(root),(0,[]))
                write(root,self.METRICS,"[]")
                replaced=commit_all(root,"metrics replaced")
                code,lines=gate_with_aggregates(root)
                self.assertEqual(code,1)
                self.assertEqual(len(lines),1,lines)
                self.assertTrue(lines[0].startswith(
                    f"X900: results/run.json as committed at {replaced[:12]} names metrics_sha256 "),lines)

    def test_a_commit_that_holds_a_symlink_or_a_gitlink_anywhere_freezes_nothing(self):
        # The runner watches the whole repository of a listed experiment and
        # refuses one that holds either, which git holds as a path and not
        # as content: a commit that holds one could not launch, and is no
        # freeze whatever its preregistration holds, so a later repair is no
        # rewrite of one.
        directory="experiments/semdb/X900-fixture"
        for kind in ("symlink","gitlink"):
            with self.subTest(kind=kind):
                root=self.tree(status="running")
                if kind=="symlink":
                    self.link(Path("README.md"),root/"unrelated-link")
                    linked=commit_all(root)
                else:
                    git(root,"init","-q")
                    git(root,"add","-A")
                    git(root,"update-index","--add","--cacheinfo",f"160000,{'a'*40},vendored")
                    git(root,"commit","-q","--no-verify","-m","tree with a gitlink")
                    linked=git(root,"rev-parse","HEAD")
                self.assertEqual(mod.commit_links(root,linked),["unrelated-link" if kind=="symlink" else "vendored"])
                self.assertIsNone(mod.launchable_at(root,linked,"X900",directory))
                self.assertEqual([commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[])
                # Repaired: the link goes and the preregistration is
                # completed. That commit is the first that could launch it.
                if kind=="symlink":
                    (root/"unrelated-link").unlink()
                else:
                    git(root,"update-index","--force-remove","vendored")
                rewritten=self.tree(status="running",table={**TABLE,"schema":2})
                for relative in (f"{directory}/config.toml",self.MANIFEST):
                    shutil.copyfile(rewritten/relative,root/relative)
                repair=commit_all(root,"repaired")
                self.assertEqual(mod.commit_links(root,repair),[])
                self.assertIsNotNone(mod.launchable_at(root,repair,"X900",directory))
                self.assertEqual(
                    [commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[repair])
                self.assertEqual(gate(root),(0,[]))
        # A link inside the tree that is no link (a file named like one) is no link.
        root=self.tree(status="running")
        write(root,"notes/unrelated-link","not a link\n")
        held=commit_all(root)
        self.assertEqual(mod.commit_links(root,held),[])
        self.assertIsNotNone(mod.launchable_at(root,held,"X900",directory))

    def test_an_aggregate_version_equal_to_the_one_now_is_bound_to_what_its_commit_held_beside_it(self):
        # The bytes of run.json may never change while the metrics beside it
        # do: the version committed with metrics it did not name is that
        # version, whatever the aggregate on disk is.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        write(root,self.METRICS,"[]")
        write(root,self.AGGREGATE,json.dumps(bound))
        first=commit_all(root,"aggregate beside other metrics")
        write(root,self.METRICS,"{}")
        commit_all(root,"metrics repaired")
        right=hashlib.sha256(b"{}").hexdigest()
        other=hashlib.sha256(b"[]").hexdigest()
        self.assertEqual(json.loads((root/self.AGGREGATE).read_text(encoding="utf-8")),bound)
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {first[:12]} names metrics_sha256 {right}, not {other}, the SHA-256 of "
            "the metrics.json committed beside it",
        )
        # Beside the metrics it names, the same bytes pass.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate with its metrics")
        self.assertEqual(gate(root),(0,[]))

    def test_an_aggregate_version_reports_what_the_evidence_beside_it_says(self):
        # The digest binds the evidence a version names, but what run.json
        # reports of it (`killed`, `total`, `git_sha`) is compared with what
        # the evidence says, as the aggregate on disk is: a version that
        # reported other counts, replaced later by the right summary, still
        # reported them where it was committed.
        directory="experiments/semdb/X900-fixture"
        evidence=b'{"killed": 2, "total": 2, "git_sha": "' + b"a"*40 + b'"}\n'
        held=hashlib.sha256(evidence).hexdigest()
        summary={"killed":2,"total":2,"git_sha":"a"*40,"sha256":held}

        def sums_up(first,carried):
            return (f"X900: results/run.json as committed at {first[:12]} carries mutation_checks "
                    f"{json.dumps(carried,sort_keys=True)}, but the mutations.json committed beside it sums up to "
                    f"{json.dumps(summary,sort_keys=True)}")

        cases=[
            ("killed",{**summary,"killed":999}),
            ("total",{**summary,"total":999}),
            ("commit",{**summary,"git_sha":"b"*40}),
            ("no commit",{key:value for key,value in summary.items() if key!="git_sha"}),
            ("another key",{**summary,"extra":1}),
        ]
        for label,carried in cases:
            with self.subTest(label=label):
                root=self.tree(status="running")
                ran=commit_all(root)
                self.seeded(root,ran)
                (root/f"{directory}/results/mutations.json").write_bytes(evidence)
                bound=self.aggregate(root,ran)
                self.publish(root,{**bound,"mutation_checks":carried})
                first=commit_all(root,"aggregate that reports other counts")
                self.publish(root,{**bound,"mutation_checks":summary,"note":"the right summary"})
                commit_all(root,"the right summary")
                self.assert_blocked(root,sums_up(first,carried))
        # A version that carries no object, or none beside evidence its own
        # commit holds (which `aggregate_problems` refuses for the aggregate
        # on disk), or evidence that cannot be read.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        (root/f"{directory}/results/mutations.json").write_bytes(evidence)
        bound=self.aggregate(root,ran)
        self.publish(root,bound)
        first=commit_all(root,"aggregate that reports nothing of the evidence")
        self.publish(root,{**bound,"mutation_checks":summary,"note":"the right summary"})
        commit_all(root,"the right summary")
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {first[:12]} carries no mutation_checks, but that commit holds a "
            "mutations.json beside it")
        # Beside no evidence, none is reported: nothing to say.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate with no mutation evidence")
        self.assertEqual(gate(root),(0,[]))
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        (root/f"{directory}/results/mutations.json").write_bytes(evidence)
        bound=self.aggregate(root,ran)
        self.publish(root,{**bound,"mutation_checks":"passed"})
        first=commit_all(root,"aggregate with no usable checks")
        self.publish(root,{**bound,"mutation_checks":summary,"note":"the right summary"})
        commit_all(root,"the right summary")
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {first[:12]} carries mutation_checks that is no object, so nothing "
            "binds it to the mutations.json committed beside it")
        for label,unreadable,reason in (
            ("not JSON",b"not json",lambda: str(self.json_error("not json"))),
            ("not an object",b"[]",lambda: "'list' object has no attribute 'get'"),
        ):
            with self.subTest(label=label):
                root=self.tree(status="running")
                ran=commit_all(root)
                self.seeded(root,ran)
                (root/f"{directory}/results/mutations.json").write_bytes(unreadable)
                digest=hashlib.sha256(unreadable).hexdigest()
                bound=self.aggregate(root,ran)
                self.publish(root,{**bound,"mutation_checks":{"sha256":digest,"killed":2,"total":2}})
                first=commit_all(root,"aggregate over unreadable evidence")
                (root/f"{directory}/results/mutations.json").write_bytes(evidence)
                self.publish(root,{**bound,"mutation_checks":summary,"note":"the right summary"})
                commit_all(root,"evidence and summary that agree")
                self.assert_blocked(
                    root,
                    f"X900: results/run.json as committed at {first[:12]} names mutation checks over the mutations.json "
                    f"committed beside it, which cannot be read: {reason()}")
        # Reporting what the evidence says passes.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        (root/f"{directory}/results/mutations.json").write_bytes(evidence)
        self.publish(root,self.aggregate(root,ran,mutation_checks=summary))
        commit_all(root,"aggregate that reports what the evidence says")
        self.assertEqual(gate(root),(0,[]))

    @staticmethod
    def json_error(text: str) -> ValueError:
        try:
            json.loads(text)
        except ValueError as error:
            return error
        raise AssertionError("valid JSON")

    def test_a_commit_that_changes_only_an_artifact_of_an_aggregate_is_one_of_its_versions(self):
        # An aggregate is run.json with the metrics.json and mutations.json
        # beside it. A commit that corrupts one of the two and leaves run.json
        # alone, restored by the next, is a state the aggregate was in, and
        # is checked as the commits that change run.json are.
        directory="experiments/semdb/X900-fixture"
        right=hashlib.sha256(b"{}").hexdigest()
        other=hashlib.sha256(b"[]").hexdigest()
        evidence=b'{"killed": 2, "total": 2}\n'
        held=hashlib.sha256(evidence).hexdigest()
        corrupt=b'{"killed": 9, "total": 9}\n'
        broken=hashlib.sha256(corrupt).hexdigest()

        def valid_pair():
            root=self.tree(status="running")
            ran=commit_all(root)
            self.seeded(root,ran)
            (root/f"{directory}/results/mutations.json").write_bytes(evidence)
            self.publish(root,self.aggregate(root,ran,mutation_checks={"sha256":held,"killed":2,"total":2,"git_sha":None}))
            commit_all(root,"valid pair")
            self.assertEqual(gate(root),(0,[]))
            return root

        root=valid_pair()
        write(root,self.METRICS,"[]")
        corrupted=commit_all(root,"metrics only")
        write(root,self.METRICS,"{}")
        commit_all(root,"metrics restored")
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {corrupted[:12]} names metrics_sha256 {right}, not {other}, the "
            "SHA-256 of the metrics.json committed beside it")
        root=valid_pair()
        (root/f"{directory}/results/mutations.json").write_bytes(corrupt)
        corrupted=commit_all(root,"evidence only")
        (root/f"{directory}/results/mutations.json").write_bytes(evidence)
        commit_all(root,"evidence restored")
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {corrupted[:12]} names mutation_checks.sha256 {held}, not {broken}, "
            "the SHA-256 of the mutations.json committed beside it")
        # A restored pair, and an artifact removed and put back, leave no
        # other state than the ones checked.
        root=valid_pair()
        (root/self.METRICS).unlink()
        removed=commit_all(root,"metrics removed")
        write(root,self.METRICS,"{}")
        commit_all(root,"metrics restored")
        self.assert_blocked(
            root,
            f"X900: results/run.json as committed at {removed[:12]} names metrics_sha256 {right}, but that commit holds no "
            "metrics.json beside it")

    def test_a_manifest_no_record_can_hold_is_no_freeze(self):
        # The runner refuses a manifest holding a TOML date or time, or a NaN,
        # before it launches, so a commit that holds one could not launch and
        # freezes nothing: a repair after it is no rewrite of a freeze.
        directory="experiments/semdb/X900-fixture"
        for label,line in (("date","held_on = 2026-01-01\n"),("time","held_at = 07:32:00\n"),("nan","ratio = nan\n")):
            with self.subTest(label=label):
                root=self.tree(status="running")
                write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+line)
                unrecordable=commit_all(root,"a manifest no record can hold")
                self.assertIsNone(mod.launchable_at(root,unrecordable,"X900",directory))
                self.assertEqual([commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[])
                # Repaired, and the preregistration completed at the same
                # time: that commit is the first that could launch it.
                rewritten=self.tree(status="running",table={**TABLE,"schema":2})
                for relative in (f"{directory}/config.toml",self.MANIFEST):
                    shutil.copyfile(rewritten/relative,root/relative)
                repair=commit_all(root,"repaired")
                self.assertIsNotNone(mod.launchable_at(root,repair,"X900",directory))
                self.assertEqual(
                    [commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[repair])
                self.assertEqual(gate(root),(0,[]))

    def test_a_run_record_once_committed_as_a_symlink_is_a_version_of_it_whatever_its_bytes(self):
        # A symlink holds its target's text as its content, so a record
        # first committed as a link whose target is the record's own text has
        # the blob of the regular file that replaces it: the two are told
        # apart by mode, not by blob name alone.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="running")
        ran=commit_all(root)
        name=f"{directory}/results/run-20260101T000000.000000Z-seed-17.json"
        text=json.dumps(self.record(root,ran,seed=17,status="completed"))
        (root/name).parent.mkdir(parents=True,exist_ok=True)
        self.link(Path(text),root/name)
        linked=commit_all(root,"the record as a symlink")
        self.assertEqual(git(root,"ls-tree",linked,"--",name).split()[0],"120000")
        (root/name).unlink()
        (root/name).write_bytes(text.encode("utf-8"))
        regular=commit_all(root,"the record as a regular file with the same bytes")
        self.assertEqual(git(root,"ls-tree",regular,"--",name).split()[0],"100644")
        self.assertEqual(git(root,"rev-parse",f"{linked}:{name}"),git(root,"rev-parse",f"{regular}:{name}"))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(
            "X900: results/run-20260101T000000.000000Z-seed-17.json was changed after it was committed "
            "(2 versions of it were committed)",lines)
        # A record committed once, and left alone, is one version.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,name,json.dumps(self.record(root,ran,seed=17,status="completed")))
        commit_all(root,"the record")
        self.assertEqual(gate(root),(0,[]))

    def test_an_experiment_once_completed_is_stale_when_the_code_changes_after_it_is_superseded(self):
        # The archived results of a completed experiment describe the code at
        # their git_sha. Superseding it moves its status only, which stays
        # no change after the runs, but any other file that changes makes
        # them stale as it does while the experiment is completed: rerun, or
        # say since when in STALE.toml.
        directory="experiments/semdb/X900-fixture"

        def real_gate(root):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                code=mod.main(root)
            return code,[line.removeprefix("ERROR: ") for line in output.getvalue().splitlines() if line.startswith("ERROR: ")]

        root=self.tree(status="completed")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"archive")
        self.assertEqual(real_gate(root),(0,[]))
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "completed"','status = "superseded"')
        commit_all(root,"superseded")
        self.assertEqual(real_gate(root),(0,[]))
        write(root,"notes/changed-after.md","a file the command could have read\n")
        commit_all(root,"a repository file changes")
        code,lines=real_gate(root)
        self.assertEqual(code,1)
        self.assertEqual(len(lines),1,lines)
        self.assertTrue(lines[0].startswith(
            f"X900: {directory}/results/run.json ran at {ran}, and the code has changed since, in "),lines)
        # An experiment that was never completed is not asked.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"aggregate")
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "running"','status = "superseded"')
        write(root,"notes/changed-after.md","a file the command could have read\n")
        commit_all(root,"superseded, and a repository file changes")
        self.assertEqual(real_gate(root),(0,[]))

    def test_a_commit_that_only_removes_a_link_is_listed_as_the_first_that_could_launch(self):
        # Removing a symlink or gitlink changes nothing a launch path holds,
        # and is the commit from which a run could launch: a preregistration
        # rewritten after it is a rewrite of that freeze.
        directory="experiments/semdb/X900-fixture"
        for kind in ("symlink","gitlink"):
            with self.subTest(kind=kind):
                root=self.tree(status="running")
                if kind=="symlink":
                    self.link(Path("README.md"),root/"unrelated-link")
                    commit_all(root)
                else:
                    git(root,"init","-q")
                    git(root,"add","-A")
                    git(root,"update-index","--add","--cacheinfo",f"160000,{'a'*40},vendored")
                    git(root,"commit","-q","--no-verify","-m","tree with a gitlink")
                if kind=="symlink":
                    git(root,"rm","-q","--cached","unrelated-link")
                    (root/"unrelated-link").unlink()
                else:
                    git(root,"update-index","--force-remove","vendored")
                git(root,"commit","-q","--no-verify","-m","the link goes")
                removal=git(root,"rev-parse","HEAD")
                self.assertEqual(git(root,"show","--format=","--name-only",removal).split(),["unrelated-link" if kind=="symlink" else "vendored"])
                self.assertIsNotNone(mod.launchable_at(root,removal,"X900",directory))
                self.assertEqual(
                    [commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[removal])
                self.assertEqual(gate(root),(0,[]))
                # The preregistration completed and rewritten after the
                # removal is a rewrite of what could launch there.
                rewritten=self.tree(status="running",table={**TABLE,"schema":2})
                for relative in (f"{directory}/config.toml",self.MANIFEST):
                    shutil.copyfile(rewritten/relative,root/relative)
                commit_all(root,"rewrite")
                code,lines=gate(root)
                self.assertEqual(code,1)
                self.assertIn(
                    f"X900 was frozen at {removal[:12]}, whose config.toml holds another [preregistration] than the frozen one",
                    lines)
        # A link added after a freeze is a commit that could not launch, and
        # is listed too, where the state changes.
        root=self.tree(status="running")
        first=commit_all(root)
        self.link(Path("README.md"),root/"unrelated-link")
        linked=commit_all(root,"link added")
        listed={commit:held for commit,held in mod.launch_listing(root,"X900",[directory])}
        self.assertIsNotNone(listed[first])
        self.assertIn(linked,listed)
        self.assertIsNone(listed[linked])
        # A merge that takes the link out, against its first parent, is
        # listed where the state changes on that line of history.
        root=self.tree(status="running")
        commit_all(root)
        git(root,"checkout","-q","-b","side")
        write(root,"notes/side.md","side\n")
        commit_all(root,"side")
        git(root,"checkout","-q","main")
        self.link(Path("README.md"),root/"unrelated-link")
        with_link=commit_all(root,"main adds a link")
        git(root,"merge","-q","--no-commit","--no-ff","-s","ours","side")
        git(root,"rm","-q","-f","unrelated-link")
        git(root,"commit","-q","--no-verify","-m","merge that takes the link out")
        merge=git(root,"rev-parse","HEAD")
        self.assertEqual(len(git(root,"rev-list","--parents","-n1","HEAD").split()),3)
        self.assertEqual(mod.link_changes_at(str(root),merge),frozenset({with_link,merge}))
        listed={commit:held for commit,held in mod.launch_listing(root,"X900",[directory])}
        self.assertIsNone(listed[with_link])
        self.assertIsNotNone(listed[merge])

    def test_an_aggregate_version_is_bound_to_the_seed_records_its_commit_held(self):
        # An aggregate committed before the run records it names, which are
        # committed afterwards, reported outcomes nothing that commit held
        # bound: the records of the final checkout do not stand in for them.
        directory="experiments/semdb/X900-fixture"
        root=self.tree(status="running")
        ran=commit_all(root)
        names=self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        # The records stay out of the first commit: only the aggregate goes in.
        write(root,self.METRICS,"{}")
        write(root,self.AGGREGATE,json.dumps(bound))
        git(root,"add",self.METRICS,self.AGGREGATE)
        git(root,"commit","-q","--no-verify","-m","aggregate before its records")
        first=git(root,"rev-parse","HEAD")
        commit_all(root,"the records")
        self.assertEqual(sorted(bound["seed_records"]),["17","29"])
        code,lines=gate(root)
        self.assertEqual(code,1)
        for seed in (17,29):
            self.assertIn(f"X900: results/run.json as committed at {first[:12]} reports on seed {seed}, which has no run record",lines)
        self.assertEqual(len(lines),2,lines)
        # Committed with them, or after them, the same aggregate passes.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        commit_all(root,"the records")
        self.publish(root,self.aggregate(root,ran))
        commit_all(root,"the aggregate")
        self.assertEqual(gate(root),(0,[]))
        # A record committed with the aggregate that is not the one it names,
        # and replaced later by the named one, is named where it was committed.
        root=self.tree(status="running")
        ran=commit_all(root)
        names=self.seeded(root,ran)
        held=self.aggregate(root,ran)
        seventeen=root/directory/names[0]
        kept=seventeen.read_bytes()
        seventeen.write_text(json.dumps({**json.loads(kept),"stdout":"another outcome"}),encoding="utf-8")
        self.publish(root,held)
        first=commit_all(root,"aggregate beside another record")
        seventeen.write_bytes(kept)
        commit_all(root,"the record it names")
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertTrue(any(line.startswith(f"X900: results/run.json as committed at {first[:12]} names seed_records[17] ") for line in lines),lines)
        # An earlier version, not the one on disk, committed before its
        # records is named as well; the records of the final checkout do
        # not stand in for them.
        root=self.tree(status="running")
        ran=commit_all(root)
        self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        write(root,self.METRICS,"{}")
        write(root,self.AGGREGATE,json.dumps(bound))
        git(root,"add",self.METRICS,self.AGGREGATE)
        git(root,"commit","-q","--no-verify","-m","aggregate before its records")
        first=git(root,"rev-parse","HEAD")
        commit_all(root,"the records")
        self.publish(root,{**bound,"note":"written again"})
        commit_all(root,"the aggregate again")
        code,lines=gate(root)
        self.assertEqual(code,1)
        for seed in (17,29):
            self.assertIn(f"X900: results/run.json as committed at {first[:12]} reports on seed {seed}, which has no run record",lines)
        self.assertEqual(len(lines),2,lines)
        # Only the records that commit held in the aggregate's own directory
        # count, and only run records that ran: a record elsewhere, a
        # command that failed to launch and a file that is no run record
        # (an aggregate that names a seed) are no run of it, and a
        # reservation nothing finished holds no outcome.
        root=self.tree(status="running")
        ran=commit_all(root)
        names=self.seeded(root,ran)
        bound=self.aggregate(root,ran)
        results=root/directory/"results"
        seventeen=results/names[0].removeprefix("results/")
        twenty_nine=results/names[1].removeprefix("results/")
        elsewhere=root/directory/"older"/names[0].removeprefix("results/")
        elsewhere.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(seventeen,elsewhere)
        seventeen.write_text(json.dumps(self.record(root,ran,seed=17,status="failed-to-launch",exit_code=None)),encoding="utf-8")
        twenty_nine.write_text(json.dumps(self.record(root,ran,seed=29,status="started")),encoding="utf-8")
        write(root,self.METRICS,"{}")
        write(root,self.AGGREGATE,json.dumps({**bound,"seed":17,"status":"completed","exit_code":0,"finished_at":"t","stdout":"","stderr":""}))
        first=commit_all(root,"an aggregate beside what is no run of its seeds")
        self.publish(root,{**bound,"note":"written again"})
        commit_all(root,"the aggregate again")
        code,lines=gate(root)
        self.assertIn(f"X900: results/run.json as committed at {first[:12]} reports on seed 17, which has no run record",lines)
        self.assertTrue(any(line.startswith(f"X900: results/run.json as committed at {first[:12]} reports on seed 29, whose run record (") and "holds no outcome" in line for line in lines),lines)

    def test_a_run_record_of_a_seed_outside_the_preregistration_is_refused(self):
        # A seed added once an outcome is seen could count towards what is
        # reported: only the seeds the table froze may have run.
        root=self.tree(status="running")
        ran=commit_all(root)
        runs=self.seeded(root,ran)
        results="experiments/semdb/X900-fixture/results"
        outside=f"results/run-20260301T000000.000000Z-seed-31.json"

        def refused(shown,seed):
            return (f"X900: {shown} ran seed {seed}, which is not one of the preregistered seeds; a listed "
                    "experiment runs only those it froze")

        self.assertEqual(gate(root),(0,[]))
        write(root,f"experiments/semdb/X900-fixture/{outside}",json.dumps(self.record(root,ran,seed=31,status="completed")))
        self.assert_blocked(root,refused(outside,31))
        # A status that saw an outcome is enough, whatever it names.
        for status in ("failed","completed","started","unknown"):
            with self.subTest(status=status):
                write(root,f"experiments/semdb/X900-fixture/{outside}",json.dumps(self.record(root,ran,seed=31,status=status)))
                self.assert_blocked(root,refused(outside,31))
        # A command that failed to launch saw no outcome, and a prepared
        # record names no seed.
        write(root,f"experiments/semdb/X900-fixture/{outside}",json.dumps(self.record(root,ran,seed=31,status="failed-to-launch")))
        self.assertEqual(gate(root),(0,[]))
        # Exactly one of the preregistered integers: another spelling of one
        # is another value.
        for seed,shown in ((17.0,"17.0"),("17",'"17"'),(True,"true"),(None,"null"),([17],"[17]"),(-17,"-17")):
            with self.subTest(seed=seed):
                write(root,f"experiments/semdb/X900-fixture/{outside}",json.dumps(self.record(root,ran,seed=seed,status="completed")))
                lines=gate(root)[1]
                self.assertIn(refused(outside,shown),lines)
        # The preregistered seeds themselves pass, in any results directory
        # the experiment has had.
        (root/f"experiments/semdb/X900-fixture/{outside}").unlink()
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(sorted(name.removeprefix("results/run-2026010").split("T")[0] for name in runs),["2","8"])

    def test_the_registry_holds_the_status_the_manifest_holds_for_a_run(self):
        # A harness reads the registry as it reads the manifest: a run under
        # a registry that says another status ran under other state than the
        # archived one.
        registry="experiments/registry.toml"
        disagreement=lambda held,manifest:(f"X900: the registry holds status {held} for it and experiment.toml "
                                           f"{manifest!r}; a listed experiment holds the same status in both")
        for manifest,held in (("running","planned"),("running","completed"),("prepared","running"),("completed","failed"),
                              ("failed","running")):
            with self.subTest(manifest=manifest,registry=held):
                root=self.tree(status=manifest)
                self.edit(root,registry,f'status = "{manifest}"',f'status = "{held}"')
                message=disagreement(repr(held),manifest)
                self.assert_blocked(root,message,*unarchived(manifest))
                self.assertEqual(mod.launch_errors(root,"X900"),[message])
                commit=commit_all(root)
                self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        # A registry entry that holds no status disagrees as well.
        root=self.tree(status="running")
        self.edit(root,registry,'status = "running"\n','')
        self.assert_blocked(root,disagreement("None","running"))
        # An experiment the registry names twice holds two statuses, of which
        # one agreeing is not enough.
        root=self.tree(status="running")
        text=(root/registry).read_text(encoding="utf-8")
        write(root,registry,text+text.split("\n\n",1)[1].replace('status = "running"','status = "planned"'))
        twice=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,twice,"X900","experiments/semdb/X900-fixture"))
        write(root,registry,text+text.split("\n\n",1)[1])
        same=commit_all(root,"twice, alike")
        self.assertIsNone(mod.launchable_at(root,same,"X900","experiments/semdb/X900-fixture"))
        # Where they agree, the commit could launch, and a run at a commit
        # where they did not is named by the commit it ran at.
        root=self.tree(status="running")
        agreed=commit_all(root)
        self.assertIsNotNone(mod.launchable_at(root,agreed,"X900","experiments/semdb/X900-fixture"))
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        self.edit(root,registry,'status = "running"','status = "planned"')
        split=commit_all(root,"registry moved alone")
        self.assertIsNone(mod.launchable_at(root,split,"X900","experiments/semdb/X900-fixture"))
        self.edit(root,registry,'status = "planned"','status = "running"')
        commit_all(root,"registry restored")
        write(root,self.RECORD,json.dumps(self.record(root,split)))
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        self.assert_blocked(
            root,
            f"X900: {name} ran at {split[:12]}, where the registry holds status 'planned' for it and experiment.toml "
            "'running'; a listed experiment holds the same status in both",
        )

    def test_a_run_record_reached_through_a_symlink_is_refused(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        # The record is a link to a file whose name is no run record's, so
        # that file could be rewritten without touching the record's path.
        target="experiments/semdb/X900-fixture/results/data/seed-17.json"
        write(root,target,json.dumps(self.record(root,ran)))
        self.link(Path("data/seed-17.json"),root/self.RECORD)
        commit_all(root,"records")
        self.assert_blocked(root,f"X900: {name} is not a regular file reached through no symlink")
        # So is an aggregate.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,"experiments/semdb/X900-fixture/results/data/aggregate.json",json.dumps(self.aggregate(root,ran)))
        self.link(Path("data/aggregate.json"),root/self.AGGREGATE)
        commit_all(root,"aggregate")
        self.assert_blocked(root,"X900: results/run.json is not a regular file reached through no symlink")

    def test_what_the_preregistration_freezes_lies_outside_the_results_directory(self):
        # The runner holds the experiment's files to HEAD except its results
        # directory, where runs write: a frozen file or baseline there could
        # change during a run unseen.
        path="experiments/semdb/X900-fixture/results/PROTOCOL.md"
        text="# Protocol\n"
        table={**TABLE,"protocol":path,"protocol_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()}
        root=self.tree(table=table,required={**REQUIRED,"protocol":"file"})
        write(root,path,text)
        self.assert_blocked(
            root,
            f"X900: preregistration key protocol names {path!r}, which lies in the results directory, whose files the runner does not hold to HEAD",
        )
        self.assertEqual(mod.launch_errors(root,"X900"),[
            f"X900: preregistration key protocol names {path!r}, which lies in the results directory, whose files the runner does not hold to HEAD",
        ])
        # Nor may a baseline's directory lie in it, or hold it.
        for config in ("experiments/semdb/X900-fixture/results/baseline/config.toml","experiments/semdb/X900-fixture/baseline.toml"):
            with self.subTest(config=config):
                directory=config.rsplit("/",1)[0]
                root=self.tree(baselines=({**BASELINE,"path":config},))
                shutil.copyfile(root/BASELINE_PATH,(root/config).parent.mkdir(parents=True,exist_ok=True) or root/config)
                self.assert_blocked(
                    root,
                    f"X900: baseline {directory}/ shares files with the results directory experiments/semdb/X900-fixture/results/, "
                    "whose files the runner does not hold to HEAD",
                )
                # A commit holding it could not launch, and froze nothing.
                commit=commit_all(root)
                self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        # Once results_dir names another directory, the same file is frozen
        # like any other.
        root=self.tree(table=table,required={**REQUIRED,"protocol":"file"})
        write(root,path,text)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "records"\n')
        self.assertEqual(gate(root),(0,[]))

    def test_a_record_rewritten_on_one_side_of_a_merge_fails(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        git(root,"checkout","-q","-b","topic")
        git(root,"checkout","-q","main")
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        # On a topic branch the record is written otherwise, the branch
        # takes main in with its own side of the conflict, and main takes
        # the branch in: the record's path changed on one side only of each
        # merge that git's default history follows.
        git(root,"checkout","-q","topic")
        write(root,self.RECORD,json.dumps(self.record(root,ran,note="rewritten")))
        commit_all(root,"rewritten")
        subprocess.run(["git","-c","user.name=PTR tests","-c","user.email=tests@example.invalid","merge","-q","--no-edit","main"],
                       cwd=root,capture_output=True,check=False)
        write(root,self.RECORD,json.dumps(self.record(root,ran,note="rewritten")))
        commit_all(root,"resolved")
        git(root,"checkout","-q","main")
        git(root,"merge","-q","--no-edit","topic")
        self.assert_blocked(root,f"X900: {name} was changed after it was committed (2 versions of it were committed)")

    def test_a_committed_freeze_binds_whether_or_not_its_runs_were_kept(self):
        # Frozen and committed, the experiment could be run; its record was
        # discarded, and the preregistration rewritten and run again.
        root=self.tree(status="running")
        frozen=commit_all(root)
        before=self.frozen(root)
        rewritten=self.tree(status="running",table={**TABLE,"schema":2})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        rewrite=commit_all(root,"rewrite")
        write(root,self.RECORD,json.dumps(self.record(root,rewrite)))
        commit_all(root,"records")
        self.assertNotEqual(before,self.frozen(root))
        errors=[
            f"X900 was frozen at {frozen[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {frozen[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        ]
        self.assert_blocked(root,*errors)
        self.assertEqual(sorted(mod.launch_errors(root,"X900")),sorted(errors))
        # The first commit that could launch it may change one file alone:
        # here a manifest committed ahead named the digest of a table that
        # only a later commit of config.toml put in place.
        second=self.tree(status="running",table={**TABLE,"schema":2})
        root=self.tree(status="running",digest=self.frozen(second)[0])
        commit_all(root)
        shutil.copyfile(second/"experiments/semdb/X900-fixture/config.toml",root/"experiments/semdb/X900-fixture/config.toml")
        launchable=commit_all(root,"config alone")
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        third=self.tree(status="running",table={**TABLE,"schema":3})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(third/relative,root/relative)
        commit_all(root,"rewrite")
        self.assert_blocked(
            root,
            f"X900 was frozen at {launchable[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {launchable[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        )
        # So may a commit that unblocks its baseline alone.
        fixed=self.tree(status="running")
        frozen_digest=mod.load(fixed/"experiments/semdb/X900-fixture/config.toml")["preregistration"]["baseline_fixture_sha256"]
        root=self.tree(status="running",baseline_config={**BASELINE_CONFIG,"status":"blocked-unpinned"},
                       table={**TABLE,"baseline_fixture_sha256":frozen_digest})
        commit_all(root)
        shutil.copyfile(fixed/BASELINE_PATH,root/BASELINE_PATH)
        launchable=commit_all(root,"baseline alone")
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(third/relative,root/relative)
        commit_all(root,"rewrite")
        self.assert_blocked(
            root,
            f"X900 was frozen at {launchable[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {launchable[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        )
        # Or one that puts a preregistered file kept outside the experiment's
        # directory in place.
        protocol="docs/protocols/x900.md"
        frozen_text="# Protocol\n"
        table={**TABLE,"protocol":protocol,"protocol_sha256":hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()}
        root=self.tree(status="running",table=table,required={**REQUIRED,"protocol":"file"})
        write(root,protocol,"# Draft\n")
        commit_all(root)
        write(root,protocol,frozen_text)
        launchable=commit_all(root,"protocol alone")
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        rewritten=self.tree(status="running",table={**table,"schema":3},required={**REQUIRED,"protocol":"file"})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        commit_all(root,"rewrite")
        self.assert_blocked(
            root,
            f"X900 was frozen at {launchable[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {launchable[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        )
        # Or one that lists it, or registers it where it is, alone.
        full=self.tree(status="running")
        for relative in ("experiments/preregistration.toml","experiments/registry.toml"):
            with self.subTest(alone=relative):
                root=self.tree(status="running")
                write(root,relative,"version = 1\n")
                commit_all(root)
                shutil.copyfile(full/relative,root/relative)
                launchable=commit_all(root,f"{relative} alone")
                self.assertEqual(mod.launch_errors(root,"X900"),[])
                for rewritten in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
                    shutil.copyfile(third/rewritten,root/rewritten)
                commit_all(root,"rewrite")
                self.assert_blocked(
                    root,
                    f"X900 was frozen at {launchable[:12]}, whose config.toml holds another [preregistration] than the frozen one",
                    f"X900 was frozen at {launchable[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
                )
        # Nor did one the runner could not launch for its results
        # directory: named `.git` or `.`, or not as git writes a path.
        for results_dir in (".git",".","results/.GIT","results/"):
            with self.subTest(results_dir=results_dir):
                root=self.tree(status="running")
                write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+f'results_dir = "{results_dir}"\n')
                commit_all(root)
                shutil.copyfile(full/self.MANIFEST,root/self.MANIFEST)
                commit_all(root,"results below the experiment")
                self.assertEqual(gate(root),(0,[]))
        # The tree check names why for git's own directory.
        root=self.tree(status="running")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "results/.GIT"\n')
        self.assert_blocked(root,"X900: results_dir 'results/.GIT' passes through git's own directory, where no record can be committed")
        # Or one whose table froze a file in its results directory, whose
        # files the runner does not hold to HEAD.
        inside="experiments/semdb/X900-fixture/results/PROTOCOL.md"
        outside="experiments/semdb/X900-fixture/PROTOCOL.md"
        protocol_text="# Protocol\n"
        digest=hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
        required={**REQUIRED,"protocol":"file"}
        root=self.tree(status="running",table={**TABLE,"protocol":inside,"protocol_sha256":digest},required=required)
        write(root,inside,protocol_text)
        commit_all(root)
        moved=self.tree(status="running",table={**TABLE,"protocol":outside,"protocol_sha256":digest},required=required)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(moved/relative,root/relative)
        (root/inside).unlink()
        write(root,outside,protocol_text)
        commit_all(root,"protocol outside the results")
        self.assertEqual(gate(root),(0,[]))
        # So may moving the results directory away from the protocol.
        root=self.tree(status="running",table={**TABLE,"protocol":inside,"protocol_sha256":digest},required=required)
        write(root,inside,protocol_text)
        commit_all(root)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "records"\n')
        commit_all(root,"results in records")
        self.assertEqual(gate(root),(0,[]))
        # Or one that held a file where its results directory would be.
        root=self.tree(status="running")
        write(root,"experiments/semdb/X900-fixture/results","not a directory\n")
        blocked=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,blocked,"X900","experiments/semdb/X900-fixture"))
        (root/"experiments/semdb/X900-fixture/results").unlink()
        cleared=commit_all(root,"results is a directory again")
        self.assertIsNotNone(mod.launchable_at(root,cleared,"X900","experiments/semdb/X900-fixture"))
        # A commit that could not launch it did not freeze it: here the
        # table still held a placeholder.
        root=self.tree(status="running",table={**TABLE,"harness":"must-be-pinned-before-prepared"})
        commit_all(root)
        fixed=self.tree(status="running")
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(fixed/relative,root/relative)
        commit_all(root,"pinned")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one whose table held a required value of the wrong type.
        root=self.tree(status="running",table={**TABLE,"schema":"1"})
        commit_all(root)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(fixed/relative,root/relative)
        commit_all(root,"typed")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one whose manifest named no digest of it yet.
        root=self.tree(status="running",digest=None)
        commit_all(root)
        shutil.copyfile(fixed/self.MANIFEST,root/self.MANIFEST)
        commit_all(root,"digest named")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one whose manifest's seeds were the table's only as ==
        # reads them.
        root=self.tree(status="running",seeds=(17.0,29))
        commit_all(root)
        shutil.copyfile(fixed/self.MANIFEST,root/self.MANIFEST)
        commit_all(root,"integer seeds")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one whose baseline was still blocked.
        blocked={**BASELINE_CONFIG,"status":"blocked-unpinned"}
        root=self.tree(status="running",baseline_config=blocked,table={**TABLE,"baseline_fixture_sha256":baseline_digest(
            {"config.toml":(self.tree(baseline_config=blocked)/BASELINE_PATH).read_text(encoding="utf-8")})})
        commit_all(root)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST,BASELINE_PATH):
            shutil.copyfile(fixed/relative,root/relative)
        commit_all(root,"unblocked")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one where the list did not name it yet.
        root=self.tree(status="running",listed_text="version = 1\n")
        commit_all(root)
        listed=self.tree(status="running")
        for relative in ("experiments/preregistration.toml",self.MANIFEST):
            shutil.copyfile(listed/relative,root/relative)
        commit_all(root,"listed")
        self.assertEqual(gate(root),(0,[]))
        # Once frozen in a commit, it does not go back to planned.
        root=self.tree(status="prepared")
        frozen=commit_all(root)
        self.edit(root,self.MANIFEST,'status = "prepared"','status = "planned"')
        self.assert_blocked(
            root,
            f"X900 is 'planned', but it was 'prepared' at {frozen[:12]}; a listed experiment that has left planned "
            "stays prepared, running, completed or failed, or is superseded",
        )

    def test_a_freeze_in_a_directory_the_registry_has_left_is_one_the_experiment_left_planned_after(self):
        # Frozen in one directory, and placed by the registry in another
        # where it is planned, the experiment has left planned all the same.
        root=self.tree(status="prepared")
        frozen=commit_all(root)
        git(root,"mv","experiments/semdb/X900-fixture","experiments/semdb/X900-moved")
        self.edit(root,"experiments/registry.toml",'path = "semdb/X900-fixture"','path = "semdb/X900-moved"')
        for relative in ("experiments/semdb/X900-moved/experiment.toml","experiments/registry.toml"):
            self.edit(root,relative,'status = "prepared"','status = "planned"')
        commit_all(root,"moved, and planned there")
        self.assert_blocked(
            root,
            f"X900 is 'planned', but it was 'prepared' at {frozen[:12]}; a listed experiment that has left planned "
            "stays prepared, running, completed or failed, or is superseded",
        )

    def test_the_freezes_of_a_listing_are_one_for_each_state_named_by_its_oldest_commit_and_ordered_by_it(self):
        # Newest first: c3 and c1 hold the state A, c2 holds B, and c0 holds
        # nothing frozen. A is one freeze, named by c1, its oldest commit, and
        # c2 is newer than that: the freezes are ordered by the oldest commit
        # of each state, newest first.
        listing=[("c3",("prepared","A")),("c2",("running","B")),("c1",("prepared","A")),("c0",None)]
        self.assertEqual(mod.freezes(listing),[("c2","running"),("c1","prepared")])
        # A state in one commit only is named by it, and no commit that holds
        # nothing frozen is a freeze.
        self.assertEqual(mod.freezes([("c2",("prepared","A")),("c1",None),("c0",("running","B"))]),
                         [("c2","prepared"),("c0","running")])
        self.assertEqual(mod.freezes([("c1",None),("c0",None)]),[])
        self.assertEqual(mod.freezes([]),[])
        # The same status in two states is two freezes.
        self.assertEqual(mod.freezes([("c1",("prepared","B")),("c0",("prepared","A"))]),
                         [("c1","prepared"),("c0","prepared")])

    def test_the_freeze_named_after_a_return_to_planned_is_the_one_whose_oldest_commit_is_the_newest(self):
        # c1 freezes the experiment in one directory, c2 in another the
        # registry moved it to, and c3 moves it back. The two states have
        # oldest commits c1 and c2, so the one c2 holds is the newest freeze,
        # and the gate names it when the experiment then goes back to planned.
        root=self.tree(status="prepared")
        first=commit_all(root)
        git(root,"mv","experiments/semdb/X900-fixture","experiments/semdb/X900-moved")
        self.edit(root,"experiments/registry.toml",'path = "semdb/X900-fixture"','path = "semdb/X900-moved"')
        second=commit_all(root,"moved to another directory")
        git(root,"mv","experiments/semdb/X900-moved","experiments/semdb/X900-fixture")
        self.edit(root,"experiments/registry.toml",'path = "semdb/X900-moved"','path = "semdb/X900-fixture"')
        commit_all(root,"moved back")
        self.assertEqual(mod.frozen_commits(root,"X900","experiments/semdb/X900-fixture"),
                         [(second,"prepared"),(first,"prepared")])
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,'status = "prepared"','status = "planned"')
        self.assert_blocked(
            root,
            f"X900 is 'planned', but it was 'prepared' at {second[:12]}; a listed experiment that has left planned "
            "stays prepared, running, completed or failed, or is superseded",
        )

    def move_to(self, root: Path, then: str, now: str) -> str:
        """Commit the manifest and the registry of `root`, which hold the
        status `then`, at the status `now`, and return the commit."""
        for relative in (self.MANIFEST,"experiments/registry.toml"):
            self.edit(root,relative,f'status = "{then}"',f'status = "{now}"')
        return commit_all(root,f"{then} to {now}")

    def regression(self, then: str, anchor: str, now: str | None, commit: str) -> str:
        """The refusal of a status `now` at `commit` after `then` at `anchor`."""
        return (f"X900 was {then!r} at {anchor[:12]} and is {now!r} at {commit[:12]}, a commit after it; a status moves "
                "only from prepared to running to completed or failed, or to superseded, and never back, whether or not "
                "a later commit puts it there again")

    def regressions(self, root: Path) -> list[tuple[str, str, str, str]]:
        """The (then, anchor, now, commit) of each regression the gate names
        on `root`, in any order; anything else the gate names fails the test,
        but for the archive a completed fixture does not hold."""
        code,lines=gate(root)
        found=[]
        for line in lines:
            if line in unarchived("completed") or ONCE_COMPLETED.fullmatch(line):
                continue
            named=re.fullmatch(
                r"X900 was '(\w+)' at ([0-9a-f]{12}) and is '(\w+)' at ([0-9a-f]{12}), a commit after it; .*",line,
            )
            self.assertIsNotNone(named,line)
            found.append(named.groups())
        self.assertEqual(code,1 if lines else 0)
        return found

    def test_a_status_that_went_back_after_a_freeze_is_named_though_a_later_commit_restores_it(self):
        # A status moves forward only; a commit that holds an earlier one
        # after a freeze could have let its runs be chosen, whatever the
        # commits after it hold.
        for path,named in (
            (("prepared","planned","prepared"),(1,)),
            (("prepared","running","prepared","running"),(2,)),
            (("prepared","completed","running","completed"),(2,)),
            (("prepared","completed","failed","superseded"),(2,)),
            (("prepared","failed","completed","superseded"),(2,)),
            (("prepared","superseded","prepared"),(2,)),
            (("prepared","running","superseded","running"),(3,)),
        ):
            with self.subTest(path=path):
                root=self.tree(status=path[0])
                commits=[commit_all(root)]
                self.assertEqual(gate(root),(0,[]))
                for then,now in zip(path,path[1:]):
                    commits.append(self.move_to(root,then,now))
                found=self.regressions(root)
                # One error for each commit that went back, naming the
                # status it went back from at a commit before it.
                self.assertEqual(sorted(commit for *_,commit in found),sorted(commits[index][:12] for index in named))
                names=[held[:12] for held in commits]
                for then,anchor,now,commit in found:
                    index=names.index(commit)
                    self.assertEqual(now,path[index])
                    before=names.index(anchor)
                    self.assertLess(before,index)
                    self.assertEqual(then,path[before])
                    self.assertFalse(mod.may_follow(then,now))
        # The nearest freeze is named when several are broken at once.
        root=self.tree(status="prepared")
        first=commit_all(root)
        second=self.move_to(root,"prepared","running")
        third=self.move_to(root,"running","planned")
        fourth=self.move_to(root,"planned","running")
        found=self.regressions(root)
        self.assertEqual([commit for *_,commit in found],[third[:12]])
        self.assertIn(found[0][1],(first[:12],second[:12]))
        self.assertEqual(fourth,git(root,"rev-parse","HEAD"))

    def test_the_descendants_of_a_freeze_are_read_once_however_many_commits_are_compared_with_it(self):
        # One `git rev-list` for each commit a check is anchored at, not one
        # for each commit compared with it.
        root=self.tree(status="prepared")
        commit_all(root)
        for then,now in (("prepared","running"),("running","prepared"),("prepared","running"),("running","prepared"),("prepared","running")):
            self.move_to(root,then,now)
        reads=[]
        real=mod.descendants_of
        def counted(where: Path, commit: str) -> set[str]:
            reads.append(commit)
            return real(where,commit)
        with mock.patch.object(mod,"descendants_of",counted):
            found=self.regressions(root)
        self.assertEqual(len(found),2)
        self.assertTrue(reads)
        self.assertEqual(len(reads),len(set(reads)))

    def test_a_status_no_freeze_came_before_moves_freely(self):
        # Nothing was frozen, so nothing was run, and nothing could be chosen.
        root=self.tree(status="planned")
        commit_all(root)
        self.move_to(root,"planned","superseded")
        self.move_to(root,"superseded","planned")
        self.assertEqual(gate(root),(0,[]))
        # A commit on a side line that never saw the freeze did not go back
        # from it, and the merge that brings the two together holds the
        # frozen status.
        root=self.tree(status="planned")
        start=commit_all(root)
        self.move_to(root,"planned","prepared")
        git(root,"checkout","-q","-b","side",start)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# a note\n")
        commit_all(root,"a note on the side line")
        git(root,"checkout","-q","main")
        git(root,"merge","-q","--no-ff","-m","merge","side")
        self.assertEqual(gate(root),(0,[]))

    def test_a_status_that_went_back_on_a_side_line_is_named_whatever_a_merge_keeps(self):
        # Merged with a strategy that keeps the main line's tree: nothing at
        # the head shows the side line went back.
        root=self.tree(status="prepared")
        frozen=commit_all(root)
        git(root,"checkout","-q","-b","side",frozen)
        back=self.move_to(root,"prepared","planned")
        git(root,"checkout","-q","main")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# a note\n")
        commit_all(root,"a note on the main line")
        git(root,"merge","-q","--no-ff","-s","ours","-m","merge","side")
        self.assertEqual(self.regressions(root),[("prepared",frozen[:12],"planned",back[:12])])
        # Merged so that the side line's later commit restores the status,
        # the going back is reachable through the merge's second parent.
        root=self.tree(status="prepared")
        frozen=commit_all(root)
        git(root,"checkout","-q","-b","side",frozen)
        back=self.move_to(root,"prepared","planned")
        self.move_to(root,"planned","prepared")
        git(root,"checkout","-q","main")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# a note\n")
        commit_all(root,"a note on the main line")
        git(root,"merge","-q","--no-ff","-m","merge","side")
        self.assertEqual(self.regressions(root),[("prepared",frozen[:12],"planned",back[:12])])

    def test_a_status_that_is_no_string_in_the_history_is_named_not_a_crash(self):
        # A commit after a freeze whose manifest holds a list, a table, a
        # number or a bool as its status is one that went back (to no status
        # at all), whatever it holds and whatever a later commit restores;
        # what it holds is never looked up as a status name.
        for spelling in ('["prepared"]',"{ prepared = 1 }","1","true","1.5"):
            with self.subTest(spelling=spelling):
                root=self.tree(status="prepared")
                frozen=commit_all(root)
                self.edit(root,self.MANIFEST,'status = "prepared"',f"status = {spelling}")
                broken=commit_all(root,"status holds no string")
                self.edit(root,self.MANIFEST,f"status = {spelling}",'status = "prepared"')
                restored=commit_all(root,"status restored")
                code,lines=gate(root)
                self.assertEqual((code,lines),(1,[self.regression("prepared",frozen,None,broken)]))
                self.assertEqual(restored,git(root,"rev-parse","HEAD"))

    def test_a_commit_holding_no_readable_manifest_holds_no_status(self):
        # A manifest that is absent or does not parse at a commit between two
        # commits that hold prepared says nothing of a status: it is neither
        # a status that went back nor one that stayed.
        for spelling in (None,"status = = 1\n"):
            with self.subTest(spelling=spelling):
                root=self.tree(status="prepared")
                kept=(root/self.MANIFEST).read_text(encoding="utf-8")
                frozen=commit_all(root)
                if spelling is None:
                    (root/self.MANIFEST).unlink()
                else:
                    write(root,self.MANIFEST,spelling)
                commit_all(root,"no readable manifest")
                write(root,self.MANIFEST,kept)
                commit_all(root,"manifest restored")
                self.assertEqual(mod.statuses_at(root,frozen,"X900"),["prepared"])
                self.assertEqual(mod.statuses_at(root,git(root,"rev-parse","HEAD~1"),"X900"),[])
                self.assertEqual(mod.statuses_at(root,"HEAD","X900"),["prepared"])
                self.assertEqual(gate(root),(0,[]))

    def test_a_commit_holds_a_status_for_every_directory_the_registry_places_the_experiment_in(self):
        # The registry places X900 in two directories at one commit; each
        # holds a manifest, and the commit holds the status of each, in the
        # order of the sorted directories.
        root=self.tree(status="prepared")
        shutil.copytree(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900-twin")
        self.edit(root,"experiments/semdb/X900-twin/experiment.toml",'status = "prepared"','status = "running"')
        write(root,"experiments/registry.toml",
              (root/"experiments/registry.toml").read_text(encoding="utf-8")
              +'\n[[experiment]]\nid = "X900"\npath = "semdb/X900-twin"\nstatus = "running"\n')
        placed=commit_all(root,"placed in two directories")
        self.assertEqual(mod.placed_directories(root,placed,"X900"),
                         ["experiments/semdb/X900-fixture","experiments/semdb/X900-twin"])
        self.assertEqual(mod.statuses_at(root,placed,"X900"),["prepared","running"])

    def two_directories(self, root: Path, twin_status: str, fixture_status: str = "running") -> str:
        """Commit a second directory for X900, in the registry too, whose
        manifest holds `twin_status`, beside the first, which holds
        `fixture_status`, and return the commit."""
        shutil.copytree(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900-twin")
        self.edit(root,"experiments/semdb/X900-twin/experiment.toml",f'status = "{fixture_status}"',f'status = "{twin_status}"')
        write(root,"experiments/registry.toml",
              (root/"experiments/registry.toml").read_text(encoding="utf-8")
              +f'\n[[experiment]]\nid = "X900"\npath = "semdb/X900-twin"\nstatus = "{twin_status}"\n')
        return commit_all(root,"placed in two directories")

    def test_a_status_the_registry_places_beside_the_frozen_one_is_checked_against_the_freeze_behind_it(self):
        # X900 runs in one directory, and a later commit places it in a
        # second, prepared, beside it: the first status follows the freeze,
        # the second goes back, and each status the commit holds is checked,
        # not the first alone.
        root=self.tree(status="running")
        first=commit_all(root)
        child=self.two_directories(root,"prepared")
        self.assertEqual(mod.statuses_at(root,child,"X900"),["running","prepared"])
        self.assertEqual(
            mod.status_regressions(root,"X900",[(child,None),(first,("running","D"))]),
            [self.regression("running",first,"prepared",child)],
        )
        # A commit is no descendant of itself in this sense: a freeze that
        # holds both statuses at once is one commit, and went back from
        # nothing.
        self.assertEqual(mod.status_regressions(root,"X900",[(child,("running","D"))]),[])

    def test_a_status_superseded_beside_a_frozen_one_is_no_final_status_of_its_own(self):
        # A superseded status is final only where a freeze is behind it, and
        # a commit that is a freeze holds its own status frozen, not the
        # superseded one another directory of it holds at once.
        root=self.tree(status="running")
        registry=(root/"experiments/registry.toml").read_text(encoding="utf-8")
        both=self.two_directories(root,"superseded")
        self.assertEqual(mod.statuses_at(root,both,"X900"),["running","superseded"])
        shutil.rmtree(root/"experiments/semdb/X900-twin")
        write(root,"experiments/registry.toml",registry)
        later=commit_all(root,"one directory again")
        self.assertEqual(mod.statuses_at(root,later,"X900"),["running"])
        self.assertEqual(mod.status_regressions(root,"X900",[(later,None),(both,("running","D"))]),[])

    def test_the_descendants_of_a_commit_are_all_of_them_on_head_and_a_commit_git_cannot_read_raises(self):
        root=self.tree(status="prepared")
        first=commit_all(root)
        git(root,"checkout","-q","-b","side",first)
        write(root,"side.md","side\n")
        side=commit_all(root,"side")
        git(root,"checkout","-q","main")
        write(root,"main.md","main\n")
        main=commit_all(root,"main")
        git(root,"merge","-q","--no-ff","-m","merge","side")
        merged=git(root,"rev-parse","HEAD")
        # A commit that descends from the first and is on no line of HEAD's
        # history is no descendant of it there.
        git(root,"checkout","-q","-b","elsewhere",first)
        write(root,"elsewhere.md","elsewhere\n")
        commit_all(root,"elsewhere")
        git(root,"checkout","-q","main")
        # Every commit below it, a merge for both of its lines, by full name,
        # and never the commit itself.
        self.assertEqual(mod.descendants_of(root,first),{side,main,merged})
        self.assertEqual(mod.descendants_of(root,side),{merged})
        self.assertEqual(mod.descendants_of(root,main),{merged})
        self.assertEqual(mod.descendants_of(root,merged),set())
        # A name git cannot resolve is a history that cannot be read, not one
        # that holds nothing.
        with self.assertRaises(mod.HistoryUnreadable) as caught:
            mod.descendants_of(root,"0"*40)
        self.assertTrue(str(caught.exception))
        # Nor is a name no process can be given one that names no descendants.
        with self.assertRaises(mod.HistoryUnreadable) as caught:
            mod.descendants_of(root,"a\0b")
        self.assertTrue(str(caught.exception))

    def test_the_history_is_listed_once_for_each_freeze_and_not_once_for_each_commit_that_is_listed(self):
        # Commits that hold the experiment planned come before its freezes, so
        # each holds a status no freeze may be followed by, and none descends
        # from a freeze: a listing of the history for each of them would make
        # the gate that the runner calls at every launch as slow as the
        # history is long.
        listings={}
        for planned in (2,9):
            with self.subTest(planned=planned):
                root=self.tree(status="planned")
                commit_all(root)
                for step in range(planned):
                    write(root,"experiments/registry.toml",
                          (root/"experiments/registry.toml").read_text(encoding="utf-8")+f"# planned {step}\n")
                    commit_all(root,f"planned {step}")
                self.move_to(root,"planned","prepared")
                self.move_to(root,"prepared","running")
                asked=[]
                original=mod.experiment_records.git
                def counting(where,*arguments,**options):
                    asked.append(arguments)
                    return original(where,*arguments,**options)
                with mock.patch.object(mod.experiment_records,"git",side_effect=counting):
                    self.assertEqual(gate(root),(0,[]))
                listings[planned]=[arguments for arguments in asked if arguments[0]=="rev-list"]
        self.assertEqual(len(listings[2]),2)
        self.assertEqual(len(listings[9]),2)

    def test_a_superseded_status_is_final_for_a_freeze_that_comes_after_it_and_not_only_for_the_newest_freeze(self):
        # The experiment is frozen as prepared, superseded, and then frozen as
        # running: the superseded commit descends from the first freeze but
        # not from the newest one, and the running commit after it is named.
        root=self.tree(status="prepared")
        commit_all(root)
        gone=self.move_to(root,"prepared","superseded")
        ran=self.move_to(root,"superseded","running")
        self.assertEqual(self.regressions(root),[("superseded",gone[:12],"running",ran[:12])])

    def test_a_status_that_went_back_at_a_merge_is_named_though_only_its_second_parent_saw_the_freeze(self):
        # The main line holds prepared and never saw the side line run; the
        # merge keeps prepared, and holds more than either parent so that it
        # is a commit of its own. Behind its first parent alone nothing ran.
        root=self.tree(status="prepared")
        start=commit_all(root)
        git(root,"checkout","-q","-b","side",start)
        ran=self.move_to(root,"prepared","running")
        git(root,"checkout","-q","main")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# a note on the main line\n")
        commit_all(root,"a note on the main line")
        git(root,"merge","-q","--no-ff","--no-commit","side")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# merged\n")
        merged=self.move_to(root,"running","prepared")
        self.assertEqual(len(git(root,"rev-list","--parents","-n","1","HEAD").split()),3)
        # A commit after the merge runs it again, so that only the merge went back.
        self.move_to(root,"prepared","running")
        self.assertEqual(self.regressions(root),[("running",ran[:12],"prepared",merged[:12])])

    def test_a_superseded_status_no_freeze_is_behind_is_no_final_status_for_a_merge_that_brings_a_freeze(self):
        # The side line supersedes the experiment before anything froze it,
        # and the main line freezes it as prepared. The side line's commit has
        # no freeze behind it, so superseded is not final there, and the merge
        # of the two, which keeps prepared, is no status that went back.
        root=self.tree(status="planned")
        start=commit_all(root)
        git(root,"checkout","-q","-b","side",start)
        superseded=self.move_to(root,"planned","superseded")
        git(root,"checkout","-q","main")
        frozen=self.move_to(root,"planned","prepared")
        git(root,"merge","-q","--no-ff","--no-commit","-s","ours","side")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"# merged\n")
        merged=commit_all(root,"merge")
        self.assertEqual(len(git(root,"rev-list","--parents","-n","1","HEAD").split()),3)
        self.assertEqual(mod.statuses_at(root,superseded,"X900"),["superseded"])
        self.assertEqual(mod.descendants_of(root,start),{superseded,frozen,merged})
        self.assertEqual(self.regressions(root),[])
        # Where a freeze is behind the superseded status, it is final, and a
        # commit after it that holds the frozen status again is named.
        root=self.tree(status="prepared")
        frozen=commit_all(root)
        gone=self.move_to(root,"prepared","superseded")
        back=self.move_to(root,"superseded","prepared")
        self.assertEqual(self.regressions(root),[("superseded",gone[:12],"prepared",back[:12])])

    def test_a_status_follows_another_only_forward(self):
        allowed={
            "prepared":("prepared","running","completed","failed","superseded"),
            "running":("running","completed","failed","superseded"),
            "completed":("completed","superseded"),
            "failed":("failed","superseded"),
            "superseded":("superseded",),
        }
        every=("planned","prepared","running","completed","failed","superseded",None,"other")
        for then in every:
            for now in every:
                with self.subTest(then=then,now=now):
                    self.assertEqual(mod.may_follow(then,now),now in allowed.get(then,()))

    def test_a_listed_experiment_names_its_entrypoint_before_it_leaves_planned(self):
        # The manifest is frozen from the first commit past planned, so an
        # entrypoint named only later could never be named at all.
        refusal=("X900: experiment.toml names no entrypoint; a listed experiment names what it runs before it "
                 "leaves planned, since its manifest is frozen from then on")
        # Empty, absent, or no string: each is the one refusal, and no
        # command is read from it.
        for spelling in ('entrypoint = ""\n',"","entrypoint = 1\n"):
            with self.subTest(spelling=spelling):
                root=self.tree(status="prepared")
                self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n',spelling)
                self.assert_blocked(root,refusal)
                self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        root=self.tree(status="prepared")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = ""\n')
        # A commit without one could not launch it, and froze nothing.
        commit_all(root)
        self.edit(root,self.MANIFEST,'entrypoint = ""\n','entrypoint = "bench <seed>"\n')
        commit_all(root,"entrypoint named")
        self.assertEqual(gate(root),(0,[]))

    def test_a_record_reverted_and_restored_is_the_record_committed(self):
        root,ran=self.ran()
        recorded=git(root,"rev-parse","HEAD")
        git(root,"revert","--no-edit",recorded)
        # While it is missing, the gate fails and nothing launches, so no run
        # or freeze can use the gap.
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        missing=f"X900: {name} was committed and has since been deleted or renamed; a run record stays as it was recorded"
        self.assert_blocked(root,missing)
        self.assertEqual(mod.launch_errors(root,"X900"),[missing])
        git(root,"revert","--no-edit","HEAD")
        self.assertTrue((root/self.RECORD).exists())
        self.assertEqual(gate(root),(0,[]))

    def test_a_listed_experiment_runs_each_seed_once(self):
        # A seed run again after its outcome was seen could keep whichever
        # run came out best.
        root=self.tree(status="running")
        commit=commit_all(root)
        results="experiments/semdb/X900-fixture/results"
        first=f"{results}/run-20260101T000000.000000Z-seed-17.json"
        second=f"{results}/run-20260102T000000.000000Z-seed-17.json"
        write(root,first,json.dumps(self.record(root,commit,seed=17,status="completed")))
        # A command that failed to launch saw no outcome, a prepared record
        # names no seed, and another seed is another run.
        write(root,second,json.dumps(self.record(root,commit,seed=17,status="failed-to-launch")))
        write(root,f"{results}/run-20260103T000000.000000Z-seed-29.json",json.dumps(self.record(root,commit,seed=29,status="completed")))
        write(root,f"{results}/run-20251231T000000.000000Z.json",json.dumps(self.record(root,commit,status="prepared")))
        # Nor is an aggregate that names a seed a run of it.
        self.publish(root,self.aggregate(root,commit,seed=17))
        self.assertEqual(gate(root),(0,[]))
        refusal=("X900: seed 17 ran more than once (results/run-20260101T000000.000000Z-seed-17.json, "
                 "results/run-20260102T000000.000000Z-seed-17.json); a listed experiment runs each seed once, so no "
                 "run of it is chosen by its outcome")
        for status in ("completed","failed","unknown"):
            with self.subTest(status=status):
                write(root,second,json.dumps(self.record(root,commit,seed=17,status=status)))
                self.assert_blocked(root,refusal)
                self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        # Committed, the second run stays, and so does the refusal.
        commit_all(root,"records")
        self.assert_blocked(root,refusal)

    def test_a_listed_experiments_seeds_are_named_once(self):
        # Each seed runs once: a seed named twice would be one run an
        # aggregate counts twice, and once frozen the list could not change.
        refusal=("X900: preregistered seeds name seed 17 more than once; each seed runs once, so an aggregate would count "
                 "its one run twice")
        root=self.tree(status="running",table={**TABLE,"seeds":[17,29,17,17]},seeds=(17,29,17,17))
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        self.assertEqual(mod.repeated_seeds([17,29,17,17,29]),[17,29])
        self.assertEqual(mod.repeated_seeds([17,29]),[])
        self.assertEqual(mod.repeated_seeds("17"),[])
        # A commit holding such a list froze nothing, so naming each seed once
        # repairs it.
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        repaired=self.tree(status="running")
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(repaired/relative,root/relative)
        repair=commit_all(root,"seeds named once")
        self.assertEqual(gate(root),(0,[]))
        self.assertIsNotNone(mod.launchable_at(root,repair,"X900","experiments/semdb/X900-fixture"))

    def test_a_listed_experiments_runs_name_one_program_toolchain_and_environment(self):
        # The program a run starts, the Rust toolchain and the environment lie
        # outside the commit its record names: one changed between seeds
        # would leave records naming one commit for different code, and a
        # program the command starts by name is found through its PATH.
        root=self.tree(status="running")
        commit=commit_all(root)
        results="experiments/semdb/X900-fixture/results"
        program={"path":"/usr/bin/bench","sha256":"a"*64}
        toolchain={"rustc":{"path":"/toolchains/pinned/bin/rustc","sha256":"b"*64},"cargo":{"path":None,"sha256":None}}
        environment={"PATH":"/a:/usr/bin","PYTHONNOUSERSITE":"1"}
        ran={"executable":program,"toolchain":toolchain,"environment":environment}
        names=[f"{results}/run-2026010{day}T000000.000000Z-seed-{seed}.json" for day,seed in ((1,17),(2,29))]
        for name,seed in zip(names,(17,29)):
            write(root,name,json.dumps({**self.record(root,commit,seed=seed,status="completed"),**ran}))
        # A command that failed to launch ran no program.
        write(root,f"{results}/run-20260103T000000.000000Z-seed-31.json",json.dumps(
            {**self.record(root,commit,seed=31,status="failed-to-launch"),"executable":{"path":None,"sha256":None}}))
        self.assertEqual(gate(root),(0,[]))
        shown=[name.removeprefix("experiments/semdb/X900-fixture/") for name in names]
        for changed in ({"executable":{**program,"sha256":"c"*64}},{"toolchain":{**toolchain,"cargo":program}},
                        {"environment":{**environment,"PATH":"/b:/usr/bin"}}):
            with self.subTest(changed=sorted(changed)):
                write(root,names[1],json.dumps({**self.record(root,commit,seed=29,status="completed"),**ran,**changed}))
                errors=gate(root)[1]
                self.assertEqual(len(errors),1,errors)
                self.assertTrue(
                    errors[0].startswith("X900: its runs name 2 programs, toolchains or environments ("),errors[0])
                self.assertIn(shown[0],errors[0])
                self.assertIn(shown[1],errors[0])
                self.assertTrue(errors[0].endswith(
                    "); a program, toolchain or environment changed between seeds lies outside the commit every record "
                    "names"),errors[0])
                self.assertEqual(mod.launch_errors(root,"X900"),errors)

    def test_a_listed_experiments_seeds_ran_one_tree(self):
        # The command may run or read any file of the repository: its seeds
        # ran one tree only if their commits differ in nothing but the seed
        # records of the results directory.
        root=self.tree(status="running")
        first=commit_all(root)
        results="experiments/semdb/X900-fixture/results"
        seventeen=f"{results}/run-20260101T000000.000000Z-seed-17.json"
        twenty_nine=f"{results}/run-20260102T000000.000000Z-seed-29.json"
        write(root,seventeen,json.dumps({**self.record(root,first,seed=17,status="completed"),"started_at":"20260101"}))
        recorded=commit_all(root,"seed 17 recorded")
        write(root,twenty_nine,json.dumps({**self.record(root,recorded,seed=29,status="completed"),"started_at":"20260102"}))
        self.assertEqual(gate(root),(0,[]))
        commit_all(root,"seed 29 recorded")
        self.assertEqual(gate(root),(0,[]))
        # The tools' other outputs are written once the seeds have run.
        for output in ("metrics.json","mutations.json","run.json"):
            write(root,f"{results}/{output}","{}" if output!="run.json" else json.dumps(self.aggregate(
                root,first,mutation_checks=mod.experiment_records.mutation_summary_of(b"{}"))))
        commit_all(root,"aggregated")
        self.assertEqual(gate(root),(0,[]))
        # One committed between two seeds is an input the later seed could
        # read, chosen once the earlier one's outcome was seen.
        for output in ("metrics.json","mutations.json"):
            with self.subTest(output=output):
                root=self.tree(status="running")
                first=commit_all(root)
                write(root,seventeen,json.dumps({**self.record(root,first,seed=17,status="completed"),"started_at":"20260101"}))
                write(root,f"{results}/{output}","{}\n")
                recorded=commit_all(root,"seed 17 recorded")
                write(root,twenty_nine,json.dumps({**self.record(root,recorded,seed=29,status="completed"),"started_at":"20260102"}))
                self.assert_blocked(root,(
                    f"X900: results/run-20260102T000000.000000Z-seed-29.json ran at {recorded[:12]}, whose repository "
                    f"differs from {first[:12]}'s, where results/run-20260101T000000.000000Z-seed-17.json ran, in "
                    f"{results}/{output}; the seeds of a listed experiment run one tree, only their seed records "
                    "committed between them"))
        # A script outside the provenance files, changed between the seeds.
        root=self.tree(status="running")
        first=commit_all(root)
        write(root,seventeen,json.dumps({**self.record(root,first,seed=17,status="completed"),"started_at":"20260101"}))
        commit_all(root,"seed 17 recorded")
        write(root,"scripts/harness.py","print('changed')\n")
        changed=commit_all(root,"harness changed")
        write(root,twenty_nine,json.dumps({**self.record(root,changed,seed=29,status="completed"),"started_at":"20260102"}))
        refusal=(f"X900: results/run-20260102T000000.000000Z-seed-29.json ran at {changed[:12]}, whose repository differs "
                 f"from {first[:12]}'s, where results/run-20260101T000000.000000Z-seed-17.json ran, in scripts/harness.py; "
                 "the seeds of a listed experiment run one tree, only their seed records committed between them")
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        # A command that failed to launch ran no tree.
        write(root,twenty_nine,json.dumps({**self.record(root,changed,seed=29,status="failed-to-launch"),"started_at":"20260102"}))
        self.assertEqual(gate(root),(0,[]))
        # A name that is not UTF-8 among the differences is one like any
        # other, named by its escapes, not a pair of commits left unchecked.
        root=self.tree(status="running")
        first=commit_all(root)
        write(root,seventeen,json.dumps({**self.record(root,first,seed=17,status="completed"),"started_at":"20260101"}))
        commit_all(root,"seed 17 recorded")
        try:
            with open(os.path.join(os.fsencode(root),b"notes\xff.md"),"wb") as handle:
                handle.write(b"x\n")
        except OSError as error:
            self.skipTest(f"cannot name a file with bytes that are not UTF-8: {error}")
        named=commit_all(root,"a name that is not UTF-8")
        write(root,twenty_nine,json.dumps({**self.record(root,named,seed=29,status="completed"),"started_at":"20260102"}))
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertEqual(len(lines),1,lines)
        self.assertIn("whose repository differs from",lines[0])
        self.assertIn("notes\\udcff.md",lines[0])
        # Nor is one the comparison cannot read.
        with mock.patch.object(mod.experiment_records,"code_changes",side_effect=mod.experiment_records.NotUTF8("unreadable")):
            code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(f"X900: results/run-20260102T000000.000000Z-seed-29.json ran at {named[:12]}, which cannot be compared "
                      f"with {first[:12]}, where results/run-20260101T000000.000000Z-seed-17.json ran: unreadable",lines)

    def test_an_experiment_is_registered_once(self):
        # Of two entries of one id, the gate would check one and the runner,
        # which finds both, launch neither.
        root=self.tree(status="running")
        registry=(root/"experiments/registry.toml").read_text(encoding="utf-8")
        shutil.copytree(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900-twin")
        write(root,"experiments/registry.toml",registry+'\n[[experiment]]\nid = "X900"\npath = "semdb/X900-twin"\nstatus = "running"\n')
        refusal=("experiments/registry.toml registers X900 2 times ('semdb/X900-fixture', 'semdb/X900-twin'); an "
                 "experiment is registered once, in one directory")
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        self.assertIn(refusal,gate(root)[1])
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        write(root,"experiments/registry.toml",registry)
        self.assertEqual(gate(root),(0,[]))

    def test_the_commit_a_run_names_decides_whether_it_is_listed(self):
        # The runner reads whether the list names an experiment from the tree
        # before it knows the commit its run would name: that commit decides,
        # and an experiment the list has named on its history runs only as
        # listed, since an unlisted run's outcome could be seen and its record
        # discarded.
        def refusal(commit,held):
            return (f"experiments/preregistration.toml changed while the launch of X900 was checked: {commit[:12]}, the "
                    f"commit its run would name, {held} it; rerun from a tree that holds HEAD")
        root=self.tree(status="running")
        listing=(root/"experiments/preregistration.toml").read_text(encoding="utf-8")
        write(root,"experiments/preregistration.toml","version = 1\n")
        before=commit_all(root,"not yet listed")
        write(root,"experiments/preregistration.toml",listing)
        listed_at=commit_all(root,"listed")
        self.assertEqual(mod.launch_mode_errors(root,"X900",listed_at,True),[])
        self.assertEqual(mod.launch_mode_errors(root,"X900",listed_at,False),[refusal(listed_at,"names")])
        # The history is the commit's: the list named X900 only after it.
        self.assertEqual(mod.launch_mode_errors(root,"X900",before,False),[])
        self.assertEqual(mod.launch_mode_errors(root,"X900",before,True),[refusal(before,"does not name")])
        write(root,"experiments/preregistration.toml","version = 1\n")
        delisted=commit_all(root,"delisted")
        self.assertEqual(mod.launch_mode_errors(root,"X900",delisted,True),[refusal(delisted,"does not name")])
        self.assertEqual(mod.launch_mode_errors(root,"X900",delisted,False),[mod.delisted_error("X900",listed_at)])
        # A list the commit does not hold readable decides nothing.
        write(root,"experiments/preregistration.toml","version = [\n")
        unreadable=commit_all(root,"unreadable")
        for listed in (True,False):
            self.assertEqual(mod.launch_mode_errors(root,"X900",unreadable,listed),[
                f"experiments/preregistration.toml cannot be read as {unreadable[:12]} holds it, so whether X900 "
                "preregisters is unknown"])

    def test_settings_are_compared_by_type_sign_and_offset_and_nan_is_itself(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        config="experiments/semdb/X900-fixture/config.toml"
        # A NaN setting, unchanged since the run, is unchanged.
        root=self.tree(status="running")
        write(root,config,(root/config).read_text(encoding="utf-8").replace('tests_dir = "tests"','tests_dir = "tests"\nloss_cap = nan\nlimits = [nan, 1.0]'))
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        self.assertEqual(gate(root),(0,[]))
        # 1, 1.0 and true are three values, which == takes as one.
        for changed in ("loss_cap = 1","loss_cap = 1.0","loss_cap = true"):
            with self.subTest(changed=changed):
                root=self.tree(status="running")
                write(root,config,(root/config).read_text(encoding="utf-8").replace('tests_dir = "tests"','tests_dir = "tests"\nloss_cap = 1'))
                ran=commit_all(root)
                write(root,self.RECORD,json.dumps(self.record(root,ran)))
                commit_all(root,"records")
                self.edit(root,config,"loss_cap = 1",changed)
                if changed=="loss_cap = 1":
                    self.assertEqual(gate(root),(0,[]))
                    continue
                self.assert_blocked(
                    root,
                    f"X900: {name} ran at {ran[:12]}, whose config.toml differs from the current one in loss_cap; after a run the configuration stays as it ran",
                    f"X900 was frozen at {ran[:12]}, whose config.toml differs from the current one in loss_cap; after a run the configuration stays as it ran",
                )
        # 0.0 and -0.0, which == takes as one, and nan and -nan differ in a
        # sign that TOML holds and a run can read; +0.0 is 0.0. So do one
        # instant at two offsets; Z is +00:00.
        unchanged={"loss_cap = +0.0","loss_cap = 2026-01-01T00:00:00+00:00"}
        for first,changed in (
            ("loss_cap = 0.0","loss_cap = -0.0"),("loss_cap = nan","loss_cap = -nan"),("loss_cap = 0.0","loss_cap = +0.0"),
            ("loss_cap = 2026-01-01T00:00:00Z","loss_cap = 2026-01-01T01:00:00+01:00"),
            ("loss_cap = 2026-01-01T00:00:00Z","loss_cap = 2026-01-01T00:00:00+00:00"),
        ):
            with self.subTest(first=first,changed=changed):
                root=self.tree(status="running")
                write(root,config,(root/config).read_text(encoding="utf-8").replace('tests_dir = "tests"',f'tests_dir = "tests"\n{first}'))
                ran=commit_all(root)
                write(root,self.RECORD,json.dumps(self.record(root,ran)))
                commit_all(root,"records")
                self.edit(root,config,first,changed)
                if changed in unchanged:
                    self.assertEqual(gate(root),(0,[]))
                    continue
                self.assert_blocked(
                    root,
                    f"X900: {name} ran at {ran[:12]}, whose config.toml differs from the current one in loss_cap; after a run the configuration stays as it ran",
                    f"X900 was frozen at {ran[:12]}, whose config.toml differs from the current one in loss_cap; after a run the configuration stays as it ran",
                )
        # A freeze is told from another by the same rule: a commit that held
        # -nan, or a date's text, between two that held nan, or the date,
        # froze a state of its own.
        for first,between in (("loss_cap = nan","loss_cap = -nan"),("loss_cap = 2026-01-02",'loss_cap = "2026-01-02"')):
            with self.subTest(between=between):
                root=self.tree(status="running")
                write(root,config,(root/config).read_text(encoding="utf-8").replace('tests_dir = "tests"',f'tests_dir = "tests"\n{first}'))
                commit_all(root)
                self.edit(root,config,first,between)
                held=commit_all(root,"between")
                self.edit(root,config,between,first)
                commit_all(root,"back")
                self.assert_blocked(
                    root,
                    f"X900 was frozen at {held[:12]}, whose config.toml differs from the current one in loss_cap; after a run the configuration stays as it ran",
                )
        nan=float("nan")

        def at(hour: int, zone) -> datetime.datetime:
            """2026-01-01 at `hour` in `zone` (None for a local datetime)."""
            return datetime.datetime(2026,1,1,hour,tzinfo=zone)

        for then,now,same in (
            (0.0,-0.0,False),(-0.0,-0.0,True),(nan,nan,True),(nan,-nan,False),(1.5,1.5,True),(1.5,-1.5,False),
            ([0.0],[-0.0],False),({"cap":-0.0},{"cap":0.0},False),({"cap":[nan]},{"cap":[nan]},True),
            (at(0,datetime.timezone.utc),at(1,datetime.timezone(datetime.timedelta(hours=1))),False),
            (at(0,datetime.timezone.utc),at(0,datetime.timezone(datetime.timedelta(0))),True),
            (at(0,None),at(0,datetime.timezone.utc),False),(datetime.time(1,2),datetime.time(1,2),True),
        ):
            with self.subTest(then=then,now=now):
                self.assertEqual(mod.same_value(then,now),same)

    def test_malformed_files_are_named_errors_not_crashes(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        config="experiments/semdb/X900-fixture/config.toml"
        # A record whose manifest_sha256 is not a string.
        root,ran=self.ran()
        record=self.record(root,ran,manifest_sha256=["x"])
        write(root,self.RECORD,json.dumps(record))
        self.assert_blocked(
            root,
            f"X900: {name} names manifest_sha256 ['x'], not the SHA-256 of experiment.toml at {ran[:12]}",
            f"X900: {name} differs from the record committed as it",
        )
        # A status that is not a string.
        root=self.tree()
        self.edit(root,self.MANIFEST,'status = "prepared"','status = ["prepared"]')
        self.assert_blocked(root,"X900: status ['prepared'] is not a status")
        self.assertEqual(mod.launch_errors(root,"X900"),[
            "X900 preregisters (experiments/preregistration.toml) and is ['prepared']: it runs only once its "
            "preregistration is frozen and it is prepared, running, completed or failed",
        ])
        # Files that do not parse as TOML.
        for relative,text in (
            (config,"version = 1\n[preregistration\n"),
            (BASELINE_PATH,'{"status": "pinned"}\n'),
            ("experiments/preregistration.toml","version = 1\n[experiment.X900\n"),
            ("experiments/registry.toml","version = 1\n[[experiment\n"),
        ):
            with self.subTest(relative=relative):
                root=self.tree()
                write(root,relative,text)
                code,lines=gate(root)
                self.assertEqual((code,len(lines)),(1,1),lines)
                self.assertTrue(lines[0].startswith(f"{relative} cannot be read as TOML: "),lines)
                refused=mod.launch_errors(root,"X900")
                self.assertEqual(len(refused),1,refused)
                self.assertTrue(refused[0].startswith(f"{relative} cannot be read as TOML: "),refused)
        # A completed experiment whose results_dir is not a path.
        root=self.tree(status="completed")
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+"results_dir = 1\n")
        self.assert_blocked(
            root,
            "X900: results_dir 1 is not a path",
            "X900: results_dir 1 is not a directory below the experiment's directory",
        )

    def test_a_listed_experiments_results_are_checked_against_the_whole_repository(self):
        # Its command may run or read any file of the repository, so its
        # archived results are stale once any of them changes
        # (`experiment_records.listed_staleness_paths`).
        root=self.tree(status="completed")
        seen=[]

        def stale(exp_id,experiment,results,checked_root,listed_experiment):
            seen.append((exp_id,listed_experiment))
            return []

        with (
            mock.patch.object(mod.experiment_records,"staleness_errors",side_effect=stale),
            mock.patch.object(mod.experiment_records,"aggregate_errors",return_value=[]),
        ):
            mod.gate_errors(root)
        self.assertEqual(seen,[("X900",True)])
        self.assertEqual(mod.listed_now(root),{"X900"})
        # A list that cannot be read names none; the gate names why.
        write(root,"experiments/preregistration.toml","version = [\n")
        self.assertEqual(mod.listed_now(root),set())

    def test_a_listed_experiment_stays_listed(self):
        root=self.tree()
        listed=(root/"experiments/preregistration.toml").read_text(encoding="utf-8")
        first=commit_all(root)
        # Taken out of the list, in the tree or committed, it is still gated:
        # the gate fails and the runner refuses it.
        write(root,"experiments/preregistration.toml","version = 1\n")
        delisted=(f"experiments/preregistration.toml: X900 was listed at {first[:12]} and no longer is; an experiment "
                  "stays listed once it is, so neither its runs nor its preregistration leave the gate")
        self.assert_blocked(root,delisted)
        self.assertEqual(mod.launch_errors(root,"X900"),[delisted])
        commit_all(root,"delisted")
        self.assert_blocked(root,delisted)
        self.assertEqual(mod.launch_errors(root,"X900"),[delisted])
        # Listed again, it passes; the newest commit that listed it is named.
        write(root,"experiments/preregistration.toml",listed)
        again=commit_all(root,"listed again")
        self.assertEqual(gate(root),(0,[]))
        write(root,"experiments/preregistration.toml",listed.replace("experiment.X900","experiment.X901"))
        self.assert_blocked(
            root,
            "experiments/preregistration.toml: X901 is not a registered experiment",
            delisted.replace(first[:12],again[:12]),
        )
        # Listed before the registry held it and registered in a later
        # commit, it is enrolled all the same.
        root=self.tree()
        registry=(root/"experiments/registry.toml").read_text(encoding="utf-8")
        write(root,"experiments/registry.toml","version = 1\n")
        early=commit_all(root,"listed first")
        write(root,"experiments/registry.toml",registry)
        commit_all(root,"registered")
        write(root,"experiments/preregistration.toml","version = 1\n")
        self.assert_blocked(root,delisted.replace(first[:12],early[:12]))
        self.assertEqual(mod.launch_errors(root,"X900"),[delisted.replace(first[:12],early[:12])])
        write(root,"experiments/preregistration.toml",listed)
        again=commit_all(root,"listed again")
        # Taken out of the registry and the list together, it is still
        # enrolled: the registry held it.
        root=self.tree()
        together=commit_all(root)
        write(root,"experiments/registry.toml","version = 1\n")
        write(root,"experiments/preregistration.toml","version = 1\n")
        self.assertEqual(mod.launch_errors(root,"X900"),[delisted.replace(first[:12],together[:12])])
        write(root,"experiments/registry.toml",registry)
        write(root,"experiments/preregistration.toml",listed)
        again=commit_all(root,"listed again")
        # A name the registry has never held, such as a mistyped one,
        # enrolls nothing: taken out again, it is not missed.
        write(root,"experiments/preregistration.toml",listed+'\n[experiment.X9O0.required]\nseeds = "int-list"\n')
        commit_all(root,"typo")
        self.assert_blocked(root,"experiments/preregistration.toml: X9O0 is not a registered experiment")
        write(root,"experiments/preregistration.toml",listed)
        commit_all(root,"typo fixed")
        self.assertEqual(gate(root),(0,[]))
        # An experiment the list has never named launches as before.
        root=self.tree(listed_text="version = 1\n")
        commit_all(root)
        self.assertEqual((gate(root),mod.launch_errors(root,"X900")),((0,[]),[]))

    def test_rules_changed_after_the_runs_fail(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        digest,rules=self.frozen(root)
        # A type changed in the list: the manifest's rules digest no longer
        # matches it, and the runs' commit lists other rules.
        listed=(root/"experiments/preregistration.toml").read_text(encoding="utf-8").replace('harness = "str"','harness = "file"')
        write(root,"experiments/preregistration.toml",listed)
        entry=mod.load(root/"experiments/preregistration.toml")["experiment"]["X900"]
        changed=mod.experiment_records.canonical_digest(entry)
        at=f"X900: {name} ran at {ran[:12]}"
        runs=[
            f"X900: {name} names preregistration_rules_sha256 {rules!r}, not {changed}, the digest the experiment is frozen at",
            f"{at}, whose experiments/preregistration.toml lists other rules for it than the frozen ones",
            f"{at}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900 was frozen at {ran[:12]}, whose experiments/preregistration.toml lists other rules for it than the frozen ones",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        ]
        self.assert_blocked(
            root,
            f"X900: preregistration_rules_sha256 {rules!r} is not {changed}, the digest of its entry in experiments/preregistration.toml",
            "X900: preregistration key harness names 'fixture', which is not a file in the repository",
            *runs,
        )
        # With the manifest rewritten to match, the runs still name the old
        # rules and ran where the list held them.
        manifest=(root/self.MANIFEST).read_text(encoding="utf-8").replace(rules,changed)
        write(root,self.MANIFEST,manifest)
        self.assert_blocked(
            root,
            "X900: preregistration key harness names 'fixture', which is not a file in the repository",
            *runs,
        )

    def test_the_runs_commit_must_hold_the_frozen_files_and_baseline_values(self):
        required={**REQUIRED,"protocol":"file"}
        path="experiments/semdb/X900-fixture/PROTOCOL.md"
        frozen_text="# Protocol\n"
        table={**TABLE,"protocol":path,"protocol_sha256":hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()}
        # The runs' commit held another protocol and other baseline values
        # than those the table freezes; the tree has since been made to match.
        root=self.tree(status="running",table=table,required=required,
                       baseline_config={**BASELINE_CONFIG,"model":{"revision":"4567def"}},freeze_baselines=False)
        manifest=(root/self.MANIFEST).read_text(encoding="utf-8")
        write(root,path,"# Another protocol\n")
        ran=commit_all(root)
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        write(root,path,frozen_text)
        write(root,BASELINE_PATH,(self.tree(table=table,required=required)/BASELINE_PATH).read_text(encoding="utf-8"))
        frozen_table={**table,"baseline_fixture_sha256":baseline_digest({"config.toml":(root/BASELINE_PATH).read_text(encoding="utf-8")})}
        fixed=self.tree(status="running",table=frozen_table,required=required)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(fixed/relative,root/relative)
        self.assertNotEqual(manifest,(root/self.MANIFEST).read_text(encoding="utf-8"))
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        short=ran[:12]
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(f"X900: {name} ran at {short}, where {path} is not the file frozen as protocol",lines)
        self.assertIn(f"X900: {name} ran at {short}, where baseline fixture is not the frozen one: its directory holds other files",lines)

    def link(self, target: Path, link: Path) -> None:
        """A symlink at `link` to `target`, or skip where none can be made."""
        link.parent.mkdir(parents=True,exist_ok=True)
        try:
            os.symlink(target,link)
        except (OSError,NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")

    def test_a_file_or_baseline_reached_through_a_symlink_is_refused(self):
        outside=Path(self.enterContext(tempfile.TemporaryDirectory()))
        text="# Protocol held elsewhere\n"
        (outside/"PROTOCOL.md").write_text(text,encoding="utf-8")
        path="experiments/semdb/X900-fixture/PROTOCOL.md"
        table={**TABLE,"protocol":path,"protocol_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()}
        root=self.tree(table=table,required={**REQUIRED,"protocol":"file"})
        self.link(outside/"PROTOCOL.md",root/path)
        refused=f"X900: preregistration key protocol names {path!r}, which is not a file in the repository"
        self.assert_blocked(root,refused)
        # A link inside the repository is refused too: git holds it as its
        # target's path, so the commit a run names could not show the content
        # frozen through it.
        (root/path).unlink()
        write(root,"experiments/semdb/X900-fixture/protocols/v1.md",text)
        self.link(root/"experiments/semdb/X900-fixture/protocols/v1.md",root/path)
        self.assert_blocked(root,refused)
        # So is a file reached through a linked directory.
        (root/path).unlink()
        self.link(root/"experiments/semdb/X900-fixture/protocols",root/"experiments/semdb/X900-fixture/linked")
        linked={**table,"protocol":"experiments/semdb/X900-fixture/linked/v1.md"}
        root2=self.tree(table=linked,required={**REQUIRED,"protocol":"file"})
        write(root2,"experiments/semdb/X900-fixture/protocols/v1.md",text)
        self.link(root2/"experiments/semdb/X900-fixture/protocols",root2/"experiments/semdb/X900-fixture/linked")
        self.assert_blocked(root2,"X900: preregistration key protocol names 'experiments/semdb/X900-fixture/linked/v1.md', which is not a file in the repository")
        # The same file, named directly, is a file in the repository.
        direct={**table,"protocol":"experiments/semdb/X900-fixture/protocols/v1.md"}
        root3=self.tree(table=direct,required={**REQUIRED,"protocol":"file"})
        write(root3,"experiments/semdb/X900-fixture/protocols/v1.md",text)
        self.assertEqual(gate(root3),(0,[]))
        # A baseline configuration linked in from elsewhere is refused too.
        root=self.tree()
        (outside/"baseline.toml").write_text((root/BASELINE_PATH).read_text(encoding="utf-8"),encoding="utf-8")
        (root/BASELINE_PATH).unlink()
        self.link(outside/"baseline.toml",root/BASELINE_PATH)
        self.assert_blocked(root,f"X900: baseline {BASELINE_PATH} is not a file in the repository")
        # So is any file of the baseline's directory reached through a link:
        # its implementation is frozen with it.
        root=self.tree()
        (outside/"runner.py").write_text("RANK = 1\n",encoding="utf-8")
        self.link(outside/"runner.py",root/"research/baselines/fixture/runner.py")
        self.assert_blocked(
            root,
            "X900: baseline research/baselines/fixture/ holds research/baselines/fixture/runner.py, "
            "which is not a regular file reached through no symlink",
        )

    def test_a_commit_freezes_only_what_it_holds_as_regular_files(self):
        # Git holds a symlink as its target's path, and the gate and the
        # runner refuse one, so a commit whose table froze a link's target
        # path could not launch the experiment: replacing the link with the
        # file it stood for repairs the experiment, it does not rewrite it.
        required={**REQUIRED,"protocol":"file"}
        protocol="experiments/semdb/X900-fixture/PROTOCOL.md"
        target="REAL.md"
        text="# Protocol\n"
        linked={**TABLE,"protocol":protocol,"protocol_sha256":hashlib.sha256(target.encode("utf-8")).hexdigest()}
        root=self.tree(status="running",table=linked,required=required)
        write(root,f"experiments/semdb/X900-fixture/{target}",text)
        self.link(Path(target),root/protocol)
        self.assert_blocked(root,f"X900: preregistration key protocol names {protocol!r}, which is not a file in the repository")
        commit_all(root)
        repaired=self.tree(status="running",table={**linked,"protocol_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()},
                           required=required)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(repaired/relative,root/relative)
        (root/protocol).unlink()
        write(root,protocol,text)
        commit_all(root,"a file, not a link")
        self.assertEqual((gate(root),mod.launch_errors(root,"X900")),((0,[]),[]))
        # Nor did one whose file key named a directory, which git lists file
        # by file: the key named no file.
        directory="experiments/semdb/X900-fixture/docs/"
        named={**linked,"protocol":directory,"protocol_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()}
        root=self.tree(status="running",table=named,required=required)
        write(root,f"{directory}only.md",text)
        self.assert_blocked(root,f"X900: preregistration key protocol names {directory!r}, which is not a file in the repository")
        commit_all(root)
        repaired=self.tree(status="running",table={**named,"protocol":f"{directory}only.md"},required=required)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(repaired/relative,root/relative)
        commit_all(root,"the file, not its directory")
        self.assertEqual(gate(root),(0,[]))
        # Nor did one whose baseline's directory held a link.
        helper="research/baselines/fixture/helper.py"
        target="../plain_model/config.toml"
        baseline_text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        root=self.tree(status="running",table={**TABLE,"baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text,"helper.py":target})})
        self.link(Path(target),root/helper)
        commit_all(root)
        (root/helper).unlink()
        code="def helper():\n    return 1\n"
        write(root,helper,code)
        repaired=self.tree(status="running",table={**TABLE,"baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text,"helper.py":code})})
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(repaired/relative,root/relative)
        commit_all(root,"a file, not a link")
        self.assertEqual(gate(root),(0,[]))

    def test_a_baseline_file_named_as_no_repository_path_froze_nothing(self):
        # The tree refuses a baseline directory holding such a name, so a
        # commit holding one could not launch, and renaming it repairs the
        # experiment rather than rewriting it.
        baseline_text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        code="x = 1\n"
        odd="research/baselines/fixture/helper[old].py"
        table={**TABLE,"baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text,"helper[old].py":code})}
        misnamed=("X900: baseline research/baselines/fixture/ holds research/baselines/fixture/helper[old].py, whose "
                  "name is not a repository path: printable ASCII, written as git writes a path, with no glob "
                  "character, leading `:` or step through .git")
        repaired=self.tree(status="running",table={**TABLE,"baseline_fixture_sha256":baseline_digest(
            {"config.toml":baseline_text,"helper_old.py":code})})
        # A table that names no baseline digest yet froze nothing either:
        # no digest is none a table could name.
        for table in (table,{key:value for key,value in TABLE.items() if key!="baseline_fixture_sha256"}):
            with self.subTest(named="baseline_fixture_sha256" in table):
                root=self.tree(status="running",table=table,freeze_baselines=False)
                write(root,odd,code)
                self.assert_blocked(root,misnamed)
                self.assertEqual(mod.launch_errors(root,"X900"),[misnamed])
                commit=commit_all(root)
                self.assertIsNone(mod.directory_digest_at(root,commit,"research/baselines/fixture"))
                self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
                (root/odd).unlink()
                write(root,"research/baselines/fixture/helper_old.py",code)
                for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
                    shutil.copyfile(repaired/relative,root/relative)
                commit_all(root,"renamed")
                self.assertEqual(gate(root),(0,[]))
        # Nor did a commit whose baseline held a link, the table naming none,
        # or naming the digest the link's target path would give as a file.
        linked=mod.file_table_digest({
            "config.toml":mod.file_entry("100644",mod.experiment_records.preregistered_bytes_digest(baseline_text.encode("utf-8"))),
            "alias.toml":mod.file_entry("120000",mod.experiment_records.preregistered_bytes_digest(b"config.toml")),
        })
        for table in ({key:value for key,value in TABLE.items() if key!="baseline_fixture_sha256"},
                      {**TABLE,"baseline_fixture_sha256":linked}):
            with self.subTest(named="baseline_fixture_sha256" in table):
                root=self.tree(status="running",table=table,freeze_baselines=False)
                self.link(Path("config.toml"),root/"research/baselines/fixture/alias.toml")
                commit=commit_all(root)
                self.assertIsNone(mod.directory_digest_at(root,commit,"research/baselines/fixture"))
                self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        # A record run there is not of the frozen baseline either, the table
        # naming none: no digest is none a table could name.
        root=self.tree(status="running",table={key:value for key,value in TABLE.items() if key!="baseline_fixture_sha256"},
                       freeze_baselines=False)
        self.link(Path("config.toml"),root/"research/baselines/fixture/alias.toml")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"record")
        self.assert_blocked(
            root,
            "X900: baseline research/baselines/fixture/ holds research/baselines/fixture/alias.toml, which is not a "
            "regular file reached through no symlink",
            f"X900: {self.RECORD.removeprefix('experiments/semdb/X900-fixture/')} ran at {ran[:12]}, where baseline "
            "fixture is not the frozen one: its directory holds other files",
        )

    def test_a_listed_experiments_entrypoint_takes_only_preregistered_values(self):
        # The runner fills each placeholder but <seed> from the frozen table;
        # one it cannot fill, a program that is no name, or a character no
        # process can be given would make a command no run can use, so the
        # tree check refuses it.
        for entrypoint,refusal in (
            ("bench <seed> <iterations>",
             "X900: entrypoint placeholder <iterations> is no key of the [preregistration] table, so the runner has no "
             "frozen value for it"),
            ("bench <seed> --programs=<programs>",
             "X900: entrypoint placeholder <programs> is preregistered as a list, which no command token takes"),
            ('bench <seed> "unterminated',"X900: entrypoint cannot be split into a command: No closing quotation"),
            ('"" <seed>',"X900: entrypoint names no program: its first token is empty"),
            ("' ' <seed>","X900: entrypoint names no program: its first token is empty"),
            ("bench\0 <seed>","X900: entrypoint holds a NUL character, which no command can be given"),
            # Two seeds, one command: each record would name a seed it did not
            # run, as `--seed 17` hardcoded runs seed 17 for `run --seed 29`.
            ("bench --seed 17",
             "X900: entrypoint takes no <seed> placeholder, so each of its 2 preregistered seeds would run one command "
             "while its record named another seed"),
            # Cargo configuration on the command line names what the commit
            # does not hold, in a file or a value.
            *((entrypoint,CARGO_CONFIGURATION) for entrypoint in (
                "cargo run --config /tmp/adapted.toml -- <seed>",
                "/home/runner/.cargo/bin/cargo run --config=build.rustc-wrapper='\"/tmp/w\"' -- <seed>",
                "rustup -v run --install stable cargo test --config c.toml -- <seed>",
                "rustup +nightly run stable cargo run --config c.toml -- <seed>",
                # As Windows spells the programs, in any case and with .exe.
                "rustup.exe run nightly cargo +nightly build --config=C:/mutable.toml -- <seed>",
                "'C:\\Rust\\bin\\CARGO.EXE' run --config c.toml -- <seed>",
                "RUSTUP run stable Cargo.Exe bench --config c.toml -- <seed>",
            )),
            # Cargo runs one of its built-in commands first, after at most a
            # +toolchain: an alias, an option before the command or an
            # external command could have it read the rest otherwise than
            # as written, or run a program no record names.
            *((entrypoint,CARGO_FIRST.format(given)) for entrypoint,given in (
                ("cargo --config /tmp/adapted.toml run -- <seed>","'--config'"),
                ("cargo -Zunstable-options -C /elsewhere run -- <seed>","'-Zunstable-options'"),
                ("cargo -vC../elsewhere run -- <seed>","'-vC../elsewhere'"),
                ("cargo -- b -- <seed>","'--'"),
                ("cargo +nightly -Zscript /tmp/run.rs <seed>","'-Zscript'"),
                ("cargo +nightly -Z=script ../run.rs <seed>","'-Z=script'"),
                ("cargo xtest -- <seed>","'xtest'"),
                ("cargo r -- <seed>","'r'"),
                ("cargo +1.85.0 +foo -- <seed>","'+foo'"),
                ("rustup run stable cargo clippy -- <seed>","'clippy'"),
            )),
            # Nor a directory to run in, whose configuration no check reads.
            ("cargo run -C=../elsewhere -- <seed>",CARGO_DIRECTORY),
            ("cargo +nightly run -vCcrates/x -- <seed>",CARGO_DIRECTORY),
            # A manifest, lockfile or directory outside the repository holds
            # sources no watch or record binds, and a target directory given
            # to Cargo could hold a build made outside the commit.
            *((entrypoint,CARGO_OUTSIDE.format(option,value)) for entrypoint,option,value in (
                ("cargo run --manifest-path /elsewhere/Cargo.toml -- <seed>","--manifest-path","/elsewhere/Cargo.toml"),
                ("rustup run stable cargo run --manifest-path=crates/../../x/Cargo.toml -- <seed>","--manifest-path",
                 "crates/../../x/Cargo.toml"),
                ("cargo run --lockfile-path C:/x/Cargo.lock -- <seed>","--lockfile-path","C:/x/Cargo.lock"),
                ("cargo run --manifest-path './../x/Cargo.toml' -- <seed>","--manifest-path","./../x/Cargo.toml"),
                ("cargo run --manifest-path '..\\elsewhere\\Cargo.toml' -- <seed>","--manifest-path","..\\elsewhere\\Cargo.toml"),
                # Git keeps its own directory, which no scan of the tree lists.
                ("cargo run --manifest-path .git/evil/Cargo.toml -- <seed>","--manifest-path",".git/evil/Cargo.toml"),
                ("cargo run --manifest-path crates/.GIT/x/Cargo.toml -- <seed>","--manifest-path","crates/.GIT/x/Cargo.toml"),
                # However Windows names it: with trailing dots or spaces, or
                # by its short name.
                ("cargo run --manifest-path .Git./evil/Cargo.toml -- <seed>","--manifest-path",".Git./evil/Cargo.toml"),
                ("cargo run --lockfile-path 'crates/git~1/Cargo.lock' -- <seed>","--lockfile-path","crates/git~1/Cargo.lock"),
                ("cargo run --manifest-path '.git::$INDEX_ALLOCATION/x/Cargo.toml' -- <seed>","--manifest-path",
                 ".git::$INDEX_ALLOCATION/x/Cargo.toml"),
                # Or a step Windows reads as `.` or `..` though it is not written
                # as one: trailing dots or spaces, dots and spaces only, or a
                # stream.
                ("cargo run --manifest-path '.. /outside/Cargo.toml' -- <seed>","--manifest-path",
                 ".. /outside/Cargo.toml"),
                ("cargo run --manifest-path crates/.../x/Cargo.toml -- <seed>","--manifest-path",
                 "crates/.../x/Cargo.toml"),
                ("cargo run --lockfile-path 'crates/. /Cargo.lock' -- <seed>","--lockfile-path","crates/. /Cargo.lock"),
                ("cargo run --manifest-path 'crates/..:stream/Cargo.toml' -- <seed>","--manifest-path",
                 "crates/..:stream/Cargo.toml"),
                ("cargo run --manifest-path 'crates/ /Cargo.toml' -- <seed>","--manifest-path","crates/ /Cargo.toml"),
                # A name Windows trims to another (`crates.` is `crates`), so
                # the path leaves the tree the watch reads.
                ("cargo run --manifest-path 'crates./Cargo.toml' -- <seed>","--manifest-path","crates./Cargo.toml"),
                ("cargo run --manifest-path crates/a../Cargo.toml -- <seed>","--manifest-path","crates/a../Cargo.toml"),
                # rustup takes the options of `run` on either side of the
                # toolchain, and runs Cargo all the same.
                ("rustup run stable --install cargo run --manifest-path /tmp/external/Cargo.toml -- <seed>","--manifest-path",
                 "/tmp/external/Cargo.toml"),
                ("rustup run --install stable cargo run --manifest-path /tmp/external/Cargo.toml -- <seed>","--manifest-path",
                 "/tmp/external/Cargo.toml"),
                ("rustup run --install stable --install cargo run --lockfile-path ../x/Cargo.lock -- <seed>","--lockfile-path",
                 "../x/Cargo.lock"),
                ("rustup run stable -- cargo run --manifest-path /tmp/external/Cargo.toml -- <seed>","--manifest-path",
                 "/tmp/external/Cargo.toml"),
                # A colon in a component is an NTFS stream of the name before
                # it on Windows, a file no scan of the tree lists: a target
                # specification, a manifest or a lockfile could hide there.
                ("cargo build --target targets/base:evil.json -- <seed>","--target","targets/base:evil.json"),
                ("cargo run --manifest-path=crates/x:y/Cargo.toml -- <seed>","--manifest-path","crates/x:y/Cargo.toml"),
                ("cargo run --lockfile-path 'Cargo.lock:stream' -- <seed>","--lockfile-path","Cargo.lock:stream"),
                # A target may be a specification file.
                ("cargo build --target /tmp/custom.json -- <seed>","--target","/tmp/custom.json"),
            )),
            ("cargo run --target-dir target-old -- <seed>",
             "X900: entrypoint gives Cargo a target directory (--target-dir), which could hold a build made outside "
             "the commit; the runner builds a listed run into a fresh one"),
        ):
            with self.subTest(entrypoint=entrypoint):
                root=self.tree(status="running")
                self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n',f"entrypoint = {json.dumps(entrypoint)}\n")
                self.assert_blocked(root,refusal)
                self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        # An integer, a string and a boolean each fill one; a planned
        # experiment is not gated.
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "bench <seed> <schema> <harness> <see_intent>"\n')
        self.assertEqual(gate(root),(0,[]))
        root=self.tree(status="planned")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "bench <seed> <iterations>"\n')
        self.assertEqual(gate(root),(0,[]))
        # A string value holding a NUL character fills no token either.
        self.assertEqual(
            mod.command_errors("X900",{"entrypoint":"bench <seed> <harness>"},{"harness":"a\0b"}),
            ["X900: entrypoint placeholder <harness> is preregistered holding a NUL character, which no command can be "
             "given"],
        )
        self.assertEqual(mod.command_errors("X900",{"entrypoint":"bench <seed> <harness>"},{"harness":"ab"}),[])
        # One seed, named once or more, needs no <seed>; one given inside a
        # token counts.
        for seeds in ([17],[17,17],[]):
            self.assertEqual(mod.command_errors("X900",{"entrypoint":"bench --seed 17"},{"seeds":seeds}),[])
        self.assertEqual(mod.command_errors("X900",{"entrypoint":"bench --seed=<seed>"},{"seeds":[17,29]}),[])
        # So is one a preregistered value fills in, as the runner fills it:
        # the option, the program, or the whole token.
        for entrypoint,table in (
            ("cargo run <option> <configuration> -- <seed>",{"option":"--config","configuration":"/tmp/adapted.toml"}),
            ("<tool> run --config c.toml -- <seed>",{"tool":"cargo"}),
            ("cargo run <option> -- <seed>",{"option":"--config=build.jobs=1"}),
        ):
            with self.subTest(entrypoint=entrypoint):
                self.assertEqual(
                    mod.command_errors("X900",{"entrypoint":entrypoint},{**table,"seeds":[17,29]}),[CARGO_CONFIGURATION])
        self.assertEqual(
            mod.command_errors("X900",{"entrypoint":"cargo run <option> -- <seed>"},{"option":"--release","seeds":[17,29]}),[])
        # A placeholder may name Cargo's command too, and is read as filled.
        self.assertEqual(
            mod.command_errors("X900",{"entrypoint":"cargo <command> -- <seed>"},{"command":"xtest","seeds":[17,29]}),
            [CARGO_FIRST.format("'xtest'")])
        # Past `--` an argument is the built program's, and another program's
        # --config is its own.
        for entrypoint in ("cargo run --manifest-path crates/bench/Cargo.toml -- <seed>",
                           "cargo run --manifest-path=./crates/../Cargo.toml -- <seed>",
                           "cargo run -FCuda -pbench -- --manifest-path /elsewhere <seed>",
                           "cargo run -FCrate/../../feature -- <seed>",
                           "cargo run --manifest-path crates/.github/Cargo.toml -- <seed>",
                           "cargo run --manifest-path crates/..x/Cargo.toml -- <seed>",
                           "cargo run --manifest-path crates/...x/Cargo.toml -- <seed>",
                           "cargo run --manifest-path crates/a.b/../Cargo.toml -- <seed>",
                           "cargo run --manifest-path crates/.hidden/Cargo.toml -- <seed>",
                           "cargo +nightly run -Zunstable-options -- /tmp <seed>",
                           "cargo build --target x86_64-unknown-linux-gnu --target=targets/custom.json -- <seed>",
                           "cargo run -- --config c.toml <seed>","python3 bench.py --config c.toml <seed>",
                           "rustup run stable python3 bench.py --config c.toml <seed>","rustup which cargo --config <seed>","rustup show stable cargo --config c.toml <seed>"):
            with self.subTest(entrypoint=entrypoint):
                self.assertEqual(mod.command_errors("X900",{"entrypoint":entrypoint},{"seeds":[17,29]}),[])
        # A commit holding such an entrypoint could not launch it and froze
        # nothing, so preregistering the value repairs it, and so does
        # naming a program.
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "\\"\\" <seed>"\n')
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        self.edit(root,self.MANIFEST,'entrypoint = "\\"\\" <seed>"\n','entrypoint = "bench <seed>"\n')
        named=commit_all(root,"program named")
        self.assertEqual(gate(root),(0,[]))
        self.assertIsNotNone(mod.launchable_at(root,named,"X900","experiments/semdb/X900-fixture"))
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "bench <seed> <iterations>"\n')
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        repaired=self.tree(status="running",table={**TABLE,"iterations":30})
        self.edit(repaired,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "bench <seed> <iterations>"\n')
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(repaired/relative,root/relative)
        repair=commit_all(root,"iterations preregistered")
        self.assertEqual(gate(root),(0,[]))
        self.assertIsNotNone(mod.launchable_at(root,repair,"X900","experiments/semdb/X900-fixture"))

    def test_a_cargo_alias_stands_in_for_no_command_a_listed_run_gives_cargo(self):
        # Cargo lets no alias shadow a built-in command, and a listed command
        # runs one of those first: what follows is read as written, whatever
        # aliases the repository's configuration defines, and an alias of
        # another name is refused rather than followed.
        files={
            ".cargo/config.toml":(
                '[alias]\nb = "run --manifest-path /tmp/x/Cargo.toml"\nrun = "run --"\n'
                + "".join(f'a{index} = "a{index+1}"\n' for index in range(40))
                + 'a40 = ["run", "--manifest-path", "/tmp/x/Cargo.toml"]\n'
            ),
        }
        seeds={"seeds":[17,29]}

        def errors(entrypoint,read=files.get):
            return mod.command_errors("X900",{"entrypoint":entrypoint},seeds,read)

        outside=CARGO_OUTSIDE.format("--manifest-path","/tmp/x/Cargo.toml")
        # `run = "run --"` would hide the rest from a check that expanded it;
        # Cargo ignores it, and so does the gate.
        self.assertEqual(errors("cargo run --manifest-path /tmp/x/Cargo.toml -- <seed>"),[outside])
        self.assertEqual(errors("cargo run --release -- <seed>"),[])
        for entrypoint,given in (("cargo b -- <seed>","'b'"),("cargo a0 -- <seed>","'a0'"),("cargo -- b -- <seed>","'--'")):
            with self.subTest(entrypoint=entrypoint):
                self.assertEqual(errors(entrypoint),[CARGO_FIRST.format(given)])
        # The launch reads the configuration in the tree, and the history the
        # configuration at each commit.
        root=self.tree(status="running")
        (root/".cargo").mkdir(exist_ok=True)
        (root/".cargo/config.toml").write_text(files[".cargo/config.toml"],encoding="utf-8")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "cargo b -- <seed>"\n')
        self.assert_blocked(root,CARGO_FIRST.format("'b'"))
        self.assertEqual(mod.launch_errors(root,"X900"),[CARGO_FIRST.format("'b'")])
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        # A NUL character in the command is refused as such, whatever Cargo
        # would make of it, at a commit too.
        self.edit(root,self.MANIFEST,'entrypoint = "cargo b -- <seed>"\n',
                  'entrypoint = "cargo run -C a\\u0000b -- <seed>"\n')
        nul=[CARGO_DIRECTORY,"X900: entrypoint holds a NUL character, which no command can be given"]
        self.assert_blocked(root,*nul)
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))

    def test_a_commit_that_changes_only_cargo_metadata_can_be_the_freeze(self):
        # The launch check reads the repository's Cargo configuration and
        # every Cargo.toml, so a commit that repairs one of them can be the
        # first the runner could launch the experiment at, whatever else it
        # holds: a run made there, seen and discarded, must not go unseen
        # when the preregistration is rewritten after it.
        directory="experiments/semdb/X900-fixture"
        def outside(name):
            """A manifest at `name` naming a path above the repository's root."""
            climb="../"*len(name.split("/"))
            return f'[package]\nname = "x"\n[dependencies]\nexternal = {{ path = "{climb}external" }}\n'

        inside='[package]\nname = "x"\n[dependencies]\n'
        wrapper='[build]\nrustc-wrapper = "/usr/bin/sccache"\n'
        allowed='[alias]\nok = "run --release"\n'
        for name,blocked,repaired in (
            (".cargo/config.toml",wrapper,allowed),
            (".cargo/config",wrapper,allowed),
            ("Cargo.toml",None,inside),
            ("crates/inner/Cargo.toml",None,inside),
            ("crates/deep/er/Cargo.toml",None,inside),
        ):
            with self.subTest(name=name):
                root=self.tree(status="running")
                self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "cargo run -- <seed>"\n')
                write(root,name,outside(name) if blocked is None else blocked)
                first=commit_all(root)
                self.assertIsNone(mod.launchable_at(root,first,"X900",directory))
                self.assertEqual([commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[])
                write(root,name,repaired)
                freeze=commit_all(root,"cargo metadata repaired")
                self.assertIsNotNone(mod.launchable_at(root,freeze,"X900",directory))
                # The commit changes no file of the experiment, the list or
                # the registry, and is still listed as the freeze.
                self.assertEqual(git(root,"show","--format=","--name-only",freeze).split(),[name])
                self.assertEqual(
                    [commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[freeze])
                self.assertEqual(gate(root),(0,[]))
                # The preregistration rewritten afterwards is a rewrite of what
                # the runner could have launched at the freeze.
                rewritten=self.tree(status="running",table={**TABLE,"schema":2})
                for relative in (f"{directory}/config.toml",self.MANIFEST):
                    shutil.copyfile(rewritten/relative,root/relative)
                self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "cargo run -- <seed>"\n')
                commit_all(root,"rewrite")
                code,lines=gate(root)
                self.assertEqual(code,1)
                for line in (
                    f"X900 was frozen at {freeze[:12]}, whose config.toml holds another [preregistration] than the frozen one",
                    f"X900 was frozen at {freeze[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
                ):
                    self.assertIn(line,lines)
        # A repair of a file no Cargo reads is not one that freezes: only
        # the names the check reads are listed.
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'entrypoint = "bench <seed>"\n','entrypoint = "cargo run -- <seed>"\n')
        write(root,".cargo/config.toml",wrapper)
        commit_all(root)
        write(root,"notes/Cargo.toml.txt","x\n")
        write(root,".cargo/other.toml","x\n")
        commit_all(root,"unrelated")
        self.assertEqual([commit for commit,held in mod.launch_listing(root,"X900",[directory]) if held is not None],[])

    def test_a_history_names_its_paths_literally_unless_it_asks_for_magic(self):
        # A path is a path, whatever characters it holds, so the commits of
        # `notes/a[1].txt` are not those of `notes/a1.txt`; only a caller that
        # names magic gets it.
        root=self.tree(status="running")
        write(root,"notes/a1.txt","one\n")
        first=commit_all(root)
        write(root,"notes/a[1].txt","bracket\n")
        second=commit_all(root,"bracket")
        write(root,"notes/b.txt","b\n")
        third=commit_all(root,"other")
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--","notes/a[1].txt"),[second])
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--","notes/a[1].txt",literal=True),[second])
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--","notes/a[1].txt",literal=False),[second,first])
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--",":(glob)notes/*.txt",literal=False),[third,second,first])
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--",":(literal)notes/a[1].txt",literal=False),[second])
        self.assertEqual(mod.history(root,"--format=%H","HEAD","--",":(glob)notes/*.txt"),[])

    def test_identical_history_queries_are_cached_within_a_gate_run(self):
        root=self.tree(status="running")
        commit_all(root)
        mod._history_cached.cache_clear()
        asked=[]
        original=mod.experiment_records.git
        def counting(where,*arguments,**options):
            asked.append(arguments)
            return original(where,*arguments,**options)
        with mock.patch.object(mod.experiment_records,"git",side_effect=counting):
            first=mod.history(root,"--format=%H","HEAD","--","experiments/registry.toml")
            second=mod.history(root,"--format=%H","HEAD","--","experiments/registry.toml")
        self.assertEqual(first,second)
        self.assertEqual([arguments for arguments in asked if "log" in arguments],[
            ("--literal-pathspecs","log","--full-history","--format=%H","HEAD","--","experiments/registry.toml")
        ])

    def test_the_launch_listing_names_the_experiments_directory_literally(self):
        # A registry path with a wildcard in it is that name, which no commit
        # holds, and not the file or directory it would match.
        root=self.tree(status="running")
        first=commit_all(root)
        write(root,"experiments/semdb/X900-fixture/results/notes.txt","x\n")
        commit_all(root,"a file of the directory")
        for named in ("experiments/semdb/X900-fixture/results/n?tes.txt","experiments/semdb/X9?0-fixture/results",
                      "experiments/semdb/X900-fixture/results/n[o]tes.txt"):
            with self.subTest(named=named):
                self.assertEqual([commit for commit,_ in mod.launch_listing(root,"X900",[named])],[first])
        self.assertEqual(len(mod.launch_listing(root,"X900",["experiments/semdb/X900-fixture"])),2)

    def test_a_listed_runs_cargo_configuration_names_no_program_source_or_flags(self):
        # Cargo runs the compiler, wrapper, linker or runner its configuration
        # names, and builds from the sources and with the flags it names:
        # none of them is what the record binds. A listed run's Cargo
        # configuration, which Cargo started from the root reads there, sets
        # only tables that name none of them.
        def errors(entrypoint,files):
            return mod.command_errors("X900",{"entrypoint":entrypoint},{"seeds":[17]},files.get)

        def refused(path,keys):
            return (f"X900: {path} sets {keys} for Cargo, which can name a program Cargo runs (a compiler, a "
                    "wrapper, a linker or a runner), a source it builds from or flags it builds with outside what a "
                    "record binds; a listed experiment's Cargo configuration sets only [alias], [cargo-new], "
                    "[future-incompat-report], [http], [net], [term]")

        allowed=('[alias]\nok = "run --release"\n[term]\nverbose = false\n'
                 '[net]\noffline = true\n[http]\ntimeout = 30\n[cargo-new]\nvcs = "none"\n'
                 '[future-incompat-report]\nfrequency = "never"\n')
        self.assertEqual(errors("cargo run -- <seed>",{".cargo/config.toml":allowed}),[])
        for text,keys in (
            ('[build]\nrustc = "/tmp/fake-rustc"\n',"build"),
            ('[build]\nrustc-wrapper = "/usr/bin/sccache"\n',"build"),
            ('[build]\nrustflags = ["--sysroot=/tmp/std"]\n',"build"),
            ("[target.x86_64-unknown-linux-gnu]\nrunner = \"/tmp/fake\"\n","target"),
            ("[target.'cfg(unix)']\nlinker = \"/tmp/ld\"\n","target"),
            ('[host]\nlinker = "/tmp/ld"\n',"host"),
            ('include = ["/tmp/other.toml"]\n',"include"),
            ('paths = ["/tmp/crate"]\n[source.local]\ndirectory = "/tmp/vendor"\n',"paths, source"),
            ('[patch.crates-io]\nserde = { path = "/tmp/serde" }\n[unstable]\nbuild-std = ["std"]\n',"patch, unstable"),
            ('[profile.release]\ncodegen-backend = "/tmp/backend.so"\n',"profile"),
            # Cargo sets its [env] for what it runs: a library preloaded
            # into the compiler and build scripts, outside the environment
            # the runner pins.
            ('[env]\nLD_PRELOAD = "/tmp/adapt.so"\n',"env"),
        ):
            with self.subTest(configuration=text):
                self.assertEqual(errors("cargo run -- <seed>",{".cargo/config.toml":text}),
                                 [refused(".cargo/config.toml",keys)])
                # Also the older name, which Cargo prefers where both are,
                # and for a command that is no Cargo but may start one.
                self.assertEqual(errors("python3 bench.py <seed>",{".cargo/config":text,".cargo/config.toml":allowed}),
                                 [refused(".cargo/config",keys)])
        runner="[target.x86_64-unknown-linux-gnu]\nrunner = \"/tmp/fake\"\n"
        # A directory Cargo is not started in is not read (a listed command
        # gives Cargo no -C).
        nested={"crates/x/.cargo/config.toml":runner,"crates/.cargo/config":runner,".cargo/config.toml":allowed}
        self.assertEqual(errors("cargo run -- <seed>",nested),[])
        # A file Cargo reads but Python's TOML reader does not, one opening
        # with a byte-order mark, say, is refused as unchecked.
        unparsed=errors("cargo run -- <seed>",{".cargo/config.toml":"\ufeff"+runner})
        self.assertEqual(len(unparsed),1,unparsed)
        self.assertTrue(unparsed[0].startswith("X900: .cargo/config.toml does not parse as TOML ("),unparsed)
        # The launch reads the configuration in the tree, and the history the
        # configuration at each commit.
        root=self.tree(status="running")
        (root/".cargo").mkdir(exist_ok=True)
        (root/".cargo/config.toml").write_text(runner,encoding="utf-8")
        self.assert_blocked(root,refused(".cargo/config.toml","target"))
        self.assertEqual(mod.launch_errors(root,"X900"),[refused(".cargo/config.toml","target")])
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))

    def test_a_cargo_manifest_names_no_path_outside_the_repository(self):
        # Cargo follows a path dependency wherever it points, and what lies
        # outside the repository is bound by no watch or record: the sources
        # could change between seeds while every record named one commit.
        def errors(files,entrypoint="cargo run -- <seed>"):
            return mod.command_errors("X900",{"entrypoint":entrypoint},{"seeds":[17]},files.get,lambda:list(files))

        def refused(name,path):
            return (f"X900: {name} names {path}, outside what the repository's watch reads, whose sources no watch or "
                    "record binds; name a path the repository holds")

        from_root=('[package]\nname = "x"\nbuild = "build.rs"\n[lib]\npath = "src/lib.rs"\n'
                   '[[bin]]\nname = "b"\npath = "src/main.rs"\n[dependencies]\n'
                   'core = { path = "crates/core" }\nlocal = "1.0"\n[workspace]\nmembers = ["crates/*"]\n')
        from_crate=('[package]\nname = "x"\nworkspace = "../.."\n[dependencies]\n'
                    'sibling = { path = "../y" }\nroot = { path = "../.." }\n')
        # Each spelling that names a path, from the root and from a crate.
        for text,path in (
            ('[dependencies]\nexternal = { path = "../external" }\n',"../external"),
            ('[dependencies.external]\npath = "../../external"\n',"../../external"),
            ('[dev-dependencies]\nexternal = { path = "../external" }\n',"../external"),
            ('[build-dependencies]\nexternal = { path = "../external" }\n',"../external"),
            ("[target.'cfg(unix)'.dependencies]\nexternal = { path = \"../external\" }\n","../external"),
            ("[target.x86_64-unknown-linux-gnu.build-dependencies]\nexternal = { path = \"../e\" }\n","../e"),
            ('[workspace.dependencies]\nexternal = { path = "../external" }\n',"../external"),
            ('[patch.crates-io]\nserde = { path = "../serde" }\n',"../serde"),
            ('[replace]\n"serde:1.0.0" = { path = "../serde" }\n',"../serde"),
            ('[workspace]\nmembers = ["../elsewhere"]\n',"../elsewhere"),
            ('[workspace]\ndefault-members = ["../elsewhere"]\n',"../elsewhere"),
            ('[workspace]\nexclude = ["../elsewhere"]\n',"../elsewhere"),
            ('[package]\nname = "x"\nworkspace = "../outer"\n',"../outer"),
            ('[package]\nname = "x"\nbuild = "../build.rs"\n',"../build.rs"),
            ('[lib]\npath = "../lib.rs"\n',"../lib.rs"),
            ('[[bin]]\nname = "b"\npath = "../main.rs"\n',"../main.rs"),
            ('[[example]]\nname = "e"\npath = "../e.rs"\n',"../e.rs"),
            ('[[test]]\nname = "t"\npath = "../t.rs"\n',"../t.rs"),
            ('[[bench]]\nname = "b"\npath = "../b.rs"\n',"../b.rs"),
            ('[dependencies]\nexternal = { path = "/opt/external" }\n',"/opt/external"),
            ('[dependencies]\nexternal = { path = "C:/external" }\n',"C:/external"),
            ('[dependencies]\nexternal = { path = "..\\\\external" }\n',"..\\external"),
            ('[dependencies]\nexternal = { path = "crates/.git/x" }\n',"crates/.git/x"),
            ('[dependencies]\nexternal = { path = "crates/x:stream" }\n',"crates/x:stream"),
        ):
            with self.subTest(root_manifest=text):
                self.assertEqual(errors({"Cargo.toml":text}),[refused("Cargo.toml",path)])
        # From a crate, a path climbs to the root and no further.
        self.assertEqual(errors({"crates/x/Cargo.toml":'[dependencies]\nsibling = { path = "../y" }\n'}),[])
        self.assertEqual(errors({"crates/x/Cargo.toml":'[dependencies]\nroot = { path = "../.." }\n'}),[])
        self.assertEqual(errors({"crates/x/Cargo.toml":'[dependencies]\nout = { path = "../../.." }\n'}),
                         [refused("crates/x/Cargo.toml","../../..")])
        self.assertEqual(errors({"crates/x/Cargo.toml":'[dependencies]\nout = { path = "../../../external" }\n'}),
                         [refused("crates/x/Cargo.toml","../../../external")])
        self.assertEqual(errors({"a/Cargo.toml":'[dependencies]\nout = { path = "../../external" }\n'}),
                         [refused("a/Cargo.toml","../../external")])
        # The older spellings of the dependency tables, which Cargo reads.
        for table in ("dev_dependencies","build_dependencies"):
            with self.subTest(table=table):
                self.assertEqual(errors({"Cargo.toml":f'[{table}]\nexternal = {{ path = "../external" }}\n'}),
                                 [refused("Cargo.toml","../external")])
        # An absolute path, on either platform's terms, is outside from any
        # directory, not a name below it (a TOML literal string holds the
        # backslashes as they are).
        for absolute in ("/opt/external","C:/external",r"\\host\share\x"):
            with self.subTest(absolute=absolute):
                self.assertEqual(errors({"crates/x/Cargo.toml":f"[dependencies]\nx = {{ path = '{absolute}' }}\n"}),
                                 [refused("crates/x/Cargo.toml",absolute)])
        # What is not a string, or a table of the shape Cargo reads, is
        # skipped rather than failing: Cargo refuses it itself.
        self.assertEqual(errors({"Cargo.toml":'[dependencies]\nx = { path = 5 }\ny = "1"\n[workspace]\nmembers = "a"\n'
                                              '[lib]\npath = 5\n[[bin]]\nname = "b"\n[package]\nbuild = 1\n'
                                              '[target]\nx = 1\n[patch]\ncrates-io = 1\n'}),[])
        # What names paths inside the repository, and other tables, pass.
        self.assertEqual(errors({"Cargo.toml":from_root,"crates/x/Cargo.toml":from_crate}),[])
        self.assertEqual(errors({"Cargo.toml":'[package]\nname = "x"\nbuild = false\n[dependencies]\nserde = "1"\n'
                                              'git = { git = "https://example.invalid/x.git" }\n'
                                              '[package.metadata.docs]\npath = "../not-cargo"\n'}),[])
        # Only files named Cargo.toml, at any depth, are manifests; every
        # one of them is read, each error in the order of the names.
        self.assertEqual(errors({"Cargo.toml.bak":'[dependencies]\nx = { path = "../x" }\n',
                                 "notes/Cargo.lock":"x = 1\n"}),[])
        self.assertEqual(errors({"b/Cargo.toml":'[lib]\npath = "../../l.rs"\n',
                                 "a/Cargo.toml":'[lib]\npath = "../../l.rs"\n'}),
                         [refused("a/Cargo.toml","../../l.rs"),refused("b/Cargo.toml","../../l.rs")])
        # A manifest that Python's TOML reader cannot parse is refused as
        # unchecked, and a name git lists that holds no readable file is not.
        unparsed=errors({"Cargo.toml":"[dependencies\n"})
        self.assertEqual(len(unparsed),1,unparsed)
        self.assertTrue(unparsed[0].startswith("X900: Cargo.toml does not parse as TOML ("),unparsed)
        self.assertEqual(mod.cargo_manifest_errors("X900",["Cargo.toml","crates/x/Cargo.toml"],lambda name:None),[])
        # A command that is no Cargo may start one, so the repository's
        # manifests count for every listed experiment; a caller that gives no
        # listing of names checks none, as before.
        self.assertEqual(errors({"Cargo.toml":'[dependencies]\nx = { path = "../x" }\n'},"python3 bench.py <seed>"),
                         [refused("Cargo.toml","../x")])
        self.assertEqual(mod.command_errors("X900",{"entrypoint":"cargo run -- <seed>"},{"seeds":[17]},
                                            {"Cargo.toml":'[dependencies]\nx = { path = "../x" }\n'}.get),[])
        # The launch reads the manifests in the tree, and the history the
        # manifests at each commit.
        root=self.tree(status="running")
        (root/"crates/x").mkdir(parents=True,exist_ok=True)
        (root/"crates/x/Cargo.toml").write_text('[dependencies]\nexternal = { path = "../../../external" }\n',encoding="utf-8")
        message=refused("crates/x/Cargo.toml","../../../external")
        self.assert_blocked(root,message)
        self.assertEqual(mod.launch_errors(root,"X900"),[message])
        commit=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,commit,"X900","experiments/semdb/X900-fixture"))
        # Committed, the manifest is still one the tree holds.
        self.assertEqual(mod.launch_errors(root,"X900"),[message])
        self.assertEqual(sorted(name for name in mod.commit_names(root,commit) if name.endswith("Cargo.toml")),
                         ["crates/x/Cargo.toml"])
        self.assertEqual(sorted(name for name in mod.tree_names(root) if name.endswith("Cargo.toml")),
                         ["crates/x/Cargo.toml"])
        with self.assertRaises(mod.HistoryUnreadable):
            mod.commit_names(root,"0"*40)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(mod.HistoryUnreadable):
            mod.tree_names(Path(directory))
        # Put right, the tree passes and the commit could launch.
        (root/"crates/x/Cargo.toml").write_text('[dependencies]\nsibling = { path = "../y" }\n',encoding="utf-8")
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        self.assertIsNotNone(mod.launchable_at(root,commit_all(root,"inside"),"X900","experiments/semdb/X900-fixture"))

    def test_a_name_that_is_not_utf8_is_no_repository_path(self):
        # Git holds a name's bytes as they are. One that is not UTF-8 is
        # refused in the tree, froze nothing at a commit, and fails no later
        # gate once it is gone.
        baseline_text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        root=self.tree(status="running",table={**TABLE,"baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text})})
        odd=os.path.join(os.fsencode(root/"research/baselines/fixture"),b"helper\xff.py")
        notes=os.path.join(os.fsencode(root/"experiments/semdb/X900-fixture"),b"notes\xff.md")
        try:
            for name in (odd,notes):
                with open(name,"wb") as handle:
                    handle.write(b"x = 1\n")
        except OSError as error:
            self.skipTest(f"cannot name a file with bytes that are not UTF-8: {error}")
        refusal="X900: baseline research/baselines/fixture/ holds a file whose name is not UTF-8, which no repository path is"
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        commit=commit_all(root)
        self.assertIsNone(mod.directory_digest_at(root,commit,"research/baselines/fixture"))
        for name in (odd,notes):
            os.unlink(name)
        commit_all(root,"removed")
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        # A record so named is refused, and the gate prints its name's bytes
        # as escapes rather than failing to print it.
        root=self.tree(status="running")
        (root/"experiments/semdb/X900-fixture/results").mkdir(exist_ok=True)
        with open(os.path.join(os.fsencode(root/"experiments/semdb/X900-fixture/results"),b"run-\xff.json"),"wb") as handle:
            handle.write(b"{}")
        commit_all(root)
        self.assert_blocked(root,"X900: results/run-\\udcff.json is not a regular file reached through no symlink")

    def test_an_executable_file_is_frozen_with_its_mode(self):
        # Git holds an executable file as mode 100755. A baseline's directory
        # freezes each file's mode with its content, so one there freezes as
        # it is; a preregistered file is frozen by its content alone, so it
        # is no executable, whose bit would change what a command does.
        protocol="experiments/semdb/X900-fixture/run.sh"
        script="#!/bin/sh\nexit 0\n"
        helper="research/baselines/fixture/helper.sh"
        baseline_text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        table={**TABLE,"protocol":protocol,"protocol_sha256":hashlib.sha256(script.encode("utf-8")).hexdigest(),
               "baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text,"helper.sh":script},executable={"helper.sh"})}
        root=self.tree(status="running",table=table,required={**REQUIRED,"protocol":"file"})
        write(root,protocol,script)
        write(root,helper,script)
        (root/helper).chmod(0o755)
        head=commit_all(root)
        self.assertEqual(git(root,"ls-tree","HEAD","--",helper).split()[0],"100755")
        self.assertEqual((gate(root),mod.launch_commit_errors(root,"X900",head)),((0,[]),[]))
        # Made executable, the preregistered file is refused, as git holds its
        # mode in the index and at a commit, which then froze nothing: its bit
        # put back repairs it.
        (root/protocol).chmod(0o755)
        git(root,"add","--",protocol)
        refusal=(f"X900: preregistration key protocol names {protocol!r}, which is executable; a preregistered file is "
                 "frozen by its content, so it is a file no command runs as a program")
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        executable=commit_all(root,"protocol executable")
        self.assertEqual(git(root,"ls-tree","HEAD","--",protocol).split()[0],"100755")
        self.assertIsNone(mod.frozen_file_at(root,executable,protocol))
        self.assertIsNone(mod.launchable_at(root,executable,"X900","experiments/semdb/X900-fixture"))
        (root/protocol).chmod(0o644)
        commit_all(root,"protocol not executable")
        self.assertEqual(gate(root),(0,[]))
        # Git's mode counts: a bit the checkout sets but git does not hold is
        # a change the runner refuses as uncommitted, not the file's mode.
        (root/protocol).chmod(0o755)
        self.assertEqual(gate(root),(0,[]))
        (root/protocol).chmod(0o644)
        # A record run where the file was executable did not run the frozen
        # file, whatever its content.
        write(root,self.RECORD,json.dumps(self.record(root,executable)))
        commit_all(root,"a record of the executable commit")
        self.assert_blocked(
            root,
            f"X900: {self.RECORD.removeprefix('experiments/semdb/X900-fixture/')} ran at {executable[:12]}, where "
            f"{protocol} is not the file frozen as protocol",
        )

    def test_a_listed_experiments_manifest_and_configuration_are_regular_files(self):
        # Read through a link, either is a file no commit holds as the
        # experiment's, and the runner could not launch from it.
        for name in ("config.toml","experiment.toml"):
            with self.subTest(name=name):
                root=self.tree(status="running")
                held=root/"experiments/semdb/X900-fixture"/name
                shutil.move(held,root/"experiments/semdb"/f"elsewhere-{name}")
                self.link(Path(f"../elsewhere-{name}"),held)
                refusal=f"X900: {name} is not a regular file reached through no symlink"
                self.assert_blocked(root,refusal)
                self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        # A link to nothing is a link the tree holds, not a missing file.
        root=self.tree(status="running")
        held=root/"experiments/semdb/X900-fixture/config.toml"
        held.unlink()
        self.link(Path("../nowhere.toml"),held)
        refusal="X900: config.toml is not a regular file reached through no symlink"
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])

    def test_history_is_read_as_committed_whatever_replaces_it(self):
        # A replacement object (git replace) would show the gate another
        # content than the commit holds.
        root=self.tree()
        write(root,"notes.md","committed\n")
        commit=commit_all(root)
        write(root,"other.md","replaced\n")
        git(root,"replace",git(root,"rev-parse",f"{commit}:notes.md"),git(root,"hash-object","-w","other.md"))
        self.assertEqual(git(root,"show",f"{commit}:notes.md"),"replaced")
        self.assertEqual(mod.blob(root,commit,"notes.md"),b"committed\n")
        # A commit's full name holds what it holds for good, so its reads are
        # kept; HEAD is read afresh once it moves.
        self.assertEqual(mod.blob(root,"HEAD","notes.md"),b"committed\n")
        write(root,"notes.md","later\n")
        commit_all(root,"later")
        self.assertEqual((mod.blob(root,"HEAD","notes.md"),mod.blob(root,commit,"notes.md")),(b"later\n",b"committed\n"))
        # Nor does a replaced tree.
        root=self.tree()
        write(root,"notes.md","committed\n")
        commit=commit_all(root)
        write(root,"notes.md","replaced\n")
        other=commit_all(root,"other")
        git(root,"replace",git(root,"rev-parse",f"{commit}^{{tree}}"),git(root,"rev-parse",f"{other}^{{tree}}"))
        self.assertEqual(git(root,"show",f"{commit}:notes.md"),"replaced")
        self.assertEqual(mod.blob(root,commit,"notes.md"),b"committed\n")

    def test_git_keeps_its_own_directory_so_no_path_through_it_is_a_repository_path(self):
        # Git refuses `.git` in any case in a path a commit holds, and, as
        # Windows reads it, with trailing dots or spaces or as `git~1`.
        for refused in (".git",".git/config",".GIT/config","a/.git/b",".Git./config",".git /config","git~1/config","a/GIT~1"):
            with self.subTest(refused=refused):
                self.assertFalse(mod.is_repository_path(refused))
        for accepted in (".gitignore",".github/workflows/ci.yml","a/.gitkeep","x.git/y","git/config","a/git~2",".gitx/y"):
            with self.subTest(accepted=accepted):
                self.assertTrue(mod.is_repository_path(accepted))

    def test_a_cargo_input_path_with_a_colon_in_a_component_names_a_file_outside_the_scan(self):
        # On Windows a colon after a name starts an NTFS stream of it, which
        # git and a walk of the tree do not list, so what a stream holds is
        # bound by no watch or record; every platform refuses the same paths.
        for outside in ("targets/base:evil.json","base:evil.json","a/b:c/d","crates/x:y/Cargo.toml","Cargo.lock:s","./a:b",
                        "a/b:","a\\b:c","x/C:y","x/1:y","ab:c","::x"):
            with self.subTest(outside=outside):
                self.assertTrue(mod.outside_repository(outside))
        # The names without one are as before.
        for inside in ("targets/base.json","crates/x/Cargo.toml","Cargo.lock","a/b c/d","./a/b","crates/..x/y","x.y/z"):
            with self.subTest(inside=inside):
                self.assertFalse(mod.outside_repository(inside))

    def test_a_name_windows_reads_as_a_step_is_no_part_of_a_repository_path(self):
        # Windows trims trailing dots and spaces, and an NTFS stream, from a
        # name, so `.. ` climbs out of its directory as `..` does, `...` and
        # ` ` name the directory itself, and `..:x` is `..` too.
        for refused in ("../x",".. /x","a/.. /b","a/.../b","a/. /b","a/..:x/b","a/ /b","a/. ./b","...","a/.. ","a/.:x/b"):
            with self.subTest(refused=refused):
                self.assertFalse(mod.is_repository_path(refused))
        # A drive letter and a colon at the start names a path on that drive
        # on Windows, whatever follows it, whatever the letter's case.
        for refused in ("C:x","C:/x","c:/x","z:","Z:y/z","C:.."):
            with self.subTest(refused=refused):
                self.assertFalse(mod.is_repository_path(refused))
        # A name with anything else in it is a name of its own on every
        # platform, and a colon after the first component is a stream's name
        # inside the directory it names, one of the files in the repository.
        for accepted in ("...x/y","a b/c","..x/y",".hidden/y","a.b/c","x/C:y","ab:c/d","1:x/y"):
            with self.subTest(accepted=accepted):
                self.assertTrue(mod.is_repository_path(accepted))
        # A name with trailing dots or spaces is another name on Windows
        # (`archive.` is `archive`), so the history would hold what the
        # runner wrote under a spelling this path does not have.
        for trimmed in ("a../b",".a./b","a/.x./b","a/x ../y","archive.","results/archive ","x./y"):
            with self.subTest(trimmed=trimmed):
                self.assertFalse(mod.is_repository_path(trimmed))
        # The gate's two predicates and the runner's read the same names as a
        # drive, so none of them can accept what another refuses.
        for drive in ("C:x","C:/x","c:/x","z:","Z:y/z","C:..","a:b","C:\\x"):
            with self.subTest(drive=drive):
                self.assertTrue(mod.names_a_drive(drive))
                self.assertTrue(mod.outside_repository(drive))
        for no_drive in ("","x","/x","x/C:y","ab:c","1:x","::x",":x","C","results/a:b"," C:x"):
            with self.subTest(no_drive=no_drive):
                self.assertFalse(mod.names_a_drive(no_drive))
        for stepping in (".. ","...","."," ",". ",".. ..","..:x",":x"):
            with self.subTest(stepping=stepping):
                self.assertEqual(mod.is_windows_dot_name(stepping),stepping not in (".",".."))
        for name in ("","..","...x","a b","..x",".git","a.b","a:b"):
            with self.subTest(name=name):
                self.assertFalse(mod.is_windows_dot_name(name))
        # Trailing dots or spaces, before any stream, change the spelling.
        for name in ("a..",".a.","archive.","archive ","a. :x","x.:y"):
            with self.subTest(name=name):
                self.assertTrue(mod.is_windows_dot_name(name))

    def test_a_history_git_cannot_read_fails_the_gate_rather_than_reading_as_empty(self):
        # Read as empty, a history would hide every freeze and record in it.
        root=self.tree(status="running")
        first=commit_all(root)
        write(root,"notes.md","later\n")
        commit_all(root,"later")
        (root/".git/objects"/first[:2]/first[2:]).unlink()
        code,lines=gate(root)
        self.assertEqual((code,len(lines)),(1,1),lines)
        self.assertTrue(lines[0].startswith(f"HEAD's history cannot be read in {root}: "),lines)
        refused=mod.launch_errors(root,"X900")
        self.assertEqual(len(refused),1,refused)
        self.assertTrue(refused[0].startswith(f"HEAD's history cannot be read in {root}: "),refused)
        # Nor does one whose HEAD commit git cannot read.
        root=self.tree(status="running")
        head=commit_all(root)
        (root/".git/objects"/head[:2]/head[2:]).unlink()
        code,lines=gate(root)
        self.assertEqual((code,len(lines)),(1,1),lines)
        self.assertTrue(lines[0].startswith(f"HEAD's history cannot be read in {root}: "),lines)
        # Without a commit at HEAD there is no history to hide.
        self.assertEqual(mod.history(self.tree(),"--format=%H","HEAD"),[])
        # A launch commit whose tree, a file of it or a baseline's subtree git
        # cannot read refuses the launch, named, rather than reading as
        # holding nothing.
        baseline_text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        for lost in ("tree","config","baseline subtree"):
            with self.subTest(lost=lost):
                table={**TABLE,"baseline_fixture_sha256":baseline_digest({"config.toml":baseline_text,"lib/x.py":"x = 1\n"})}
                root=self.tree(status="running",table=table)
                write(root,"research/baselines/fixture/lib/x.py","x = 1\n")
                head=commit_all(root)
                self.assertEqual(mod.launch_commit_errors(root,"X900",head),[])
                spelled={"tree":f"{head}^{{tree}}","config":f"{head}:experiments/semdb/X900-fixture/config.toml",
                         "baseline subtree":f"{head}:research/baselines/fixture/lib"}[lost]
                name=git(root,"rev-parse",spelled)
                mod.listed_entry.cache_clear()
                mod.object_bytes.cache_clear()
                (root/".git/objects"/name[:2]/name[2:]).unlink()
                refused=mod.launch_commit_errors(root,"X900",head)
                self.assertEqual(len(refused),1,refused)
                self.assertTrue(refused[0].startswith(f"HEAD's history cannot be read in {root}: "),refused)

    def test_v4_decisions_bind_only_the_trees_that_hold_the_experiments(self):
        # A tree without M001-v4 and M002-v4, such as a fixture, has no v4
        # decision to bind: it is no error that their files are absent.
        root=self.tree()
        self.assertEqual(gate(root),(0,[]))
        # The check itself is unchanged where it runs: a tree asked about one
        # of them without its decision file still reports it.
        for exp_id in mod.V4_NO_GO:
            with self.subTest(experiment=exp_id):
                self.assertEqual(mod.v4_no_go_errors(exp_id,root),[f"{exp_id}: DECISION.toml is not a regular repository file"])

    def test_a_listed_experiment_is_reached_through_no_symlink(self):
        # Git holds a link as its target's path, so the history of the
        # directory the registry names would hold none of its files.
        root=self.tree(status="running")
        shutil.move(root/"experiments/semdb/X900-fixture",root/"experiments/real-X900")
        self.link(Path("../real-X900"),root/"experiments/semdb/X900-fixture")
        refusal=("X900: experiments/registry.toml places it in experiments/semdb/X900-fixture, which is reached through a "
                 "symlink, so its runs and freezes there could not be found")
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])

    def test_a_repository_path_is_written_as_git_writes_it(self):
        # git lists a directory's files for `dir/` and names no entry
        # `./x`, so the tree and a commit read one entry only by the path
        # git writes.
        for refused in ("a/","./a","a//b","a/./b","a/b/","a/b/."):
            with self.subTest(refused=refused):
                self.assertFalse(mod.is_repository_path(refused))
        for accepted in ("a","a/b","a.b/c.d"):
            with self.subTest(accepted=accepted):
                self.assertTrue(mod.is_repository_path(accepted))
        # No checkout holds a longer path than PATH_MAX, and git could not be
        # handed one as an argument.
        self.assertTrue(mod.is_repository_path("a"*4096))
        self.assertFalse(mod.is_repository_path("a"*4097))

    def test_a_listed_experiment_lives_where_the_gate_can_search_its_history(self):
        # The gate searches the directories the registry has given an
        # experiment only where they are repository paths, so a listed one
        # placed elsewhere is refused: a run there could not be found.
        root=self.tree(status="running")
        shutil.move(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900[a]")
        write(root,"experiments/registry.toml",'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/X900[a]"\nstatus = "running"\n')
        refusal=("X900: experiments/registry.toml places it in 'experiments/semdb/X900[a]', which is not a repository "
                 "path, so its runs and freezes there could not be found")
        self.assert_blocked(root,refusal)
        self.assertEqual(mod.launch_errors(root,"X900"),[refusal])
        head=commit_all(root)
        self.assertIsNone(mod.launchable_at(root,head,"X900","experiments/semdb/X900[a]"))
        # Nor one whose registry path climbs out and back in, which the tree
        # reads as the directory and git names as no path at all.
        root=self.tree(status="running")
        write(root,"experiments/registry.toml",'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/../semdb/X900-fixture"\nstatus = "running"\n')
        climbing=("X900: experiments/registry.toml places it in 'experiments/semdb/../semdb/X900-fixture', which is not a "
                  "repository path, so its runs and freezes there could not be found")
        self.assert_blocked(root,climbing)
        self.assertEqual(mod.launch_errors(root,"X900"),[climbing])
        # A registry path with a trailing slash names the directory git
        # names without it, and a freeze there is found after a move.
        root=self.tree(status="running")
        write(root,"experiments/registry.toml",'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/X900-fixture/"\nstatus = "running"\n')
        self.assertEqual((gate(root),mod.launch_errors(root,"X900")),((0,[]),[]))
        frozen=commit_all(root)
        rewritten=self.tree(status="running",table={**TABLE,"schema":2})
        shutil.move(root/"experiments/semdb/X900-fixture",root/"experiments/semdb/X900-moved")
        for name in ("config.toml","experiment.toml"):
            shutil.copyfile(rewritten/"experiments/semdb/X900-fixture"/name,root/"experiments/semdb/X900-moved"/name)
        write(root,"experiments/registry.toml",'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/X900-moved"\nstatus = "running"\n')
        commit_all(root,"moved and rewritten")
        self.assert_blocked(
            root,
            f"X900 was frozen at {frozen[:12]}, where the registry placed it in experiments/semdb/X900-fixture, not in experiments/semdb/X900-moved",
            f"X900 was frozen at {frozen[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {frozen[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        )
        # A registry path that leaves the repository at some commit does not
        # take the directories it gave before out of the search, as a
        # pathspec outside the repository would take all of them.
        root=self.tree(status="running")
        frozen=commit_all(root)
        registry=(root/"experiments/registry.toml").read_text(encoding="utf-8")
        write(root,"experiments/registry.toml",registry.replace('path = "semdb/X900-fixture"','path = "../../outside"'))
        commit_all(root,"placed outside")
        write(root,"experiments/registry.toml",registry)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        commit_all(root,"placed back and rewritten")
        self.assert_blocked(
            root,
            f"X900 was frozen at {frozen[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {frozen[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        )

    def test_a_launch_names_a_commit_that_holds_the_experiment_frozen(self):
        # The gate finds a freeze by what a commit holds, so a run may name
        # only a commit that holds the experiment frozen as the tree
        # launches it.
        def refusal(commit: str) -> str:
            return (f"X900: {commit[:12]}, the commit its run would name, does not hold it frozen as the tree launches it; "
                    "every file that decides its launch must be a regular file that commit holds, not a file git ignores "
                    "or keeps in its own directory, nor a symlink")

        root=self.tree(status="running")
        head=commit_all(root)
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[])
        # A protocol that HEAD's .gitignore hides is frozen in the tree, and
        # the runner's watch sees no change to it, but HEAD does not hold it.
        protocol="experiments/semdb/X900-fixture/notes/protocol.md"
        text="# Protocol\n"
        table={**TABLE,"protocol":protocol,"protocol_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()}
        root=self.tree(status="running",table=table,required={**REQUIRED,"protocol":"file"})
        write(root,".gitignore","notes/\n")
        write(root,protocol,text)
        head=commit_all(root)
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[refusal(head)])
        # The list and the registry are read as the commit holds them, so a
        # list emptied in the tree after the runner looked changes nothing.
        write(root,"experiments/preregistration.toml","version = 1\n")
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[refusal(head)])
        # Nor may a commit whose registry does not place it launch it.
        root=self.tree(status="running")
        write(root,"experiments/registry.toml","version = 1\n")
        head=commit_all(root)
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[refusal(head)])
        # An experiment the list does not name runs as before.
        root=self.tree(status="running",listed_text="version = 1\n")
        head=commit_all(root)
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[])
        # A list the commit does not hold readable refuses the launch.
        write(root,"experiments/preregistration.toml","version = 1\n[experiment\n")
        head=commit_all(root,"unreadable list")
        self.assertEqual(mod.launch_commit_errors(root,"X900",head),[
            f"experiments/preregistration.toml cannot be read as {head[:12]} holds it, so whether X900 preregisters is unknown",
        ])

    def test_a_missing_list_fails_the_gate_and_refuses_every_launch(self):
        root=self.tree()
        (root/"experiments/preregistration.toml").unlink()
        self.assert_blocked(root,"experiments/preregistration.toml does not exist")
        self.assertEqual(
            mod.launch_errors(root,"X900"),
            ["experiments/preregistration.toml does not exist, so whether X900 preregisters is unknown"],
        )

    def test_a_listed_experiment_launches_only_with_a_frozen_preregistration(self):
        self.assertEqual(mod.launch_errors(self.tree(),"X900"),[])
        self.assertEqual(
            mod.launch_errors(self.tree(status="planned"),"X900"),
            ["X900 preregisters (experiments/preregistration.toml) and is 'planned': it runs only once its "
             "preregistration is frozen and it is prepared, running, completed or failed"],
        )
        self.assertEqual(
            mod.launch_errors(self.tree(table={**TABLE,"harness":"must-be-pinned-x"}),"X900"),
            ["X900: preregistration key harness is a placeholder ('must-be-pinned-x')"],
        )
        self.assertEqual(
            mod.launch_errors(self.tree(required={**REQUIRED,"seeds":"list"}),"X900"),
            ["experiments/preregistration.toml: X900: key seeds has unknown type 'list'",
             "experiments/preregistration.toml: X900 must require seeds as an int-list: every listed experiment preregisters the seeds it runs"],
        )
        # An experiment the list does not name launches as before.
        root=self.tree(listed_text="version = 1\n")
        self.assertEqual(mod.launch_errors(root,"X900"),[])
        # The runner holds what decides a launch to HEAD: the list, the
        # registry, the gate, and the baseline's whole directory.
        self.assertEqual(
            mod.launch_inputs(self.tree(),"X900"),
            ["experiments/preregistration.toml","experiments/registry.toml","scripts/check_research_gates.py","research/baselines/fixture"],
        )

    def test_the_matched_baseline_rules_read_the_tree_they_are_given(self):
        # M001-M005 and E002 keep their rules, on the tree the gate is given:
        # the repository's own baselines are unpinned, so only a fixture whose
        # baselines are pinned tells the trees apart.
        def tree(plain: str, strong: str) -> Path:
            root=Path(self.enterContext(tempfile.TemporaryDirectory()))
            registry="version = 1\n"
            for exp_id,path in (("M001","model/M001-fixture"),("E002","system/E002-fixture")):
                registry+=f'\n[[experiment]]\nid = "{exp_id}"\npath = "{path}"\nstatus = "running"\n'
                write(root,f"experiments/{path}/experiment.toml",f'version = 1\nid = "{exp_id}"\nstatus = "running"\n')
            write(root,"experiments/registry.toml",registry)
            write(root,"experiments/preregistration.toml","version = 1\n")
            write(root,"research/baselines/plain_model/config.toml",plain)
            write(root,"research/baselines/strong_rag/config.toml",strong)
            return root

        unpinned=tree(PLAIN_MODEL,'status = "blocked-unpinned"\n[dense]\nrevision = ""\n')
        self.assert_blocked(
            unpinned,
            "M001: matched plain-model baseline is not pinned",
            "E002: strong RAG model/generator revisions are not pinned",
            "E002: strong RAG baseline is still blocked",
        )
        pinned=tree(
            'version = 1\n[model]\nbackend = "b"\nmodel = "m"\n',
            'status = "pinned"\n[dense]\nrevision = "r1"\n[reranker]\nrevision = "r2"\n[answer]\nrevision = "r3"\n',
        )
        self.assertEqual(gate(pinned),(0,[]))

ENROLLED=("S003","F003","Q003","R004","M008","E005")
# The list also holds the experiments that ran or were superseded; they are no
# part of the six this fixture moves to `prepared`.
RETIRED=("M001-v2","M001-v4","M002-v2","M002-v3","M002-v4","M002-v5","M002-v6","M002-v7","M002-v8","M002-v9")

class EnrolledExperimentTests(unittest.TestCase):
    """The repository's own list and configurations, copied into a fixture
    tree with every listed experiment moved to `prepared`."""

    @staticmethod
    def list_of(ids) -> str:
        """The repository's list with only the entries of `ids`: its header and
        each `[experiment.<id>...]` table whose id is one of them."""
        kept=[]
        keep=True
        for line in (ROOT/"experiments/preregistration.toml").read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith("[experiment."):
                keep=line.removeprefix("[experiment.").split("]",1)[0].split(".",1)[0] in ids
            if keep:
                kept.append(line)
        return "".join(kept)

    def prepared_tree(self, edit=None) -> Path:
        """A fixture tree holding the repository's list, the six listed
        experiments at status `prepared` and their baselines, changed by
        `edit(root, paths)` when given."""
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        paths={item["id"]:item["path"] for item in mod.load(ROOT/"experiments/registry.toml")["experiment"]}
        registry="version = 1\n"
        for exp_id in ENROLLED:
            registry+=f'\n[[experiment]]\nid = "{exp_id}"\npath = "{paths[exp_id]}"\nstatus = "prepared"\n'
            for name in ("experiment.toml","config.toml"):
                source=ROOT/"experiments"/paths[exp_id]/name
                # Leaving planned, the owner names what each one runs.
                text=source.read_text(encoding="utf-8").replace('status = "planned"','status = "prepared"').replace('entrypoint = ""','entrypoint = "bench <seed>"')
                write(root,f"experiments/{paths[exp_id]}/{name}",text)
        write(root,"experiments/registry.toml",registry)
        write(root,"experiments/preregistration.toml",self.list_of(ENROLLED))
        for relative in (
            "research/baselines/plain_model/config.toml",
            "research/baselines/rag_reference/config.toml",
            "research/baselines/strong_rag/config.toml",
        ):
            (root/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,root/relative)
        if edit:
            edit(root,paths)
        git(root,"init","-q")
        return root

    def test_the_list_enrolls_the_experiments_that_preregister(self):
        listed=mod.load(ROOT/"experiments/preregistration.toml")["experiment"]
        self.assertEqual(sorted(listed),sorted((*ENROLLED,*RETIRED)))
        # S003 lists every key its design preregisters (doc 35 design, section
        # 3.10), with the type of the design's value, low_cells included, in
        # the design's order.
        design={
            "schema":"int","harness":"str","seeds":"int-list","pilot_seeds":"int-list",
            "cases_per_seed":"int","tasks_per_case":"int","agents":"int-list","groups_ladder":"int-list",
            "groups_ladder_fallback":"int-list","items_per_group":"int","insert_cap":"int","counters":"int",
            "counter_start":"int","sets":"int","set_members":"int","policies":"int",
            "tick_us":"int","think_min_ticks":"int","think_max_ticks":"int","step_ticks":"int",
            "merge_ticks":"int","max_attempts":"int","review_min_ticks":"int","review_max_ticks":"int",
            "programs":"str-list","program_weights_permille":"int-list","rmw_delta_max":"int","counter_add_max":"int",
            "guard_amount":"int","rely_permille":"int","ingress_permille_per_tick":"int","external_write_permille_per_tick":"int",
            "lifecycle_permille_per_tick":"int","rewire_permille_per_tick":"int","rewire_swap_min_ticks":"int","rewire_swap_max_ticks":"int",
            "required_verification":"str","verifiers":"str-list","host_counter_jump_max":"int","auto_policy":"str",
            "auto_threshold_permille":"int","auto_calibration_permille":"int","review_policy":"str","review_threshold_permille":"int",
            "review_calibration_permille":"int","calibration_seed":"int","reviewer":"str","conflict_threshold_permille":"int",
            "efficiency_floor_permille":"int","bootstrap_resamples":"int","bootstrap_seed":"int","bootstrap_interval_permille":"int",
            "min_hazard_trials_per_class_per_seed":"int","min_cells_per_side":"int","merge_wall_budget_us_p99":"int","durable_roundtrip_every_cases":"int",
            "nondeterminism_rerun_case":"int","low_cells":"str-list",
        }
        self.assertEqual(len(design),58)
        # Every listed experiment preregisters the seeds its manifest declares.
        for exp_id,entry in listed.items():
            self.assertEqual(entry["required"].get("seeds"),"int-list",exp_id)
        self.assertEqual(list(listed["S003"]["required"].items()),list(design.items()))

    def test_the_enrolled_experiments_are_held_until_their_placeholders_are_pinned(self):
        code,lines=gate(self.prepared_tree())
        self.assertEqual(code,1)
        strong="research/baselines/strong_rag/config.toml"
        strong_errors=lambda exp_id:[
            f"{exp_id}: baseline {strong}: dense.revision is a placeholder ('')",
            f"{exp_id}: baseline {strong}: reranker.revision is a placeholder ('')",
            f"{exp_id}: baseline {strong}: answer.generator is a placeholder ('')",
            f"{exp_id}: baseline {strong}: answer.revision is a placeholder ('')",
            f"{exp_id}: baseline {strong} is blocked-unpinned-model-revisions",
        ]
        pinned="'must-be-pinned-before-prepared'"
        signed="'must-be-signed-by-owner'"
        expected=[
            "S003: config.toml has no [preregistration] table",
            f"F003: preregistration key uncalibrated_threshold_permille is a placeholder ({pinned})",
            f"F003: preregistration key calibration_set_rule is a placeholder ({signed})",
            f"F003: preregistration key adjudicators_see_intent is a placeholder ({signed})",
            f"F003: preregistration key harm_breakdown_namespaces is a placeholder ({signed})",
            f"F003: preregistration key adjudication_protocol is a placeholder ({signed})",
            f"F003: preregistration key adjudication_protocol_sha256 is a placeholder ({pinned})",
            "F003: experiment.toml names no preregistration_sha256",
            "F003: experiment.toml names no preregistration_rules_sha256",
            f"Q003: preregistration key recall_margin_permille is a placeholder ({pinned})",
            f"Q003: preregistration key embedding_model is a placeholder ({pinned})",
            f"Q003: preregistration key embedding_revision is a placeholder ({pinned})",
            f"Q003: preregistration key baseline_rag_reference_sha256 is a placeholder ({pinned})",
            f"Q003: preregistration key baseline_strong_rag_sha256 is a placeholder ({pinned})",
            "Q003: experiment.toml names no preregistration_sha256",
            "Q003: experiment.toml names no preregistration_rules_sha256",
            "Q003: baseline research/baselines/rag_reference/config.toml: hybrid.embedding_model is a placeholder ('must-be-pinned-before-reported-run')",
            *strong_errors("Q003"),
            f"R004: preregistration key public_regression_gate_permille is a placeholder ({pinned})",
            f"R004: preregistration key backward_transfer_min_permille is a placeholder ({pinned})",
            "R004: experiment.toml names no preregistration_sha256",
            "R004: experiment.toml names no preregistration_rules_sha256",
            f"M008: preregistration key token_budget is a placeholder ({pinned})",
            f"M008: preregistration key recency_window is a placeholder ({pinned})",
            f"M008: preregistration key recency_buffer_baseline is a placeholder ({pinned})",
            f"M008: preregistration key hybrid_retrieval_baseline is a placeholder ({pinned})",
            "M008: experiment.toml names no preregistration_sha256",
            "M008: experiment.toml names no preregistration_rules_sha256",
            f"E005: preregistration key token_budget is a placeholder ({pinned})",
            f"E005: preregistration key full_context_baseline is a placeholder ({pinned})",
            f"E005: preregistration key agent_memory_baselines has element 0 that is a placeholder ({pinned})",
            f"E005: preregistration key baseline_strong_rag_sha256 is a placeholder ({pinned})",
            "E005: experiment.toml names no preregistration_sha256",
            "E005: experiment.toml names no preregistration_rules_sha256",
            *strong_errors("E005"),
        ]
        self.assertEqual(sorted(lines),sorted(expected))

    def test_f003_leaves_planned_once_its_owner_decisions_are_signed_and_its_digest_named(self):
        signed={
            "uncalibrated_threshold_permille":500,
            "calibration_set_rule":"every adjudication recorded before the policy version; held-out = every one after",
            "adjudicators_see_intent":False,
            "harm_breakdown_namespaces":"first ':'-separated segment of each changed key, descriptive only",
            "adjudication_protocol":"experiments/feedback/F003-calibrated-arbiter/PROTOCOL.md",
        }

        def sign(root: Path, paths: dict) -> None:
            directory=root/"experiments"/paths["F003"]
            protocol=root/signed["adjudication_protocol"]
            protocol.write_text("# F003 adjudication protocol\n",encoding="utf-8")
            config=mod.load(directory/"config.toml")
            table={
                **config.pop("preregistration"),**signed,
                "adjudication_protocol_sha256":mod.experiment_records.preregistered_file_digest(protocol),
            }
            header="".join(f"{key} = {toml_value(value)}\n" for key,value in config.items())
            (directory/"config.toml").write_text(header+"\n"+toml_table("preregistration",table),encoding="utf-8")
            digest=mod.experiment_records.preregistration_digest(table)
            rules=mod.experiment_records.canonical_digest(mod.load(root/"experiments/preregistration.toml")["experiment"]["F003"])
            manifest=(directory/"experiment.toml").read_text(encoding="utf-8")
            (directory/"experiment.toml").write_text(
                manifest+f'preregistration_sha256 = "{digest}"\npreregistration_rules_sha256 = "{rules}"\n',encoding="utf-8")

        _,lines=gate(self.prepared_tree(sign))
        self.assertEqual([line for line in lines if line.startswith("F003")],[])
        # Its decided values are the ones the design fixed.
        table=mod.load(ROOT/"experiments/feedback/F003-calibrated-arbiter/config.toml")["preregistration"]
        self.assertEqual(
            {key:table[key] for key in ("alpha_permille","delta_permille","threshold_rule","threshold_grid_steps","min_adjudications_per_policy")},
            {"alpha_permille":50,"delta_permille":50,"threshold_rule":"learn_then_test","threshold_grid_steps":1000,"min_adjudications_per_policy":59},
        )
        # 59 is the fewest adjudications, all without harm, whose one-sided
        # Clopper-Pearson bound at confidence 1 - delta is at most alpha.
        self.assertTrue(0.95**59<=0.05<0.95**58)

if __name__=="__main__":
    unittest.main()

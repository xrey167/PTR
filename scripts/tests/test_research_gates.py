import contextlib
import importlib.util
import io
import json
import shutil
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

    def test_stale_results_of_a_completed_experiment_fail_the_gate(self):
        completed=[
            item["id"]
            for item in mod.load(ROOT/"experiments/registry.toml").get("experiment",[])
            if mod.load(ROOT/"experiments"/item["path"]/"experiment.toml").get("status")=="completed"
        ]
        self.assertTrue(completed)
        checked=[]

        def stale(exp_id,experiment,results,root):
            checked.append(exp_id)
            self.assertEqual((root,results.parent),(mod.ROOT,experiment))
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

def toml_table(name: str, table: dict) -> str:
    return f"[{name}]\n"+"".join(f"{key} = {toml_value(value)}\n" for key,value in table.items())

def write(root: Path, relative: str, text: str) -> None:
    path=root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding="utf-8")

PLAIN_MODEL='version = 1\n[model]\nbackend = "must-be-pinned-before-run"\nmodel = "must-be-pinned-before-run"\n'
CONFIG_HEADER='version = 1\nkind = "workspace-area"\nname = "X900-fixture"\npath = "experiments/semdb/X900-fixture"\ntests_dir = "tests"\n'
TABLE={"schema":1,"harness":"fixture","seeds":[17,29],"programs":["rmw","set_op"],"see_intent":False}
REQUIRED={"schema":"int","harness":"str","seeds":"int-list","programs":"str-list","see_intent":"bool"}
BASELINE_PATH="research/baselines/fixture/config.toml"
BASELINE={"path":BASELINE_PATH,"keys":["model.revision","answer.generators"],"status_key":"status"}
BASELINE_CONFIG={"status":"reference-implemented","model":{"revision":"0123abc"},"answer":{"generators":["g1","g2"]}}
DIGEST=object()

def gate(root: Path) -> tuple[int, list[str]]:
    """Run the gate on the tree at `root`, with the checks of completed
    experiments' archived results (git history the fixtures do not have)
    passing once they are asked about that tree, and return its exit code
    and error lines."""
    def archived(exp_id,experiment,results,given_root):
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

class PreregistrationGateTests(unittest.TestCase):
    """The preregistration gate on fixture trees holding one listed
    experiment, X900, and one baseline."""

    def tree(
        self,
        status="prepared",
        table=TABLE,
        required=REQUIRED,
        digest=DIGEST,
        seeds=(17,29),
        baselines=(BASELINE,),
        baseline_config=BASELINE_CONFIG,
        entry_extra=None,
        listed_text=None,
    ) -> Path:
        """A fixture tree; `digest` is the manifest's preregistration_sha256,
        the table's own digest unless given (None leaves it out), and
        `listed_text`, when given, the whole of the list."""
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        write(root,"experiments/registry.toml",f'version = 1\n\n[[experiment]]\nid = "X900"\npath = "semdb/X900-fixture"\nstatus = "{status}"\n')
        manifest={"version":1,"id":"X900","status":status,"seeds":list(seeds),"required_artifacts":[]}
        if digest is DIGEST:
            digest=mod.experiment_records.preregistration_digest(table)
        if digest is not None:
            manifest["preregistration_sha256"]=digest
        write(root,"experiments/semdb/X900-fixture/experiment.toml","".join(f"{key} = {toml_value(value)}\n" for key,value in manifest.items()))
        config=CONFIG_HEADER+("\n"+toml_table("preregistration",table) if table is not None else "")
        write(root,"experiments/semdb/X900-fixture/config.toml",config)
        listed="version = 1\n\n"+toml_table("experiment.X900.required",required)
        for baseline in baselines:
            listed+="\n[[experiment.X900.baseline]]\n"+"".join(f"{key} = {toml_value(value)}\n" for key,value in baseline.items())
        if entry_extra:
            listed+="\n"+entry_extra
        write(root,"experiments/preregistration.toml",listed if listed_text is None else listed_text)
        write(root,"research/baselines/plain_model/config.toml",PLAIN_MODEL)
        if baseline_config is not None:
            values={name:value for name,value in baseline_config.items() if not isinstance(value,dict)}
            text="".join(f"{name} = {toml_value(value)}\n" for name,value in values.items())
            for name,section in baseline_config.items():
                if name not in values:
                    text+="\n"+toml_table(name,section)
            write(root,BASELINE_PATH,text)
        return root

    def assert_blocked(self, root: Path, *errors: str) -> None:
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
                self.assertEqual((code,lines),(0,[]))
        # A key the list does not require is preregistered too, empty lists
        # included; only required lists must hold something.
        self.assertEqual(gate(self.tree(table={**TABLE,"low_cells":[],"note":"x"})),(0,[]))
        # Without a baseline, only the table is checked.
        self.assertEqual(gate(self.tree(baselines=(),baseline_config=None)),(0,[]))

    def test_a_missing_required_key_keeps_an_experiment_from_leaving_planned(self):
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                without={key:value for key,value in TABLE.items() if key!="harness"}
                self.assert_blocked(self.tree(status=status,table=without),"X900: preregistration key harness is missing")
                self.assert_blocked(
                    self.tree(status=status,table=None,digest="0"*64),
                    "X900: config.toml has no [preregistration] table",
                )
                root=self.tree(status=status)
                (root/"experiments/semdb/X900-fixture/config.toml").unlink()
                self.assert_blocked(root,"X900: config.toml does not exist")

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
        digest=mod.experiment_records.preregistration_digest(TABLE)
        self.assert_blocked(self.tree(digest=None),"X900: experiment.toml names no preregistration_sha256")
        # The table changed after the manifest named its digest.
        changed={**TABLE,"schema":2}
        self.assert_blocked(
            self.tree(table=changed,digest=digest),
            f"X900: preregistration_sha256 {digest!r} is not {mod.experiment_records.preregistration_digest(changed)}, "
            "the digest of config.toml's [preregistration]",
        )
        for recorded in (digest.upper(),digest[:63],"",1):
            with self.subTest(recorded=recorded):
                self.assert_blocked(
                    self.tree(digest=recorded),
                    f"X900: preregistration_sha256 {recorded!r} is not {digest}, the digest of config.toml's [preregistration]",
                )

    def test_preregistered_seeds_must_be_the_manifests_seeds(self):
        for seeds in ((29,17),(17,),(17,29,43)):
            with self.subTest(seeds=seeds):
                self.assert_blocked(
                    self.tree(seeds=seeds),
                    f"X900: preregistered seeds [17, 29] are not the manifest's seeds {list(seeds)!r}",
                )
        # A table without seeds leaves the manifest's alone.
        without={key:value for key,value in TABLE.items() if key!="seeds"}
        required={key:kind for key,kind in REQUIRED.items() if key!="seeds"}
        self.assertEqual(gate(self.tree(table=without,required=required,seeds=(1,))),(0,[]))

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
            (None,[f"{where} does not exist"]),
        ]
        for config,errors in cases:
            with self.subTest(config=config):
                self.assert_blocked(self.tree(baseline_config=config),*errors)
        # A planned experiment's baseline is not checked.
        self.assertEqual(gate(self.tree(status="planned",baseline_config=None)),(0,[]))

    def test_the_list_names_only_registered_experiments_known_types_and_well_formed_baselines(self):
        where="experiments/preregistration.toml: X900"
        cases=[
            ({"required":{**REQUIRED,"seeds":"list"}},{},[f"{where}: key seeds has unknown type 'list'"]),
            ({"required":{}},{},[f"{where} requires no keys"]),
            ({"baselines":({**BASELINE,"path":"/etc/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"path":"../outside.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"keys":[]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths"]),
            ({"baselines":({**BASELINE,"keys":["model.revision",""]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths"]),
            ({"baselines":({key:value for key,value in BASELINE.items() if key!="status_key"},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({**BASELINE,"revision":"x"},)},{},[f"{where}: baseline 0 has unknown field revision"]),
            ({},{"entry_extra":"[experiment.X901.required]\nschema = \"int\"\n"},["experiments/preregistration.toml: X901 is not a registered experiment"]),
            ({},{"entry_extra":"[experiment.X900.optional]\nschema = \"int\"\n"},[f"{where} has unknown field optional"]),
            ({"required":{**REQUIRED,"seeds":["int"]}},{},[f"{where}: key seeds has unknown type ['int']"]),
            ({"required":{**REQUIRED,"seeds":1}},{},[f"{where}: key seeds has unknown type 1"]),
            ({"baselines":({**BASELINE,"keys":"model.revision"},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths"]),
            ({"baselines":({**BASELINE,"keys":[1]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths"]),
            ({"baselines":({**BASELINE,"status_key":""},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({**BASELINE,"status_key":1},)},{},[f"{where}: baseline 0 needs a status_key"]),
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

class EnrolledExperimentTests(unittest.TestCase):
    """The repository's own list and configurations, copied into a fixture
    tree with every listed experiment moved to `prepared`."""

    def prepared_tree(self, edit=None) -> Path:
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        paths={item["id"]:item["path"] for item in mod.load(ROOT/"experiments/registry.toml")["experiment"]}
        registry="version = 1\n"
        for exp_id in ENROLLED:
            registry+=f'\n[[experiment]]\nid = "{exp_id}"\npath = "{paths[exp_id]}"\nstatus = "prepared"\n'
            for name in ("experiment.toml","config.toml"):
                source=ROOT/"experiments"/paths[exp_id]/name
                text=source.read_text(encoding="utf-8").replace('status = "planned"','status = "prepared"')
                write(root,f"experiments/{paths[exp_id]}/{name}",text)
        write(root,"experiments/registry.toml",registry)
        for relative in (
            "experiments/preregistration.toml",
            "research/baselines/plain_model/config.toml",
            "research/baselines/rag_reference/config.toml",
            "research/baselines/strong_rag/config.toml",
        ):
            (root/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,root/relative)
        if edit:
            edit(root,paths)
        return root

    def test_the_list_enrolls_the_experiments_that_preregister(self):
        listed=mod.load(ROOT/"experiments/preregistration.toml")["experiment"]
        self.assertEqual(sorted(listed),sorted(ENROLLED))
        # S003 lists every key its design preregisters, low_cells included.
        self.assertEqual(len(listed["S003"]["required"]),58)
        self.assertEqual(listed["S003"]["required"]["low_cells"],"str-list")

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
            "F003: experiment.toml names no preregistration_sha256",
            f"Q003: preregistration key recall_margin_permille is a placeholder ({pinned})",
            f"Q003: preregistration key embedding_model is a placeholder ({pinned})",
            f"Q003: preregistration key embedding_revision is a placeholder ({pinned})",
            "Q003: experiment.toml names no preregistration_sha256",
            "Q003: baseline research/baselines/rag_reference/config.toml: hybrid.embedding_model is a placeholder ('must-be-pinned-before-reported-run')",
            *strong_errors("Q003"),
            f"R004: preregistration key public_regression_gate_permille is a placeholder ({pinned})",
            f"R004: preregistration key backward_transfer_min_permille is a placeholder ({pinned})",
            "R004: experiment.toml names no preregistration_sha256",
            f"M008: preregistration key token_budget is a placeholder ({pinned})",
            f"M008: preregistration key recency_window is a placeholder ({pinned})",
            f"M008: preregistration key recency_buffer_baseline is a placeholder ({pinned})",
            f"M008: preregistration key hybrid_retrieval_baseline is a placeholder ({pinned})",
            "M008: experiment.toml names no preregistration_sha256",
            f"E005: preregistration key token_budget is a placeholder ({pinned})",
            f"E005: preregistration key full_context_baseline is a placeholder ({pinned})",
            f"E005: preregistration key agent_memory_baselines has element 0 that is a placeholder ({pinned})",
            "E005: experiment.toml names no preregistration_sha256",
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
            config=mod.load(directory/"config.toml")
            table={**config.pop("preregistration"),**signed}
            header="".join(f"{key} = {toml_value(value)}\n" for key,value in config.items())
            (directory/"config.toml").write_text(header+"\n"+toml_table("preregistration",table),encoding="utf-8")
            digest=mod.experiment_records.preregistration_digest(table)
            manifest=(directory/"experiment.toml").read_text(encoding="utf-8")
            (directory/"experiment.toml").write_text(manifest+f'preregistration_sha256 = "{digest}"\n',encoding="utf-8")

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

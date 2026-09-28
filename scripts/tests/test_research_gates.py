import contextlib
import hashlib
import importlib.util
import io
import json
import os
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
TABLE={"schema":1,"harness":"fixture","seeds":[17,29],"programs":["rmw","set_op"],"see_intent":False}
REQUIRED={"schema":"int","harness":"str","seeds":"int-list","programs":"str-list","see_intent":"bool"}
BASELINE_PATH="research/baselines/fixture/config.toml"
BASELINE={"name":"fixture","path":BASELINE_PATH,"keys":["model.revision","answer.generators"],"status_key":"status"}
BASELINE_CONFIG={"status":"reference-implemented","model":{"revision":"0123abc"},"answer":{"generators":["g1","g2"]}}
DIGEST=object()

def gate(root: Path) -> tuple[int, list[str]]:
    """Run the gate on the tree at `root`, with the checks of completed
    experiments' archived results (git history the fixtures do not have)
    passing once they are asked about that tree, and return its exit code
    and error lines."""
    def archived(exp_id,experiment,results,given_root):
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
        `baseline_<name>_sha256`, the digest of its configuration file,
        unless the table sets it itself."""
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
                    table[f"baseline_{name}_sha256"]=hashlib.sha256(baseline_text.encode("utf-8")).hexdigest()
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
        manifest={"version":1,"id":"X900","status":status,"seeds":list(seeds),"required_artifacts":[]}
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
            ({"required":{**REQUIRED,"seeds":"list"}},{},[f"{where}: key seeds has unknown type 'list'"]),
            ({"required":{}},{},[f"{where} requires no keys"]),
            ({"baselines":({**BASELINE,"path":"/etc/config.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"path":"../outside.toml"},)},{},[f"{where}: baseline 0 needs a path inside the repository"]),
            ({"baselines":({**BASELINE,"keys":[]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({**BASELINE,"keys":["model.revision",""]},)},{},[f"{where}: baseline 0 needs a non-empty list of key paths in printable ASCII"]),
            ({"baselines":({key:value for key,value in BASELINE.items() if key!="status_key"},)},{},[f"{where}: baseline 0 needs a status_key"]),
            ({"baselines":({**BASELINE,"revision":"x"},)},{},[f"{where}: baseline 0 has unknown field revision"]),
            ({},{"entry_extra":"[experiment.X901.required]\nschema = \"int\"\n"},["experiments/preregistration.toml: X901 is not a registered experiment"]),
            ({},{"entry_extra":"[experiment.X900.optional]\nschema = \"int\"\n"},[f"{where} has unknown field optional"]),
            ({"required":{**REQUIRED,"gr\u00f6\u00dfe":"int"}},{},[f"{where}: key 'gr\u00f6\u00dfe' is not printable ASCII"]),
            ({"required":{**REQUIRED,"seeds":["int"]}},{},[f"{where}: key seeds has unknown type ['int']"]),
            ({"required":{**REQUIRED,"seeds":1}},{},[f"{where}: key seeds has unknown type 1"]),
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
        # A checkout that writes CRLF line endings holds the same protocol.
        self.assertEqual(gate(tree(good,content=text.replace("\n","\r\n"))),(0,[]))
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
        for named in (path,"../PROTOCOL.md","/etc/hostname","experiments/semdb/X900-fixture"):
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

    def test_a_baseline_is_frozen_by_its_whole_configuration(self):
        text=(self.tree()/BASELINE_PATH).read_text(encoding="utf-8")
        frozen=hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(gate(self.tree(table={**TABLE,"baseline_fixture_sha256":frozen})),(0,[]))
        self.assert_blocked(
            self.tree(freeze_baselines=False),
            f"X900: preregistration key baseline_fixture_sha256 is missing; it must be {frozen}, the digest of {BASELINE_PATH}",
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
                moved=mod.experiment_records.preregistered_file_digest(root/BASELINE_PATH)
                self.assert_blocked(
                    root,
                    f"X900: preregistration key baseline_fixture_sha256 {frozen!r} is not {moved}, the digest of {BASELINE_PATH}",
                )
        # A checkout that writes CRLF line endings holds the same baseline.
        root=self.tree(table={**TABLE,"baseline_fixture_sha256":frozen})
        (root/BASELINE_PATH).write_bytes(text.replace("\n","\r\n").encode("utf-8"))
        self.assertEqual(gate(root),(0,[]))
        self.assert_blocked(
            self.tree(table={**TABLE,"baseline_fixture_sha256":"must-be-pinned-before-prepared"}),
            "X900: preregistration key baseline_fixture_sha256 is a placeholder ('must-be-pinned-before-prepared')",
        )

    RECORD="experiments/semdb/X900-fixture/results/run-20260101T000000.000000Z-seed-17.json"
    AGGREGATE="experiments/semdb/X900-fixture/results/run.json"
    MANIFEST="experiments/semdb/X900-fixture/experiment.toml"

    def frozen(self, root: Path) -> tuple[str, str]:
        """The digests the fixture's manifest names: the table's and the rules'."""
        manifest=mod.load(root/self.MANIFEST)
        return manifest["preregistration_sha256"],manifest["preregistration_rules_sha256"]

    def record(self, root: Path, commit: str, **changes) -> dict:
        """A run record as run_experiment.py writes one at `commit`."""
        digest,rules=self.frozen(root)
        record={
            "git_sha":commit,
            "manifest":{"id":"X900","preregistration_sha256":digest,"preregistration_rules_sha256":rules},
            "manifest_sha256":hashlib.sha256(git(root,"show",f"{commit}:{self.MANIFEST}").encode("utf-8")+b"\n").hexdigest(),
        }
        record.update(changes)
        return record

    def aggregate(self, root: Path, commit: str, **changes) -> dict:
        """An aggregate run.json of the runs at `commit`."""
        digest,rules=self.frozen(root)
        return {"git_sha":commit,"preregistration_sha256":digest,"preregistration_rules_sha256":rules,**changes}

    def test_archived_runs_must_name_the_frozen_digests_and_a_commit_that_holds_them(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        for status in ("prepared","running","completed","failed"):
            with self.subTest(status=status):
                root=self.tree(status=status)
                commit=commit_all(root)
                short=commit[:12]
                digest,rules=self.frozen(root)
                write(root,self.RECORD,json.dumps(self.record(root,commit)))
                write(root,self.AGGREGATE,json.dumps(self.aggregate(root,commit)))
                # Files other than run records are not run records.
                write(root,"experiments/semdb/X900-fixture/results/metrics.json","{}")
                write(root,"experiments/semdb/X900-fixture/results/run_notes.json","[]")
                self.assertEqual(gate(root),(0,[]))
                # A checkout that converts line endings hashed the manifest
                # with CRLF; that is the committed manifest too.
                crlf=hashlib.sha256(git(root,"show",f"{commit}:{self.MANIFEST}").replace("\n","\r\n").encode("utf-8")+b"\r\n").hexdigest()
                write(root,self.RECORD,json.dumps(self.record(root,commit,manifest_sha256=crlf)))
                self.assertEqual(gate(root),(0,[]))
                write(root,self.RECORD,json.dumps(self.record(root,commit)))
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
                write(root,self.RECORD,json.dumps(self.record(root,commit)))
                self.assertEqual(gate(root),(0,[]))
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

    def test_a_preregistration_rewritten_after_its_runs_fails_whatever_else_is_rewritten(self):
        name=self.RECORD.removeprefix("experiments/semdb/X900-fixture/")
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        commit_all(root,"records")
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
        write(root,self.AGGREGATE,json.dumps({**self.aggregate(root,ran),"preregistration_sha256":after[0]}))
        commit_all(root,"rewrite")
        short=ran[:12]
        self.assert_blocked(
            root,
            f"X900: {name} ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: {name} ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900: {name} was changed after it was committed (2 commits touch it)",
            f"X900: results/run.json ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: results/run.json ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
        )

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
        frozen_table={**table,"baseline_fixture_sha256":mod.experiment_records.preregistered_file_digest(root/BASELINE_PATH)}
        fixed=self.tree(status="running",table=frozen_table,required=required)
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(fixed/relative,root/relative)
        self.assertNotEqual(manifest,(root/self.MANIFEST).read_text(encoding="utf-8"))
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        short=ran[:12]
        code,lines=gate(root)
        self.assertEqual(code,1)
        self.assertIn(f"X900: {name} ran at {short}, where {path} is not the file frozen as protocol",lines)
        self.assertIn(f"X900: {name} ran at {short}, where baseline fixture is not the frozen configuration",lines)

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
            ["experiments/preregistration.toml: X900: key seeds has unknown type 'list'"],
        )
        # An experiment the list does not name launches as before.
        root=self.tree(listed_text="version = 1\n")
        self.assertEqual(mod.launch_errors(root,"X900"),[])

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

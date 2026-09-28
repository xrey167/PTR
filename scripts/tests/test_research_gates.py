import contextlib
import datetime
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
    SHA-256 of its text with CRLF read as LF."""
    return mod.experiment_records.canonical_digest({
        name:("100755" if name in executable else "100644")+" "+hashlib.sha256(text.replace("\r\n","\n").encode("utf-8")).hexdigest()
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
        # A checkout that writes CRLF line endings holds the same protocol as
        # the LF text committed.
        root=tree(good)
        commit_all(root)
        (root/path).write_bytes(text.replace("\n","\r\n").encode("utf-8"))
        self.assertEqual(gate(root),(0,[]))
        # Committed with CRLF, the text would read alike with LF line
        # endings, which a command reading it tells apart: it is refused, in
        # the tree and at the commit, which froze nothing.
        crlf=(f"X900: preregistration key protocol names {path!r}, which is committed with CRLF line endings; its "
              "digest reads them as LF, so a preregistered text file is committed with LF line endings")
        root=tree(good,content=text.replace("\n","\r\n"))
        self.assert_blocked(root,crlf)
        committed=commit_all(root)
        self.assert_blocked(root,crlf)
        self.assertIsNone(mod.frozen_file_at(root,committed,path))
        self.assertIsNone(mod.launchable_at(root,committed,"X900","experiments/semdb/X900-fixture"))
        # Committed with LF line endings, it is the frozen protocol.
        (root/path).write_bytes(text.encode("utf-8"))
        commit_all(root,"protocol with LF line endings")
        self.assertEqual(gate(root),(0,[]))
        # A binary file's carriage returns are content, digested as they are.
        binary=b"\x00\r\n\x01"
        self.assertIsNone(mod.file_form_problem(binary,"100644"))
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
        write(root,self.AGGREGATE,json.dumps({**self.aggregate(root,ran),"preregistration_sha256":after[0]}))
        commit_all(root,"rewrite")
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
            # The aggregate as it was first committed saw the outcome under
            # the old preregistration.
            f"X900: results/run.json as committed at {recorded[:12]} names preregistration_sha256 {before[0]!r}, not {after[0]}, "
            "the digest the experiment is frozen at",
            f"X900: results/run.json as committed at {recorded[:12]} ran at {short}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900: results/run.json as committed at {recorded[:12]} ran at {short}, whose experiment.toml names other preregistration digests than the frozen ones",
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

    def ran(self, status="running", **tree) -> tuple[Path, str]:
        """A fixture tree at `status` with one run record committed, and the
        commit the record names."""
        root=self.tree(status=status,**tree)
        ran=commit_all(root)
        write(root,self.RECORD,json.dumps(self.record(root,ran)))
        commit_all(root,"records")
        self.assertEqual(gate(root),(0,[]))
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
        # The aggregate is bound the same way.
        root,ran=self.ran()
        at=f"X900: {name} ran at {ran[:12]}"
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
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
                        self.assertEqual(gate(root),(0,[]))
                    else:
                        self.assert_blocked(
                            root,
                            f"X900: {name} ran at {ran[:12]}, where it was {then!r}; it cannot be {now!r} after that, "
                            "since a status moves only from prepared to running to completed or failed, or to superseded",
                            f"X900 was frozen at {ran[:12]}, where it was {then!r}; it cannot be {now!r} after that, "
                            "since a status moves only from prepared to running to completed or failed, or to superseded",
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
        self.assertEqual(gate(root),(0,[]))
        self.assertEqual(mod.launch_errors(root,"X900"),[
            "X900 preregisters (experiments/preregistration.toml) and is 'superseded': it runs only once its "
            "preregistration is frozen and it is prepared, running, completed or failed",
        ])
        # A record whose commit held the experiment before it was prepared
        # did not come from the runner, which refuses to launch it there.
        root=self.tree(status="running")
        self.edit(root,self.MANIFEST,'status = "running"','status = "planned"')
        early=commit_all(root)
        self.edit(root,self.MANIFEST,'status = "planned"','status = "running"')
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
                self.assertEqual(gate(root),(0,[]))
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
        for results_dir in ("../shared",".","./"):
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
        # Only the aggregate was committed; it shows the outcome.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
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
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,rewrite)))
        commit_all(root,"aggregate again")
        at=f"X900: results/run.json as committed at {recorded[:12]}"
        earlier=[
            f"{at} names preregistration_sha256 {before[0]!r}, not {after[0]}, the digest the experiment is frozen at",
            f"{at} ran at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"{at} ran at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
            f"X900 was frozen at {ran[:12]}, whose config.toml holds another [preregistration] than the frozen one",
            f"X900 was frozen at {ran[:12]}, whose experiment.toml names other preregistration digests than the frozen ones",
        ]
        self.assert_blocked(root,*earlier)
        # Written again in a new results directory instead, the old
        # aggregate is still checked where it was committed.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        commit_all(root,"aggregate")
        for relative in ("experiments/semdb/X900-fixture/config.toml",self.MANIFEST):
            shutil.copyfile(rewritten/relative,root/relative)
        write(root,self.MANIFEST,(root/self.MANIFEST).read_text(encoding="utf-8")+'results_dir = "new-results"\n')
        rewrite=commit_all(root,"rewrite")
        write(root,self.AGGREGATE.replace("/results/","/new-results/"),json.dumps(self.aggregate(root,rewrite)))
        commit_all(root,"new aggregate")
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
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        commit_all(root,"aggregate")
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,metrics_sha256="0"*64)))
        commit_all(root,"aggregate again")
        self.assertEqual(gate(root),(0,[]))
        # Deleted and put back as it was, it is the aggregate committed.
        (root/self.AGGREGATE).unlink()
        commit_all(root,"aggregate deleted")
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran,metrics_sha256="0"*64)))
        commit_all(root,"aggregate restored")
        self.assertEqual(gate(root),(0,[]))
        # A version committed as a link was another file's content, read
        # through it; git holds only the link's target path.
        root=self.tree(status="running")
        ran=commit_all(root)
        write(root,"experiments/semdb/X900-fixture/elsewhere.json",json.dumps(self.aggregate(root,ran)))
        self.link(Path("../elsewhere.json"),root/self.AGGREGATE)
        linked=commit_all(root,"aggregate linked")
        (root/self.AGGREGATE).unlink()
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,ran)))
        commit_all(root,"aggregate")
        self.assert_blocked(root,f"X900: results/run.json as committed at {linked[:12]} is not a regular file")

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
        write(root,self.AGGREGATE,json.dumps(self.aggregate(root,commit,seed=17)))
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
            write(root,f"{results}/{output}","{}\n" if output!="run.json" else json.dumps(self.aggregate(root,first)))
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
                # Leaving planned, the owner names what each one runs.
                text=source.read_text(encoding="utf-8").replace('status = "planned"','status = "prepared"').replace('entrypoint = ""','entrypoint = "bench <seed>"')
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
        git(root,"init","-q")
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

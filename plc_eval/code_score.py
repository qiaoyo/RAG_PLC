import json
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import lizard

from st_wrapper import wrap_st_code


MATIEC_INCLUDE = "/home/simple/matiec/lib"
MATIEC_GCC_INCLUDE = "/home/simple/matiec/lib/C"


def run_cmd(cmd: str, cwd: Path) -> Tuple[int, str, str]:
    res = subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode, res.stdout, res.stderr


def wrap_candidate_to_full_st(candidate_file: Path, workdir: Path) -> Path:
    """Wrap raw candidate code file to full ST, saved as wrapped.st in workdir."""
    wrapped = workdir / "wrapped.st"
    wrap_st_code(str(candidate_file), str(wrapped))
    return wrapped


def compile_and_link(wrapped_st: Path, main_c_path: Path, workdir: Path, iec2c_flags: str = "") -> Tuple[float, Dict, Path]:
    detail: Dict = {"stage": "compile"}
    # iec2c conversion in-place
    ret, out, err = run_cmd(f"iec2c -I {MATIEC_INCLUDE} {iec2c_flags} {wrapped_st.name}", cwd=workdir)
    detail["iec2c_stdout"] = out
    detail["iec2c_stderr"] = err
    if ret != 0:
        detail["status"] = "iec2c_failed"
        return 0.0, detail, workdir / "plc_runner"

    # copy main.c
    target_main = workdir / "main.c"
    if not target_main.exists():
        shutil.copy(main_c_path, target_main)

    # compile single binary (no duplicate compile in runtime step)
    ret, _, gcc_err = run_cmd(
        f"gcc -w -I {MATIEC_GCC_INCLUDE} main.c Config0.c Res0.c POUS.c -o plc_runner",
        cwd=workdir,
    )
    detail["gcc_stderr"] = gcc_err
    if ret != 0:
        detail["status"] = "gcc_failed"
        return 0.0, detail, workdir / "plc_runner"

    detail["status"] = "ok"
    return 1.0, detail, workdir / "plc_runner"


def static_score(workdir: Path) -> Tuple[float, Dict]:
    c_files = [str(p) for p in workdir.glob("*.c")]
    if not c_files:
        return 0.0, {"stage": "static", "status": "no_c_files"}
    complexities: List[int] = []
    for f in c_files:
        analysis = lizard.analyze_file(f)
        for func in analysis.function_list:
            complexities.append(func.cyclomatic_complexity)
    if not complexities:
        return 1.0, {"stage": "static", "status": "no_functions"}
    avg_cplx = sum(complexities) / len(complexities)
    score = max(0.0, min(1.0, 1 - (avg_cplx - 10) / 20)) if avg_cplx > 10 else 1.0
    return score, {"stage": "static", "status": "ok", "avg_complexity": avg_cplx}


def runtime_score(binary_path: Path, workdir: Path, timeout_s: int = 2) -> Tuple[float, Dict]:
    detail = {"stage": "runtime"}
    if not binary_path.exists():
        detail["status"] = "no_binary"
        return 0.0, detail
    proc = subprocess.run(
        f"timeout {timeout_s}s ./plc_runner",
        shell=True,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    detail["stdout"] = proc.stdout
    detail["stderr"] = proc.stderr
    if proc.returncode == 0:
        detail["status"] = "ok"
        return 1.0, detail
    if proc.returncode == 124:
        detail["status"] = "timeout"
    else:
        detail["status"] = "crash"
    return 0.0, detail


def logic_score(candidate_code: str, reference_code: str, judge=None) -> Tuple[float, Dict]:
    if judge is not None:
        result = judge.score_raw(reference_code=reference_code, candidate_code=candidate_code)
        return float(result.get("score", 0.0)), {"stage": "logic", "status": "llm", "reason": result.get("reason", "")}
    ratio = SequenceMatcher(None, candidate_code, reference_code).ratio()
    return ratio, {"stage": "logic", "status": "similarity", "ratio": ratio}


def evaluate_candidate_file(
    candidate_file: Path,
    reference_code: str,
    sample_dir: Path,
    main_c_path: Path,
    iec2c_flags: str = "",
    judge=None,
) -> Dict:
    """Evaluate a candidate code file, saving all artifacts/logs in sample_dir."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    wrapped = wrap_candidate_to_full_st(candidate_file, sample_dir)
    compile_s, compile_detail, binary_path = compile_and_link(wrapped, main_c_path, sample_dir, iec2c_flags)
    static_s, static_detail = static_score(sample_dir)
    runtime_s, runtime_detail = runtime_score(binary_path, sample_dir)
    candidate_text = candidate_file.read_text(encoding="utf-8")
    logic_s, logic_detail = logic_score(candidate_text, reference_code, judge=judge)

    total = (2 * compile_s + 2 * static_s + 2 * runtime_s + 4 * logic_s) / 10.0
    result = {
        "score": total,
        "components": {
            "compile": compile_s,
            "static": static_s,
            "runtime": runtime_s,
            "logic": logic_s,
        },
        "details": [compile_detail, static_detail, runtime_detail, logic_detail],
    }
    # persist log
    log_path = sample_dir / "log.json"
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_candidate(
    candidate_code: str,
    reference_code: str,
    main_c_path: Path,
    iec2c_flags: str = "",
    judge=None,
) -> Dict:
    """Backward-compatible wrapper: write candidate_code into a temp dir and evaluate."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        raw_file = tmpdir / "raw_code.txt"
        raw_file.write_text(candidate_code, encoding="utf-8")
        return evaluate_candidate_file(
            candidate_file=raw_file,
            reference_code=reference_code,
            sample_dir=tmpdir,
            main_c_path=main_c_path,
            iec2c_flags=iec2c_flags,
            judge=judge,
        )


def evaluate_split(root: Path, split: str) -> None:
    """Batch score a split under root, using plc_eval/splits.json as reference."""
    split_file = Path("/home/simple/Downloads/RAG_PLC/plc_eval/split.json")
    main_c = Path("/home/simple/Downloads/RAG_PLC/plc_eval/main.c")
    limit = None
    # load references
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    refs = splits.get(split, [])

    split_dir = root / split
    subdirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    if limit:
        subdirs = subdirs[:limit]
    for idx, sub in enumerate(subdirs, start=1):
        ref_item = refs[idx - 1] if idx - 1 < len(refs) else {}
        reference = ref_item.get("output", "")
        candidate_file = sub / "raw_code.st"
        result = evaluate_candidate_file(
            candidate_file=candidate_file,
            reference_code=reference,
            sample_dir=sub,
            main_c_path=main_c,
        )
        # log.json already written inside evaluate_candidate_file
        print(f"[{split}/{sub.name}] score={result['score']:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch score code directories under data/result")
    parser.add_argument("--root", type=Path, default=Path("data/result"), help="根目录，包含train/val/test子目录")
    parser.add_argument("--split", type=str, default="test", help="选择子目录")
    args = parser.parse_args()

    evaluate_split(root=args.root, split=args.split)

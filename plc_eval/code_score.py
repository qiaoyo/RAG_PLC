import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import lizard

from plc_eval.st_wrapper import wrap_st_code


def run_cmd(cmd: str, cwd: Path) -> Tuple[int, str, str]:
    res = subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode, res.stdout, res.stderr


def wrap_candidate_to_full_st(candidate_code: str, workdir: Path) -> Path:
    raw = workdir / "candidate_raw.st"
    raw.write_text(candidate_code, encoding="utf-8")
    wrapped = workdir / "wrapped.st"
    wrap_st_code(str(raw), str(wrapped))
    return wrapped


def compile_score(wrapped_st: Path, main_c_path: Path, workdir: Path, iec2c_flags: str = "") -> Tuple[float, Dict]:
    detail: Dict = {"stage": "compile"}
    ret, out, err = run_cmd(f"iec2c {iec2c_flags} {wrapped_st.name}", cwd=workdir)
    detail["iec2c_stdout"] = out
    detail["iec2c_stderr"] = err
    if ret != 0:
        detail["status"] = "iec2c_failed"
        return 0.0, detail

    c_sources = " ".join(str(p.name) for p in workdir.glob("*.c"))
    ret, _, err = run_cmd(f"gcc -w -I {workdir} {main_c_path} {c_sources} -o runner", cwd=workdir)
    detail["gcc_stderr"] = err
    if ret != 0:
        detail["status"] = "gcc_failed"
        return 0.0, detail

    detail["status"] = "ok"
    return 1.0, detail


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
    # Normalize: <=10 =>1, >=30 =>0, linear in between
    score = max(0.0, min(1.0, 1 - (avg_cplx - 10) / 20)) if avg_cplx > 10 else 1.0
    return score, {"stage": "static", "status": "ok", "avg_complexity": avg_cplx}


def runtime_score(main_c_path: Path, workdir: Path, timeout_s: int = 2) -> Tuple[float, Dict]:
    c_sources = " ".join(str(p.name) for p in workdir.glob("*.c"))
    env = {"ASAN_OPTIONS": "detect_leaks=0"}
    compile_cmd = f"gcc -fsanitize=address -g -I {workdir} {main_c_path} {c_sources} -o runner_asan"
    ret, _, err = run_cmd(compile_cmd, cwd=workdir)
    detail = {"stage": "runtime", "compile_stderr": err}
    if ret != 0:
        detail["status"] = "asan_compile_failed"
        return 0.0, detail
    proc = subprocess.run(
        "timeout {t}s ./runner_asan".format(t=timeout_s),
        shell=True,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**env, **dict(**env)},
    )
    if proc.returncode == 0:
        detail["status"] = "ok"
        return 1.0, detail
    if proc.returncode == 124:
        detail["status"] = "timeout"
    else:
        detail["status"] = "crash"
        detail["stderr"] = proc.stderr
    return 0.0, detail


def logic_score(candidate_code: str, reference_code: str, judge=None) -> Tuple[float, Dict]:
    """Logic correctness: prefer LLM judge; fallback to similarity."""
    if judge is not None:
        result = judge.score_raw(reference_code=reference_code, candidate_code=candidate_code)
        return float(result.get("score", 0.0)), {"stage": "logic", "status": "llm", "reason": result.get("reason", "")}
    ratio = SequenceMatcher(None, candidate_code, reference_code).ratio()
    return ratio, {"stage": "logic", "status": "similarity", "ratio": ratio}


def evaluate_candidate(
    candidate_code: str,
    reference_code: str,
    main_c_path: Path,
    iec2c_flags: str = "",
    judge=None,
) -> Dict:
    """Return weighted score and details."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        wrapped = wrap_candidate_to_full_st(candidate_code, workdir)
        compile_s, compile_detail = compile_score(wrapped, main_c_path, workdir, iec2c_flags)
        static_s, static_detail = static_score(workdir)
        runtime_s, runtime_detail = runtime_score(main_c_path, workdir)
        logic_s, logic_detail = logic_score(candidate_code, reference_code, judge=judge)

    total = (2 * compile_s + 2 * static_s + 2 * runtime_s + 4 * logic_s) / 10.0
    return {
        "score": total,
        "components": {
            "compile": compile_s,
            "static": static_s,
            "runtime": runtime_s,
            "logic": logic_s,
        },
        "details": [compile_detail, static_detail, runtime_detail, logic_detail],
    }


if __name__ == "__main__":
    from pathlib import Path

    sample_code = "(* sample *)\nVAR\nx:BOOL;\nEND_VAR\nx := TRUE;"
    ref_code = sample_code
    result = evaluate_candidate(
        candidate_code=sample_code,
        reference_code=ref_code,
        main_c_path=Path(__file__).parent / "main.c",
    )
    print(result)

import modal
from pydantic import BaseModel
import prover

MATHLIB_PATH = "/root/mathlib4"
DEFAULT_LAKE_PATH = "/root/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/root/mathlib4/"

class VerifyRequest(BaseModel):
    code: str

app = modal.App("verifier")

verifier_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .apt_install("git")
    .apt_install("curl")
    .workdir("/root")
    .copy_local_file("mathlib4.tar.gz")
    .run_commands("tar xzvf mathlib4.tar.gz")
    .env(
        {
            # "LEAN_VERSION": "leanprover/lean4:nightly",
            "LEAN_VERSION": "leanprover/lean4:v4.9.0-rc1",
        }
    )
    .run_commands(
        "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain $LEAN_VERSION",
    )
    .workdir("/root/mathlib4")
    .run_commands("/root/.elan/bin/lake build")
)

with verifier_image.imports():
    import json
    import time
    import tempfile
    import subprocess
    import traceback
    import asyncio


@app.function(
    image=verifier_image,
    mounts=[
        modal.Mount.from_local_python_packages("prover"),
    ],
    cpu=16.0,
)
def verify_lean4_file(
    code,
    lake_path=DEFAULT_LAKE_PATH,
    lean_workspace=DEFAULT_LEAN_WORKSPACE,
    last_env=None,
    verbose=False,
    timeout=300,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
):
    if verbose:
        import os

        print("CURRENT DIRECTORY", os.getcwd(), os.listdir())
        print("ROOT", os.listdir("/"))
        print("ELAN", os.listdir("/root/.elan/bin"))
        print("MATHLIB", os.listdir("/root/mathlib4"))
    command = dict(
        cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises
    )
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ""
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
            )
        if verbose:
            print("stdout", outputs.stdout)
            print("stderr", outputs.stderr)
        result = json.loads(outputs.stdout)
        ast_results = (
            lean4_parser(code, result["ast"])
            if "ast" in result and result["ast"]
            else {}
        )
        result = {
            "sorries": result.get("sorries", []),
            "tactics": result.get("tactics", []),
            "errors": [
                m for m in result.get("messages", []) if m["severity"] == "error"
            ],
            "warnings": [
                m for m in result.get("messages", []) if m["severity"] == "warning"
            ],
            "infos": [m for m in result.get("messages", []) if m["severity"] == "info"],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in warning["data"]
                or "failed" in warning["data"]
                for warning in result["warnings"]
            )
        )
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
        }
    result["verify_time"] = time.time() - start_time
    return result


@app.function()
@modal.web_endpoint(method="POST")
async def verify(requests: list[VerifyRequest]):
    """
    Execute evaluation of each result async and return all results
    """
    results = await asyncio.gather(*[
        verify_lean4_file.remote.aio(r.code)
        for r in requests
    ])
    return results


@app.local_entrypoint()
def main():
    code = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,
    Nat.succ_add]
  have h₁' : a * r = 2 := by simpa [h₀] using h₁
  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂
  have h₃ : r ^ 2 = 3 := by
    nlinarith
  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
    apply eq_or_eq_neg_of_sq_eq_sq <;>
    field_simp <;>
    nlinarith
  simpa [h₀] using h₄
"""
    #print(verify_lean4_file.remote(code, verbose=True))
    import datasets
    dataset = datasets.load_dataset("cat-searcher/minif2f-lean4")
    sample = dataset["test"][0]
    code = "import Mathlib\n" + sample["formal_statement"]
    print(code)
    print(verify_lean4_file.remote(code, verbose=True))



# # Fast inference with vLLM (Gemma 7B)
#
# In this example, we show how to run basic LLM inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of [PagedAttention](https://arxiv.org/abs/2309.06180), which speeds up inference on longer sequences with optimized key-value caching.
# You can read more about PagedAttention [here](https://charlesfrye.github.io/programming/2023/11/10/llms-systems.html).
#
# We'll run the [Gemma 7B Instruct](https://huggingface.co/google/gemma-7b-it) large language model.
# Gemma is the weights-available version of Google's Gemini model series.
#
# The "7B" in the name refers to the number of parameters (floating point numbers used to control inference)
# in the model. Applying those 7,000,000,000 numbers onto an input is a lot of work,
# so we'll use a GPU to speed up the process -- specifically, a top-of-the-line [NVIDIA H100](https://modal.com/blog/introducing-h100).
#
# "Instruct" means that this version of Gemma is not simply a statistical model of language,
# but has been fine-tuned to follow instructions -- like ChatGPT or Claude,
# it is a model of an assistant that can understand and follow instructions.
#
# You can expect cold starts in under 30 seconds and well over 1000 tokens/second throughput.
# The larger the batch of prompts, the higher the throughput. For example, with the 64 prompts below,
# we can produce nearly 15k tokens with a latency just over 5 seconds, for a throughput of >2.5k tokens/second.
# That's a lot of text!
#
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# just change the model name. You may also need to change engine configuration, like `trust_remote_code`,
# or GPU configuration, in order to run some models.
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

import modal

MODEL_DIR = "/model"
# MODEL_NAME = "arbius/DeepSeek-Prover-V1.5-RL-GGUF"
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V1.5-RL"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HF_TOKEN`. Functions that inject this secret
# will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# You may need to accept the license agreement from an account associated with that Hugging Face Token
# to download the model.
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Docker Hub image by NVIDIA and install `vLLM`.
# Then we’ll use `run_function` to execute `download_model_to_image`
# and save the resulting files to the container image -- that way we don't need
# to redownload the weights every time we change the server's code or start up more instances of the server.

cuda_version = "12.4.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
ios = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{ios}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .apt_install("curl")
    .pip_install("pip", "packaging", "setuptools", "uv")
    .copy_local_dir("./src")
    .copy_local_dir("./configs")
    .copy_local_file("./pyproject.toml")
    .copy_local_file("./requirements.txt")
    .copy_local_file("./README.md")
    .run_commands(
        # "uv venv",
        # ". .venv/bin/activate",
        "uv pip install --quiet --no-progress --system .[server]",
        "uv pip install --quiet --no-progress --system hatchling editables ninja setuptools packaging wheel",
        "uv pip install --quiet --no-progress --system flash-attn --no-build-isolation",
        "uv pip install --quiet --no-progress --system hf-transfer",
    )
    # Use the barebones hf-transfer package for maximum download speeds. Varies from 100MB/s to 1.5 GB/s,
    # so download times can vary from under a minute to tens of minutes.
    # If your download slows down or times out, try interrupting and restarting.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("my-huggingface-secret")],
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
)

app = modal.App("prover", image=image)

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm

# ## Encapulate the model in a class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean!

GPU_CONFIG = modal.gpu.A100(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("my-huggingface-secret")])
class Model:
    @modal.enter()
    def load(self):
        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = vllm.LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
            max_num_batched_tokens=8192,
            seed=1,
            trust_remote_code=True,
        )

    @modal.method()
    def generate(self, user_questions):
        prompts = user_questions

        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=2048,
            n=1,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )
        # return result
        return result[0].outputs[0].text

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. Run it by executing the command `modal run vllm_inference.py`.
#
# The examples below are meant to put the model through its paces, with a variety of questions and prompts.
# We also calculate the throughput and latency we achieve.
@app.local_entrypoint()
def main():
    import re
    import prover

    prompt = r"""Complete the following Lean 4 code:

    ```lean4
    """

    code_prefix = r"""import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat

    /-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
    Show that it is $\frac{2\sqrt{3}}{3}$.-/
    theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
      (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
    """
    questions = [
        prompt + code_prefix,
    ]
    model = Model()
    # model_outputs = model.generate.remote(questions)
    # generated_text = model_outputs[0].outputs[0].text
    generated_text = model.generate.remote(questions)
    result = prompt + code_prefix + generated_text

    # evaluate generated text with another modal function
    verify = modal.Function.lookup("verifier", "verify_lean4_file")
    request = re.search(r"```lean4\n(.*?)\n```", result, re.DOTALL).group(1)
    print(verify.remote(request))

    # Expected output (verify_time may vary):
    """{'sorries': [], 'tactics': [], 'errors': [], 'warnings': [{'severity': 'warning', 'pos': {'line': 14, 'column': 7}, 'endPos': {'line': 14, 'column': 10}, 'data': "unused variable `h₁'`\nnote: this linter can be disabled with `set_option linter.unusedVariables false`"}, {'severity': 'warning', 'pos': {'line': 15, 'column': 7}, 'endPos': {'line': 15, 'column': 10}, 'data': "unused variable `h₂'`\nnote: this linter can be disabled with `set_option linter.unusedVariables false`"}, {'severity': 'warning', 'pos': {'line': 19, 'column': 35}, 'endPos': {'line': 19, 'column': 38}, 'data': 'Used `tac1 <;> tac2` where `(tac1; tac2)` would suffice\nnote: this linter can be disabled with `set_option linter.unnecessarySeqFocus false`'}, {'severity': 'warning', 'pos': {'line': 20, 'column': 15}, 'endPos': {'line': 20, 'column': 18}, 'data': 'Used `tac1 <;> tac2` where `(tac1; tac2)` would suffice\nnote: this linter can be disabled with `set_option linter.unnecessarySeqFocus false`'}], 'infos': [], 'system_messages': '', 'system_errors': None, 'ast': {}, 'verified_code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?\nShow that it is $\x0crac{2\\sqrt{3}}{3}$.-/\ntheorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)\n  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,\n    Nat.succ_add]\n  have h₁' : a * r = 2 := by simpa [h₀] using h₁\n  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂\n  have h₃ : r ^ 2 = 3 := by\n    nlinarith\n  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by\n    apply eq_or_eq_neg_of_sq_eq_sq <;>\n    field_simp <;>\n    nlinarith\n  simpa [h₀] using h₄", 'pass': True, 'complete': True, 'verify_time': 23.28123140335083}"""

    lean4_scheduler.close()

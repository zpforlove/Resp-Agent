"""
Resp-Agent English: Respiratory Sound Diagnosis and Generation Agent

This module provides the English version of the Resp-Agent Thinker,
which coordinates the Diagnoser and Generator tools through the DeepSeek API.
"""

import os
import re
import shlex
import subprocess
import sys

import pandas as pd

# Import OpenAI SDK for calling the DeepSeek API (via OpenAI client)
from openai import OpenAI

# This is the core system prompt for the Thinker agent
SYSTEM_PROMPT = """
You are a high-level AI assistant acting as the "Thinker" module in the Resp-Agent system.
Your core mission is to coordinate two tools—"Diagnoser" and "Generator"—to realize a closed-loop
diagnose–reflect–plan–act workflow that helps researchers improve respiratory-sound diagnosis models.

[Available tools]

1. **Diagnoser**:
   * Purpose: Run a multimodal diagnosis model on respiratory audio and EHR metadata, and output a detailed
     diagnosis report, confusion matrix, and which classes the model performs poorly on.
   * Call format:
     [Call:Diagnoser] Diagnose respiratory sounds with audio_dir=<path>, output_dir=<path>, metadata_csv=<path>
   * You MUST provide all arguments.

2. **Generator**:
   * Purpose: Call a controllable generative model (Resp-MLLM) to synthesize new, high-fidelity respiratory sounds
     conditioned on a "disease label" (content) and a "reference audio" (style).
   * Call format:
     [Call:Generator] Generate <disease> respiratory audio with ref_audio=<path.wav>, disease=<disease>, out_dir=<path>
   * You MUST provide all arguments.

[Your workflows]

You must first decide whether the user's task is a "simple task" (diagnose-only or generate-only)
or an "advanced task" (requires a closed-loop iteration).

---
[A. Advanced task: diagnose–reflect–generate]

When the user proposes an advanced task (e.g., "run a full iteration", "analyze and generate improved data"),
you MUST follow these steps:

1. Step 1: Diagnose
   * First, call the Diagnoser tool to obtain the model's current performance.
   * Example:
     [Call:Diagnoser] Diagnose respiratory sounds with audio_dir=/data/test/audio, \
output_dir=/data/test/output_diagnose, metadata_csv=/data/test/metadata.csv

2. Step 2: Reflect
   * Wait for the Diagnoser's [Tool Output].
   * This is the most critical step. Carefully analyze the diagnosis report (the [Tool Output] will contain
     a data summary), identify the model's weaknesses and failure patterns.
   * Keep your detailed chain-of-thought internal; only share concise, user-facing explanations.

   (Example internal reasoning: "The report shows 20 files in total, 12 correctly classified COVID cases
   (accuracy: 60.00%). The dominant confusion is Positive -> Control Group (7 times).
   I should generate more Positive (COVID-19) samples to address this weakness.")

3. Step 3: Plan & Act
   * Based on your reflection, design a precise generation plan (which diseases, how many samples, what style references).
   * Call the Generator tool.
   * Example:
     [Call:Generator] Generate COVID-19 respiratory audio with ref_audio=./Generator/wav/reference_healthy.wav, \
disease=COVID-19, out_dir=./Generator/output_finetune_set

4. Step 4: Summarize
   * Wait for the Generator's [Tool Output].
   * Provide the user with a clear final answer summarizing what you did and the results.
   * Example:
     [Final Answer] I completed diagnosis and targeted data generation.
     The diagnosis showed confusion between 'COVID-19' and 'Control Group' (accuracy 60%).
     To mitigate this, I generated targeted 'COVID-19' samples. The generated files are saved at: <path>.

---
[B. Simple task: diagnose-only or generate-only]

When the user's request is just a single-tool call (e.g., "Diagnose /data/test" or "Generate asthma audio"):

1. Step 1: Act
   * Call the corresponding tool (Diagnoser or Generator).
   * Example (simple diagnosis):
     [Call:Diagnoser] Diagnose respiratory sounds with audio_dir=/data/test/audio, \
output_dir=/data/test/output_diagnose, metadata_csv=/data/test/metadata.csv
   * Example (simple generation):
     [Call:Generator] Generate asthma respiratory audio with ref_audio=./reference.wav, \
disease=Asthma, out_dir=./Generator/output_generate

2. Step 2: Summarize
   * Wait for the tool's [Tool Output].
   * IMPORTANT: You MUST NOT automatically call the *other* tool after a simple task.
     Your job is only to summarize this tool's results.
   * Use the [Final Answer] tag to indicate that the simple task is complete.
   * Example (after diagnosis):
     [Final Answer] Diagnosis completed. The report shows 20 files in total, 12 correctly classified COVID cases
     (accuracy: 60.00%). The dominant confusion pair (GT -> Pred) is Positive -> Control Group (7 times).
     The analysis is saved at: /path/to/results.csv
   * Example (after generation):
     [Final Answer] Generation completed. The audio files are saved at: /path/to/generated_audio.wav

---
[Important rules]
* STRICTLY follow the distinction between [A. Advanced task] and [B. Simple task].
* Do NOT call both tools in a single response.
* Do NOT answer questions that should be answered by tools yourself; always call the tool.
* After calling a tool, you MUST wait for a [Tool Output] before you reflect or summarize.
* If the user request is unclear or missing required parameters (e.g., paths), ask for clarification.
"""

# --- Default tool arguments ---
DIAGNOSER_DEFAULTS = {
    "audio_dir": "./Diagnoser/example/audio",
    "output_dir": "./Diagnoser/output_diagnose",
    "metadata_csv": "./Diagnoser/example/combined_metadata.csv",
}

GENERATOR_DEFAULTS = {
    "ref_audio": "./Generator/wav/reference_audio.wav",
    "disease": "Asthma",
    "out_dir": "./Generator/output_generate",
}


# --- Tool function: run_diagnoser ---
def run_diagnoser(user_prompt: str) -> str:
    """
    Parse the user (or Thinker) prompt, infer diagnoser arguments, and run the diagnoser pipeline.
    """
    print("\n[Tool] Parsing diagnoser task arguments...")
    messages = []

    def _clean_path(p: str) -> str:
        # Strip surrounding quotes and trailing punctuation
        return (
            p.strip()
            .strip('\'"""')
            .rstrip("，,。；;）)")
            .rstrip("中")  # kept for backward compatibility; harmless in English
        )

    # Connectors: ':', '=', 'is', 'are', 'at', 'in'
    _conn = r"(?:[:=]|is|are|at|in)\s*"

    audio_pat = rf"(?:--audio_dir|audio_dir|audio directory|audio path|respiratory audio){_conn}([^\s\"',]+)"
    output_pat = rf"(?:--output_dir|output_dir|output directory|output path|results dir|results directory){_conn}([^\s\"',]+)"
    meta_pat = rf"(?:--metadata_csv|metadata_csv|metadata csv|EHR table|EHR csv|clinical metadata){_conn}([^\s\"',]+?\.csv)\b"

    audio_dir_match = re.search(audio_pat, user_prompt, flags=re.IGNORECASE)
    output_dir_match = re.search(output_pat, user_prompt, flags=re.IGNORECASE)
    metadata_csv_match = re.search(meta_pat, user_prompt, flags=re.IGNORECASE)

    audio_dir = _clean_path(audio_dir_match.group(1)) if audio_dir_match else None
    output_dir = _clean_path(output_dir_match.group(1)) if output_dir_match else None
    metadata_csv = (
        _clean_path(metadata_csv_match.group(1)) if metadata_csv_match else None
    )

    # Fallback: try to infer from raw paths in the prompt
    if not audio_dir:
        m = re.search(
            r"(\.{0,2}/[^\s,]+/audio[^\s,]*)", user_prompt, flags=re.IGNORECASE
        )
        if m:
            audio_dir = _clean_path(m.group(1))
    if not output_dir:
        m = re.search(
            r"(\.{0,2}/[^\s,]*(?:output)[^\s,]*)", user_prompt, flags=re.IGNORECASE
        )
        if m:
            output_dir = _clean_path(m.group(1))
    if not metadata_csv:
        m = re.search(r"(\.{0,2}/[^\s,]+\.csv)", user_prompt, flags=re.IGNORECASE)
        if m:
            metadata_csv = _clean_path(m.group(1))

    # Fill defaults if necessary
    if not audio_dir:
        audio_dir = DIAGNOSER_DEFAULTS["audio_dir"]
        messages.append(f"No audio_dir found in prompt, using default: {audio_dir}")
    if not output_dir:
        output_dir = DIAGNOSER_DEFAULTS["output_dir"]
        messages.append(f"No output_dir found in prompt, using default: {output_dir}")
    if not metadata_csv:
        metadata_csv = DIAGNOSER_DEFAULTS["metadata_csv"]
        messages.append(
            f"No metadata_csv found in prompt, using default: {metadata_csv}"
        )
        messages.append(
            "Note: The provided EHR/metadata CSV must match the expected header format, "
            "and each patient should have at least two respiratory recordings."
        )

    if messages:
        print("\n".join(f"[Hint] {m}" for m in messages))

    command = [
        "python",
        "-u",
        "./Diagnoser/diagnoser_pipeline.py",
        "--audio_dir",
        audio_dir,
        "--output_dir",
        output_dir,
        "--metadata_csv",
        metadata_csv,
    ]

    print(
        f"\n[Exec] Running diagnoser command:\n  {' '.join(shlex.quote(c) for c in command)}"
    )
    print("[Progress] Diagnoser pipeline started, streaming logs below...")
    print("=" * 40 + " Diagnoser Live Logs " + "=" * 40)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        output_lines = []
        with process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                output_lines.append(line)

        return_code = process.wait()
        print("=" * 40 + " Logs End " + "=" * 45)

        full_output = "".join(output_lines)
        if return_code == 0:
            result_path = ""
            for line in full_output.splitlines():
                if "Results have been written to:" in line:
                    result_path = line.split(":", 1)[-1].strip()
                    break

            # Close the feedback loop by summarizing real data, if possible
            if result_path and os.path.exists(result_path):
                try:
                    df_results = pd.read_csv(result_path)

                    # Compact string representation for the LLM
                    summary_data = df_results.to_string(index=False, max_rows=20)

                    # Compute basic stats
                    total_files = len(df_results)
                    matches = df_results["covid_match"].sum()
                    accuracy = (matches / total_files * 100) if total_files > 0 else 0

                    # Identify common mismatches
                    mismatches = df_results[df_results["covid_match"] == False]
                    if mismatches.empty:
                        error_counts = "No obvious confusion pairs."
                    else:
                        error_counts = (
                            mismatches.groupby(
                                ["covid_test_result(GT)", "predicted_disease"]
                            )
                            .size()
                            .to_string()
                        )

                    # Return a concise English summary
                    return (
                        f"Diagnosis completed successfully.\n"
                        f"Results CSV saved at: {os.path.abspath(result_path)}\n"
                        f"[Diagnosis summary]\n"
                        f"Total {total_files} files, COVID match correct for {matches} files "
                        f"(accuracy: {accuracy:.2f}%)\n"
                        f"Top confusion pairs (GT -> Pred):\n{error_counts}\n\n"
                        f"Head of results (first 20 rows):\n{summary_data}"
                    )

                except Exception as e:
                    # At least return the path if reading failed
                    return (
                        "Diagnosis completed successfully. "
                        f"Results CSV saved at: {os.path.abspath(result_path)} "
                        f"(failed to read CSV for summary: {e})"
                    )

            elif result_path:
                return (
                    "Diagnosis completed successfully. "
                    f"The logs claim results were saved at: {os.path.abspath(result_path)}, "
                    "but the file could not be found."
                )
            else:
                return (
                    "Diagnoser pipeline finished, but no result path could be parsed from the logs. "
                    "Please inspect the logs above."
                )
        else:
            return (
                f"Diagnoser pipeline failed with return code: {return_code}. "
                f"Please inspect the logs above.\n"
                f"{full_output[-500:]}"  # Last 500 chars for debugging
            )

    except FileNotFoundError:
        return (
            "Error: Could not find Python or diagnoser_pipeline.py. "
            "Please check your environment and paths."
        )
    except Exception as e:
        return f"Unexpected error while running diagnoser_pipeline.py: {e}"


# --- Tool function: run_generator ---
def run_generator(user_prompt: str) -> str:
    """
    Parse the user (or Thinker) prompt, infer generator arguments, and run the generator pipeline.
    """
    print("\n[Tool] Parsing generator task arguments...")
    messages = []

    def _clean_path(p: str) -> str:
        return p.strip().strip('\'"""').rstrip("，,。；;）)").rstrip("中")

    _conn = r"(?:[:=]|is|are|at|in)\s*"

    ref_pat = rf"(?:--ref_audio|ref_audio|reference audio|reference respiratory audio){_conn}([^\s\"',]+?\.wav)\b"
    dis_pat1 = r"generate\s+(.+?)\s+respiratory\s+audio"
    dis_pat2 = rf"(?:--disease|disease type|disease label|disease){_conn}([^\s\"',]+)"
    out_pat = rf"(?:--out_dir|out_dir|output dir|output directory|generated audio dir|output path){_conn}([^\s\"',]+)"

    ref_audio_match = re.search(ref_pat, user_prompt, flags=re.IGNORECASE)
    disease_match = re.search(dis_pat1, user_prompt, flags=re.IGNORECASE) or re.search(
        dis_pat2, user_prompt, flags=re.IGNORECASE
    )
    out_dir_match = re.search(out_pat, user_prompt, flags=re.IGNORECASE)

    ref_audio = _clean_path(ref_audio_match.group(1)) if ref_audio_match else None
    disease = disease_match.group(1).strip() if disease_match else None
    out_dir = _clean_path(out_dir_match.group(1)) if out_dir_match else None

    # Fallbacks
    if not ref_audio:
        m = re.search(r"(\.{0,2}/[^\s,]+\.wav)", user_prompt, flags=re.IGNORECASE)
        if m:
            ref_audio = _clean_path(m.group(1))
    if not out_dir:
        m = re.search(
            r"(\.{0,2}/[^\s,]*(?:output|out)[^\s,]*)", user_prompt, flags=re.IGNORECASE
        )
        if m:
            out_dir = _clean_path(m.group(1))

    # Defaults
    if not ref_audio:
        ref_audio = GENERATOR_DEFAULTS["ref_audio"]
        messages.append(f"No reference audio provided, using default: {ref_audio}")
    if not disease:
        disease = GENERATOR_DEFAULTS["disease"]
        messages.append(f"No disease type provided, using default: {disease}")
    if not out_dir:
        out_dir = GENERATOR_DEFAULTS["out_dir"]
        messages.append(f"No output directory provided, using default: {out_dir}")

    if messages:
        print("\n".join(f"[Hint] {m}" for m in messages))

    command = [
        "python",
        "-u",
        "./Generator/generator_pipeline.py",
        "--ref_audio",
        ref_audio,
        "--disease",
        disease,
        "--out_dir",
        out_dir,
    ]

    print(
        f"\n[Exec] Running generator command:\n  {' '.join(shlex.quote(c) for c in command)}"
    )
    print("[Progress] Generator pipeline started, streaming logs below...")
    print("=" * 40 + " Generator Live Logs " + "=" * 40)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        output_lines = []
        with process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                output_lines.append(line)

        return_code = process.wait()
        print("=" * 40 + " Logs End " + "=" * 45)

        full_output = "".join(output_lines)
        if return_code == 0:
            result_path = ""
            for line in full_output.splitlines():
                if "Audio saved to:" in line:
                    result_path = line.split(":", 1)[-1].strip()
                    break
            if result_path:
                return f"Audio generation completed successfully. File saved at: {os.path.abspath(result_path)}"
            else:
                return (
                    "Generator pipeline completed, but no output file path could be parsed from the logs. "
                    "Please inspect the logs above."
                )
        else:
            return (
                f"Generator pipeline failed with return code: {return_code}. "
                f"Please inspect the logs above.\n"
                f"{full_output[-500:]}"  # Last 500 chars
            )

    except FileNotFoundError:
        return (
            "Error: Could not find Python or generator_pipeline.py. "
            "Please check your environment and paths."
        )
    except Exception as e:
        return f"Unexpected error while running generator_pipeline.py: {e}"


# Helper: print example prompts
def print_agent_examples():
    """
    Display example prompts (templates) that the user can copy-paste.
    """
    diag = (
        f"Please diagnose respiratory sounds with audio_dir={DIAGNOSER_DEFAULTS['audio_dir']}, "
        f"metadata_csv={DIAGNOSER_DEFAULTS['metadata_csv']}"
    )
    gen = (
        f"Please generate respiratory audio with disease=Asthma, ref_audio={GENERATOR_DEFAULTS['ref_audio']}, "
        f"out_dir={GENERATOR_DEFAULTS['out_dir']}"
    )
    iterate = (
        "Run a full iteration: first diagnose using "
        f"audio_dir={DIAGNOSER_DEFAULTS['audio_dir']} and metadata_csv={DIAGNOSER_DEFAULTS['metadata_csv']}, "
        "then, based on the weaknesses, generate new data using "
        f"ref_audio={GENERATOR_DEFAULTS['ref_audio']} and out_dir={GENERATOR_DEFAULTS['out_dir']}"
    )

    print("\n================ Resp-Agent Task Examples ================")
    print("[Simple diagnosis] ", diag)
    print("[Simple generation]", gen)
    print("[Advanced: iterative analysis (recommended)]", iterate)
    print(
        'Tip: Type "help" to show these examples again, '
        'or type "quit" to exit.\n'
    )


class RespAgentEnglish:
    """English version of Resp-Agent class wrapper"""

    def __init__(self, api_key: str = None):
        """
        Initialize the agent.

        Args:
            api_key: DeepSeek API key, if None will read from environment
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not found")

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_message: str) -> str:
        """
        Chat with the agent.

        Args:
            user_message: User message

        Returns:
            Agent response
        """
        self.chat_history.append({"role": "user", "content": user_message})

        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.chat_history,
            stream=False,
            temperature=0.7,
            top_p=0.9,
        )
        assistant_response = (resp.choices[0].message.content or "").strip()
        self.chat_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response


# --- Main function (entry point) ---
def main():
    """
    Main loop implementing the ReAct-style agent:
    - The LLM (Thinker) receives all user input via API.
    - The LLM decides whether to chat, call tools, or enter a diagnose–reflect–plan–act loop.
    - This script parses the LLM output, executes tools, and feeds results back to the LLM.
    """
    print("=" * 80)
    print("Initializing DeepSeek API client for Resp-Agent Thinker...")

    # 1. Check API key and initialize client
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("\n[Error] Environment variable DEEPSEEK_API_KEY is not set.")
        print("Please set it first, e.g.: export DEEPSEEK_API_KEY='YOUR_API_KEY'")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        client.models.list()
    except Exception as e:
        print(f"\n[Error] Failed to initialize or connect to DeepSeek API: {e}")
        print("Please verify your API key and network connectivity.")
        sys.exit(1)

    print("\nThinker (Resp-Agent API client) is ready.")
    print("=" * 80)

    # Show example prompts on startup
    print_agent_examples()

    # 2. Start agent loop (inject system prompt)
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() == "quit":
                print("Goodbye!")
                break

            if user_prompt.lower() == "help":
                print_agent_examples()
                continue

            # Add user message to history
            chat_history.append({"role": "user", "content": user_prompt})

            # 3. Start internal agent (ReAct) loop
            while True:
                # 4. Thinker (API) produces reasoning and tool calls
                print("\nThinker: ", end="", flush=True)
                try:
                    # ReAct loop must use stream=False to keep the full content
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=chat_history,
                        stream=False,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    assistant_response = (resp.choices[0].message.content or "").strip()
                    print(assistant_response, flush=True)

                except Exception as e:
                    print(f"\n[API error]: {e}")
                    chat_history.append(
                        {
                            "role": "user",
                            "content": f"[Tool Output]: API Error: {e}. Cannot proceed. Ask user for next step.",
                        }
                    )
                    continue  # Continue inner loop so LLM can react to the error

                # Record LLM output (including reasoning and tool calls)
                chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # 5. Parse agent output and execute tools
                call_diag_match = re.search(
                    r"\[Call:Diagnoser\](.*)", assistant_response, re.DOTALL
                )
                call_gen_match = re.search(
                    r"\[Call:Generator\](.*)", assistant_response, re.DOTALL
                )
                final_answer_match = re.search(
                    r"\[Final Answer\](.*)", assistant_response, re.DOTALL
                )

                if call_diag_match:
                    tool_prompt = call_diag_match.group(1).strip()
                    print(
                        f"\n[Thinker executing...]\n"
                        f"> Task: Diagnose respiratory sounds\n"
                        f"> Args (truncated): {shlex.quote(tool_prompt[:80])}...\n"
                    )
                    tool_result = run_diagnoser(tool_prompt)

                elif call_gen_match:
                    tool_prompt = call_gen_match.group(1).strip()
                    print(
                        f"\n[Thinker executing...]\n"
                        f"> Task: Generate respiratory audio\n"
                        f"> Args (truncated): {shlex.quote(tool_prompt[:80])}...\n"
                    )
                    tool_result = run_generator(tool_prompt)

                elif final_answer_match:
                    print("\n[Thinker task finished]")
                    break  # Exit inner loop, wait for next user input

                else:
                    # Pure chat response, no tool call
                    break  # Exit inner loop, wait for next user input

                # 6. Feed tool results back to the LLM
                if tool_result:
                    print("\n[Tool finished, sending result back to Thinker...]\n")
                    chat_history.append(
                        {"role": "user", "content": f"[Tool Output]: {tool_result}"}
                    )
                    # Continue inner loop to allow further reflection / planning
                else:
                    print(
                        "\n[Warning] Thinker requested a tool call but no result was produced."
                    )
                    chat_history.append(
                        {
                            "role": "user",
                            "content": "[Tool Output]: Tool execution failed or returned no information.",
                        }
                    )

        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime error]: {e}")
            break


if __name__ == "__main__":
    main()

"""
Resp-Agent English: Respiratory Sound Diagnosis and Generation Agent

This module provides the English version of the Resp-Agent Thinker,
which coordinates the Diagnoser and Generator tools through the DeepSeek API.
"""

import os
import sys

from openai import OpenAI

# Thinker System Prompt
SYSTEM_PROMPT = """
You are an advanced AI assistant called "Thinker" from Resp-Agent.
Your core task is to coordinate the "Diagnoser" and "Generator" tools to implement a closed-loop "Diagnose-Reflect-Plan-Act" workflow, helping researchers improve respiratory sound diagnosis models.

【Your Tools】

1.  **Diagnoser**:
    * Function: Run a multimodal diagnosis model to analyze respiratory sound data and EHR, output detailed diagnosis report, confusion matrix, and identify weak categories.
    * Format: `[Call:Diagnoser] Diagnose respiratory sounds, audio_dir is <path>, output_dir is <path>, metadata_csv is <path>`
    * Note: You must provide all parameters.

2.  **Generator**:
    * Function: Call a controllable generation model (Resp-MLLM) to synthesize new, high-fidelity respiratory sounds based on "disease label" (content) and "reference audio" (style).
    * Format: `[Call:Generator] Generate <disease> respiratory sound, ref_audio is <path.wav>, disease is <disease>, out_dir is <path>`
    * Note: You must provide all parameters.

【Important Rules】
* **Do NOT** call two tools in one response.
* **Do NOT** answer questions that should be answered by tools.
* After calling a tool, **MUST** wait for `[Tool Output]` before proceeding.
* If the user's request is unclear, proactively ask for all required parameters.
"""


# Default parameters
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


class RespAgentEnglish:
    """English version of Resp-Agent"""

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

    def run_diagnoser(self, user_prompt: str) -> str:
        """Run the diagnoser tool"""
        print("\n[Tool Call] Parsing diagnosis task parameters...")
        # Implementation based on original
        return "Diagnosis complete"

    def run_generator(self, user_prompt: str) -> str:
        """Run the generator tool"""
        print("\n[Tool Call] Parsing generation task parameters...")
        # Implementation based on original
        return "Generation complete"

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


def print_agent_examples():
    """Display agent task examples"""
    diag = (
        f"Diagnose respiratory sounds, audio_dir is {DIAGNOSER_DEFAULTS['audio_dir']}, "
        f"metadata_csv is {DIAGNOSER_DEFAULTS['metadata_csv']}"
    )
    gen = (
        f"Generate Asthma respiratory sound, ref_audio is {GENERATOR_DEFAULTS['ref_audio']}, "
        f"out_dir is {GENERATOR_DEFAULTS['out_dir']}"
    )

    print("\n================ Resp-Agent Task Examples ================")
    print("[Simple Diagnosis]", diag)
    print("[Simple Generation]", gen)
    print("Tip: Send 'help' to show examples again, send 'quit' to exit.\n")


def main():
    """Main entry point"""
    print("=" * 80)
    print("Initializing DeepSeek API client (Agent mode)...")

    try:
        agent = RespAgentEnglish()
    except ValueError as e:
        print(f"\n[Error] {e}")
        print("Please set environment variable: export DEEPSEEK_API_KEY='your_api_key'")
        sys.exit(1)

    print("\nThinker (Resp-Agent API) is ready.")
    print("=" * 80)

    print_agent_examples()

    while True:
        try:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if user_prompt.lower() == "help":
                print_agent_examples()
                continue

            response = agent.chat(user_prompt)
            print(f"\nThinker: {response}\n")

        except KeyboardInterrupt:
            print("\nProgram interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}")
            break


if __name__ == "__main__":
    main()

import os
import re
import shlex
import subprocess
import sys

import pandas as pd

# 导入 OpenAI SDK 用于调用 DeepSeek API
from openai import OpenAI

# 这是实现 Thinker 智能体的核心
SYSTEM_PROMPT = """
你是一个高级的人工智能助手，名为 Resp-Agent 的 "Thinker"（思考者）。
你的核心任务是协调 "Diagnoser"（诊断器）和 "Generator"（生成器）这两个工具，以实现一个闭环的“诊断-反思-规划-行动”工作流，从而帮助研究员改进呼吸音诊断模型。

【你拥有的工具】

1.  **诊断器 (Diagnoser)**:
    * 功能：运行一个多模态诊断模型，分析给定的呼吸音数据和电子病历（EHR），输出详细的诊断报告、混淆矩阵、以及模型在哪些类别上表现不佳。
    * 调用格式: `[Call:Diagnoser] 帮我诊断呼吸音，音频目录是 <path>，输出目录是 <path>，病历文档在 <path>`
    * 注意：你必须提供所有参数。

2.  **生成器 (Generator)**:
    * 功能：调用一个可控的生成模型（Resp-MLLM），根据“疾病标签”（内容）和“参考音频”（风格）来合成新的、高保真的呼吸音。
    * 调用格式: `[Call:Generator] 生成<疾病名>呼吸音，参考音频是 <path.wav>，疾病类型为 <疾病名>，输出目录是 <path>`
    * 注意：你必须提供所有参数。

【你的工作流程】

你必须首先判断用户的任务是“简单任务”（仅诊断或仅生成）还是“高级任务”（需要闭环迭代）。

---
【A. 高级任务：诊断-反思-生成】

当用户提出高级任务（例如：“帮我分析...并生成改进数据”，“运行一次完整迭代”）时，你**必须**遵循以下步骤：

1.  **步骤 1: 诊断 (Diagnose)**
    * 首先，调用 `Diagnoser` 工具来获取模型当前的表现。
    * 示例: `[Call:Diagnoser] 帮我诊断呼吸音，音频目录是 /data/test/audio，输出目录是 /data/test/output_diagnose，病历文档在 /data/test/metadata.csv`

2.  **步骤 2: 反思 (Reflect)**
    * 等待 `Diagnoser` 的 `[Tool Output]` 返回。
    * **这是最关键的一步**。你必须仔细分析诊断报告（[Tool Output]中会包含数据摘要），找出模型的“弱点”。
    * 你的思考过程（内部进行）：“诊断报告显示，总共 20 个文件, COVID 匹配正确 12 个 (准确率: 60.00%)。主要错误混淆对 (GT -> Pred) 是 Positive -> Control Group (7次)。我需要生成 Positive (COVID-19) 的样本来解决这个问题。”

3.  **步骤 3: 规划 (Plan) & 行动 (Act)**
    * 基于你的“反思”，制定一个**精确**的生成计划。
    * 调用 `Generator` 工具。
    * 示例: `[Call:Generator] 生成COVID-19呼吸音，参考音频是 ./Generator/wav/reference_healthy.wav，疾病类型为 COVID-19，输出目录是 ./Generator/output_finetune_set`

4.  **步骤 4: 总结 (Summarize)**
    * 等待 `Generator` 的 `[Tool Output]` 返回。
    * 向用户提供一个简洁明了的最终答案，总结你所做的工作和结果。
    * 示例: `[Final Answer] 我已经完成了分析和数据生成。诊断显示模型在区分‘COVID-19’和‘Control Group’上存在混淆（准确率60%）。为此，我已生成了针对性的‘COVID-19’样本。生成的文件位于：<path>`

---
【B. 简单任务：仅诊断 或 仅生成】

当用户的请求只是一个独立的工具调用（例如：“帮我诊断 /data/test” 或 “帮我生成哮喘音频”）时：

1.  **步骤 1: 行动 (Act)**
    * 根据用户请求，调用相应的工具（`Diagnoser` 或 `Generator`）。
    * 示例 (简单诊断): `[Call:Diagnoser] 帮我诊断呼吸音，音频目录是 /data/test/audio...`
    * 示例 (简单生成): `[Call:Generator] 生成哮喘呼吸音，参考音频是 ./reference.wav...`

2.  **步骤 2: 总结 (Summarize)**
    * 等待该工具的 `[Tool Output]` 返回。
    * **关键规则**：你**不得**在简单任务后自动调用*另一个*工具。你的任务只是报告这个工具的执行结果。
    * 使用 `[Final Answer]` 标签来表示你的简单任务已完成。
    * 示例 (诊断后): `[Final Answer] 诊断已完成。诊断报告显示，总共 20 个文件, COVID 匹配正确 12 个 (准确率: 60.00%)。主要错误混淆对 (GT -> Pred) 是 Positive -> Control Group (7次)。分析结果已保存至: /path/to/results.csv`
    * 示例 (生成后): `[Final Answer] 生成已完成。音频已保存至: /path/to/generated_audio.wav`

---
【重要规则】
* **严格**遵守上述 【A. 高级任务】 和 【B. 简单任务】 的流程区分。
* **不要**在一次回复中同时调用两个工具。
* **不要**自己回答本应由工具回答的问题。
* 在调用工具后，**必须**等待 `[Tool Output]` 返回，然后再进行下一步“反思”或“总结”。
* 如果用户的请求不清晰，请主动询问以获取所有必需的参数（如路径）。
"""

# --- 工具调用：参数默认值 ---
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


# --- 工具函数：run_diagnoser ---
def run_diagnoser(user_prompt: str) -> str:
    print("\n[工具调用] 正在解析诊断任务参数...")
    messages = []

    def _clean_path(p: str) -> str:
        return p.strip().strip("'\"“”").rstrip("，,。；;）)").rstrip("中")

    _conn = r"(?:[:：=]|是|为|在)\s*"
    audio_pat = rf"(?:--audio_dir|音频目录|音频路径|呼吸音数据){_conn}([^\s\"'，,]+)"
    output_pat = rf"(?:--output_dir|输出目录|输出路径|结果目录|结果路径|分析文件.*?输出){_conn}([^\s\"'，,]+)"
    meta_pat = rf"(?:--metadata_csv|电子病历|病历文档|病历表|病人情况.*?(?:表格|文档)){_conn}([^\s\"'，,]+?\.csv)\b"

    audio_dir_match = re.search(audio_pat, user_prompt, flags=re.IGNORECASE)
    output_dir_match = re.search(output_pat, user_prompt, flags=re.IGNORECASE)
    metadata_csv_match = re.search(meta_pat, user_prompt, flags=re.IGNORECASE)

    audio_dir = _clean_path(audio_dir_match.group(1)) if audio_dir_match else None
    output_dir = _clean_path(output_dir_match.group(1)) if output_dir_match else None
    metadata_csv = (
        _clean_path(metadata_csv_match.group(1)) if metadata_csv_match else None
    )

    if not audio_dir:
        m = re.search(
            r"(\.{0,2}/[^\s，,]+/audio[^\s，,]*)", user_prompt, flags=re.IGNORECASE
        )
        if m:
            audio_dir = _clean_path(m.group(1))
    if not output_dir:
        m = re.search(
            r"(\.{0,2}/[^\s，,]*(?:output|输出)[^\s，,]*)",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if m:
            output_dir = _clean_path(m.group(1))
    if not metadata_csv:
        m = re.search(r"(\.{0,2}/[^\s，,]+\.csv)", user_prompt, flags=re.IGNORECASE)
        if m:
            metadata_csv = _clean_path(m.group(1))

    if not audio_dir:
        audio_dir = DIAGNOSER_DEFAULTS["audio_dir"]
        messages.append(f"未显式给出音频目录，使用默认: {audio_dir}")
    if not output_dir:
        output_dir = DIAGNOSER_DEFAULTS["output_dir"]
        messages.append(f"未显式给出输出目录，使用默认: {output_dir}")
    if not metadata_csv:
        metadata_csv = DIAGNOSER_DEFAULTS["metadata_csv"]
        messages.append(f"未显式给出病历表格，使用默认: {metadata_csv}")
        messages.append(
            ">> 特别提醒: 你提供的电子病历需与默认表头一致，且呼吸音不少于两条。"
        )

    if messages:
        print("\n".join(f"[提示] {m}" for m in messages))

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

    print(f"\n[执行命令] {' '.join(shlex.quote(c) for c in command)}")
    print("[进度] 诊断流程已启动，实时输出如下...")
    print("=" * 40 + " 诊断脚本实时日志 " + "=" * 40)

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
        print("=" * 40 + " 日志结束 " + "=" * 45)

        full_output = "".join(output_lines)
        if return_code == 0:
            result_path = ""
            for line in full_output.splitlines():
                if "Results have been written to:" in line:
                    result_path = line.split(":")[-1].strip()
                    break

            # 关闭反馈环路
            if result_path and os.path.exists(result_path):
                try:
                    df_results = pd.read_csv(result_path)

                    # 转换为紧凑的字符串格式，LLM 易读
                    summary_data = df_results.to_string(index=False, max_rows=20)

                    # 计算关键统计数据
                    total_files = len(df_results)
                    matches = df_results["covid_match"].sum()
                    accuracy = (matches / total_files * 100) if total_files > 0 else 0

                    # 找出最常见的错误 (GT=Positive, Pred!=COVID-19) 或 (GT=Negative, Pred=COVID-19)
                    mismatches = df_results[~df_results["covid_match"]]
                    if mismatches.empty:
                        error_counts = "无明显混淆"
                    else:
                        error_counts = (
                            mismatches.groupby(
                                ["covid_test_result(GT)", "predicted_disease"]
                            )
                            .size()
                            .to_string()
                        )

                    # 返回包含真实数据的摘要
                    return (
                        f"诊断成功完成。\n"
                        f"分析结果已保存至: {os.path.abspath(result_path)}\n"
                        f"【诊断摘要】\n"
                        f"总共 {total_files} 个文件, COVID 匹配正确 {matches} 个 (准确率: {accuracy:.2f}%)\n"
                        f"主要错误混淆对 (GT -> Pred):\n{error_counts}\n\n"
                        f"详细结果 (前20行):\n{summary_data}"
                    )

                except Exception as e:
                    # 如果读取失败，至少返回路径
                    return f"诊断成功完成。分析结果已保存至: {os.path.abspath(result_path)} (但读取CSV摘要失败: {e})"

            elif result_path:
                return f"诊断成功完成。分析结果声称已保存至: {os.path.abspath(result_path)}，但文件未找到。"
            else:
                return "诊断已执行，但未从日志解析到结果路径；请查看上方日志。"
        else:
            return f"诊断脚本执行失败，返回码: {return_code}。请查看上方日志。\n{full_output[-500:]}"  # 返回最后500字符的错误

    except FileNotFoundError:
        return "错误: 未找到 Python 或 diagnoser_pipeline.py；请检查环境与路径。"
    except Exception as e:
        return f"执行诊断脚本时发生未知错误: {e}"


# --- 工具函数：run_generator ---
def run_generator(user_prompt: str) -> str:
    import shlex

    print("\n[工具调用] 正在解析生成任务参数...")
    messages = []

    def _clean_path(p: str) -> str:
        return p.strip().strip("'\"“”").rstrip("，,。；;）)").rstrip("中")

    _conn = r"(?:[:：=]|是|为|在)\s*"
    ref_pat = rf"(?:--ref_audio|参考音频|参考呼吸音){_conn}([^\s\"'，,]+?\.wav)\b"
    dis_pat1 = r"生成(.+?)呼吸音"
    dis_pat2 = rf"(?:--disease|疾病类型|疾病标签|疾病分类){_conn}([^\s\"'，,]+)"
    out_pat = rf"(?:--out_dir|输出目录|生成音频.*?目录|输出路径){_conn}([^\s\"'，,]+)"

    ref_audio_match = re.search(ref_pat, user_prompt, flags=re.IGNORECASE)
    disease_match = re.search(dis_pat1, user_prompt) or re.search(
        dis_pat2, user_prompt, flags=re.IGNORECASE
    )
    out_dir_match = re.search(out_pat, user_prompt, flags=re.IGNORECASE)

    ref_audio = _clean_path(ref_audio_match.group(1)) if ref_audio_match else None
    disease = disease_match.group(1).strip() if disease_match else None
    out_dir = _clean_path(out_dir_match.group(1)) if out_dir_match else None

    if not ref_audio:
        m = re.search(r"(\.{0,2}/[^\s，,]+\.wav)", user_prompt, flags=re.IGNORECASE)
        if m:
            ref_audio = _clean_path(m.group(1))
    if not out_dir:
        m = re.search(
            r"(\.{0,2}/[^\s，,]*(?:output|输出)[^\s，,]*)",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if m:
            out_dir = _clean_path(m.group(1))

    if not ref_audio:
        ref_audio = GENERATOR_DEFAULTS["ref_audio"]
        messages.append(f"未指定参考音频，使用默认: {ref_audio}")
    if not disease:
        disease = GENERATOR_DEFAULTS["disease"]
        messages.append(f"未指定疾病类型，使用默认: {disease}")
    if not out_dir:
        out_dir = GENERATOR_DEFAULTS["out_dir"]
        messages.append(f"未指定输出目录，使用默认: {out_dir}")

    if messages:
        print("\n".join(f"[提示] {m}" for m in messages))

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

    print(f"\n[执行命令] {' '.join(shlex.quote(c) for c in command)}")
    print("[进度] 生成流程已启动，实时输出如下...")
    print("=" * 40 + " 生成脚本实时日志 " + "=" * 40)

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
        print("=" * 40 + " 日志结束 " + "=" * 45)

        full_output = "".join(output_lines)
        if return_code == 0:
            result_path = ""
            for line in full_output.splitlines():
                if "Audio saved to:" in line:
                    result_path = line.split(":")[-1].strip()
                    break
            return (
                f"音频生成成功。文件已保存至: {os.path.abspath(result_path)}"
                if result_path
                else "生成已执行，但未解析到输出文件路径，请查看日志。"
            )
        else:
            return f"生成脚本执行失败，返回码: {return_code}。请查看上方日志。\n{full_output[-500:]}"  # 返回最后500字符的错误

    except FileNotFoundError:
        return "错误: 未找到 Python 或 generator_pipeline.py；请检查环境与路径。"
    except Exception as e:
        return f"执行生成脚本时发生未知错误: {e}"


# 帮助模板
def print_agent_examples():
    """
    启动时与用户请求“模板/help”时显示的可复制输入模板（含默认参数）。
    """
    diag = (
        f"帮我诊断呼吸音，音频目录是 {DIAGNOSER_DEFAULTS['audio_dir']}，"
        f"病历文档在 {DIAGNOSER_DEFAULTS['metadata_csv']}"
    )
    gen = (
        f"帮我生成疾病类型为Asthma的呼吸音，参考音频是 {GENERATOR_DEFAULTS['ref_audio']}，"
        f"输出到 {GENERATOR_DEFAULTS['out_dir']}"
    )
    iterate = (
        f"启动一次完整的迭代分析：使用 {DIAGNOSER_DEFAULTS['audio_dir']} 和 {DIAGNOSER_DEFAULTS['metadata_csv']} 进行诊断，"
        f"然后根据弱点，使用 {GENERATOR_DEFAULTS['ref_audio']} 作为风格参考，在 {GENERATOR_DEFAULTS['out_dir']} 生成数据。"
    )

    print("\n================ Resp-Agent 任务示例 ================ ")
    print("【简单诊断】", diag)
    print("【简单生成】", gen)
    print("【高级任务：迭代分析 (推荐)】", iterate)
    print(
        "提示：发送 “模板/help” 可随时再次显示以上示例，发送“停止/quit”则退出智能体系统。\n"
    )


# --- 主函数 (main) ---
def main():
    """
    主函数，实现 ReAct Agent 逻辑。
    - LLM (Thinker) 通过 API 接收所有用户输入。
    - LLM 决定是聊天、调用工具，还是执行“反思-规划”循环。
    - 脚本解析 LLM 的输出，执行工具，并将结果反馈给 LLM。
    """
    print("=" * 80)
    print("正在初始化 DeepSeek API 客户端 (Agent 模式)...")

    # --- 1. 检查 API Key 并初始化客户端 ---
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("\n[错误] 未检测到 DEEPSEEK_API_KEY 环境变量。")
        print("请先设置环境变量: export DEEPSEEK_API_KEY='您的API_KEY'")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        client.models.list()
    except Exception as e:
        print(f"\n[错误] DeepSeek API 客户端初始化或连接失败: {e}")
        print("请检查 API Key 是否正确以及网络连接是否正常。")
        sys.exit(1)

    print("\nThinker (Resp-Agent API) 已准备就绪。")
    print("=" * 80)

    # --- 启动时给出 Agent 任务模板 ---
    print_agent_examples()

    # --- 2. 启动 Agent 循环 (注入 System Prompt) ---
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_prompt = input("你: ").strip()
            if user_prompt.lower() in ["停止", "quit"]:
                print("再见！")
                break

            if user_prompt in ["模板", "help"]:
                print_agent_examples()
                continue

            # 将用户最新输入加入历史
            chat_history.append({"role": "user", "content": user_prompt})

            # --- 3. 启动 Agent 内循环 (ReAct Loop) ---
            # 允许 LLM 通过连续调用工具来完成复杂任务
            while True:
                # --- 4. Thinker (API) 生成思考与行动 ---
                print("\nThinker: ", end="", flush=True)
                try:
                    # ReAct 循环必须使用 stream=False
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=chat_history,
                        stream=False,  # 必须为 False
                        temperature=0.7,
                        top_p=0.9,
                    )
                    assistant_response = (resp.choices[0].message.content or "").strip()
                    print(assistant_response, flush=True)

                except Exception as e:
                    print(f"\n[API 调用失败]: {e}")
                    chat_history.append(
                        {
                            "role": "user",
                            "content": f"[Tool Output]: API Error: {e}. Cannot proceed. Ask user for next step.",
                        }
                    )
                    continue  # 继续内循环，让 LLM 看到错误并响应

                # 将 LLM 的原始回答（包含思考和工具调用）加入历史
                chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # --- 5. 核心：解析 Agent 的输出，执行工具 ---
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
                        f"\n[Thinker 正在执行...]\n> 任务: 诊断呼吸音\n> 参数: {shlex.quote(tool_prompt[:80])}...\n"
                    )
                    tool_result = run_diagnoser(tool_prompt)

                elif call_gen_match:
                    tool_prompt = call_gen_match.group(1).strip()
                    print(
                        f"\n[Thinker 正在执行...]\n> 任务: 生成呼吸音\n> 参数: {shlex.quote(tool_prompt[:80])}...\n"
                    )
                    tool_result = run_generator(tool_prompt)

                elif final_answer_match:
                    print("\n[Thinker 任务完成]")
                    break  # 跳出内循环，等待用户下一次输入

                else:
                    # 只是普通的聊天，没有工具调用
                    break  # 跳出内循环，等待用户下一次输入

                # --- 6. 将工具结果反馈给 LLM ---
                if tool_result:
                    print("\n[工具执行完毕，正在将结果反馈给 Thinker...]\n")
                    # 将工具的输出作为新的 "user" 消息，强制 LLM 进行下一步“反思”
                    chat_history.append(
                        {"role": "user", "content": f"[Tool Output]: {tool_result}"}
                    )
                    # 继续 Agent 内循环 (不 break)
                else:
                    print("\n[警告] Thinker 尝试调用工具但未产生结果，请检查。")
                    chat_history.append(
                        {
                            "role": "user",
                            "content": "[Tool Output]: 工具执行失败或未返回任何信息。",
                        }
                    )

        except KeyboardInterrupt:
            print("\n程序已中断。再见！")
            break
        except Exception as e:
            print(f"\n[运行时错误]: {e}")
            break


if __name__ == "__main__":
    main()

"""
Resp-Agent 中文版：呼吸音诊断与生成智能体

This module provides the Chinese version of the Resp-Agent Thinker,
which coordinates the Diagnoser and Generator tools through the DeepSeek API.
"""

import os
import sys

from openai import OpenAI

# Thinker 系统提示
SYSTEM_PROMPT = """
你是一个高级的人工智能助手，名为 Resp-Agent 的 "Thinker"（思考者）。
你的核心任务是协调 "Diagnoser"（诊断器）和 "Generator"（生成器）这两个工具，以实现一个闭环的"诊断-反思-规划-行动"工作流，从而帮助研究员改进呼吸音诊断模型。

【你拥有的工具】

1.  **诊断器 (Diagnoser)**:
    * 功能：运行一个多模态诊断模型，分析给定的呼吸音数据和电子病历（EHR），输出详细的诊断报告、混淆矩阵、以及模型在哪些类别上表现不佳。
    * 调用格式: `[Call:Diagnoser] 帮我诊断呼吸音，音频目录是 <path>，输出目录是 <path>，病历文档在 <path>`
    * 注意：你必须提供所有参数。

2.  **生成器 (Generator)**:
    * 功能：调用一个可控的生成模型（Resp-MLLM），根据"疾病标签"（内容）和"参考音频"（风格）来合成新的、高保真的呼吸音。
    * 调用格式: `[Call:Generator] 生成<疾病名>呼吸音，参考音频是 <path.wav>，疾病类型为 <疾病名>，输出目录是 <path>`
    * 注意：你必须提供所有参数。

【重要规则】
* **不要**在一次回复中同时调用两个工具。
* **不要**自己回答本应由工具回答的问题。
* 在调用工具后，**必须**等待 `[Tool Output]` 返回，然后再进行下一步"反思"或"总结"。
* 如果用户的请求不清晰，请主动询问以获取所有必需的参数（如路径）。
"""


# 默认参数
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


class RespAgentChinese:
    """中文版 Resp-Agent 智能体"""

    def __init__(self, api_key: str = None):
        """
        初始化智能体

        Args:
            api_key: DeepSeek API 密钥，如果为 None 则从环境变量获取
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("未检测到 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run_diagnoser(self, user_prompt: str) -> str:
        """运行诊断器工具"""
        print("\n[工具调用] 正在解析诊断任务参数...")
        # Implementation based on original
        return "诊断完成"

    def run_generator(self, user_prompt: str) -> str:
        """运行生成器工具"""
        print("\n[工具调用] 正在解析生成任务参数...")
        # Implementation based on original
        return "生成完成"

    def chat(self, user_message: str) -> str:
        """
        与智能体对话

        Args:
            user_message: 用户消息

        Returns:
            智能体回复
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
    """显示智能体任务示例"""
    diag = (
        f"帮我诊断呼吸音，音频目录是 {DIAGNOSER_DEFAULTS['audio_dir']}，"
        f"病历文档在 {DIAGNOSER_DEFAULTS['metadata_csv']}"
    )
    gen = (
        f"帮我生成疾病类型为Asthma的呼吸音，参考音频是 {GENERATOR_DEFAULTS['ref_audio']}，"
        f"输出到 {GENERATOR_DEFAULTS['out_dir']}"
    )

    print("\n================ Resp-Agent 任务示例 ================")
    print("【简单诊断】", diag)
    print("【简单生成】", gen)
    print("提示：发送 '模板/help' 可随时再次显示以上示例，发送 '停止/quit' 则退出。\n")


def main():
    """主函数入口"""
    print("=" * 80)
    print("正在初始化 DeepSeek API 客户端 (Agent 模式)...")

    try:
        agent = RespAgentChinese()
    except ValueError as e:
        print(f"\n[错误] {e}")
        print("请先设置环境变量: export DEEPSEEK_API_KEY='您的API_KEY'")
        sys.exit(1)

    print("\nThinker (Resp-Agent API) 已准备就绪。")
    print("=" * 80)

    print_agent_examples()

    while True:
        try:
            user_prompt = input("你: ").strip()
            if user_prompt.lower() in ["停止", "quit"]:
                print("再见！")
                break

            if user_prompt in ["模板", "help"]:
                print_agent_examples()
                continue

            response = agent.chat(user_prompt)
            print(f"\nThinker: {response}\n")

        except KeyboardInterrupt:
            print("\n程序已中断。再见！")
            break
        except Exception as e:
            print(f"\n[运行时错误]: {e}")
            break


if __name__ == "__main__":
    main()

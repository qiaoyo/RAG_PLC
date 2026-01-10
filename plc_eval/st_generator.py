import sys
import re

# 定义基础模板
CONFIG_TEMPLATE = """
{proxy_program}

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : {entry_program_name};
  END_RESOURCE
END_CONFIGURATION
"""

PROXY_PROGRAM_TEMPLATE = """
PROGRAM Auto_Proxy_Prg
  VAR
    instance : {fb_name};
  END_VAR
  instance();
END_PROGRAM
"""

WRAPPER_TEMPLATE = """
FUNCTION_BLOCK LLM_Wrapper_FB
{content}
END_FUNCTION_BLOCK
"""

def clean_llm_output(content):
    """去除 Markdown 标记和首尾空白"""
    # 去除 ```st, ```iec, ``` 等标记
    content = re.sub(r'```[a-zA-Z]*', '', content)
    content = content.replace('```', '')
    return content.strip()

def analyze_and_generate(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # 1. 清洗内容
    clean_content = clean_llm_output(raw_content)
    
    # 2. 正则表达式匹配，寻找现有的块定义
    # 匹配 FUNCTION_BLOCK xxx 或 PROGRAM xxx
    # re.IGNORECASE 忽略大小写
    fb_match = re.search(r'FUNCTION_BLOCK\s+(\w+)', clean_content, re.IGNORECASE)
    prog_match = re.search(r'PROGRAM\s+(\w+)', clean_content, re.IGNORECASE)

    final_st_content = ""
    entry_program_name = ""
    proxy_program_code = ""

    # --- 逻辑分支 ---

    if prog_match:
        # CASE 1: LLM 输出了一个完整的 PROGRAM
        print(f"Detected existing PROGRAM: {prog_match.group(1)}")
        final_st_content = clean_content
        entry_program_name = prog_match.group(1)
        proxy_program_code = "" # 不需要代理程序

    elif fb_match:
        # CASE 2: LLM 输出了一个完整的 FUNCTION_BLOCK
        fb_name = fb_match.group(1)
        print(f"Detected existing FUNCTION_BLOCK: {fb_name}")
        
        final_st_content = clean_content
        # 我们需要创建一个代理 Program 来实例化这个 FB
        entry_program_name = "Auto_Proxy_Prg"
        proxy_program_code = PROXY_PROGRAM_TEMPLATE.format(fb_name=fb_name)

    else:
        # CASE 3: 只有代码片段 (只有 VAR 或 逻辑)
        print("Detected raw code fragment. Wrapping in FUNCTION_BLOCK.")
        
        # 封装代码
        final_st_content = WRAPPER_TEMPLATE.format(content=clean_content)
        
        # 创建代理 Program 调用封装后的 FB
        entry_program_name = "Auto_Proxy_Prg"
        proxy_program_code = PROXY_PROGRAM_TEMPLATE.format(fb_name="LLM_Wrapper_FB")

    # 3. 组合最终文件
    full_source = f"{final_st_content}\n\n{CONFIG_TEMPLATE.format(proxy_program=proxy_program_code, entry_program_name=entry_program_name)}"

    # 4. 写入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_source)
    
    print(f"✅ Generated {output_file} successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python st_generator_v2.py <input.txt> <output.st>")
    else:
        analyze_and_generate(sys.argv[1], sys.argv[2])
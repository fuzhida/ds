#!/usr/bin/env python3
"""测试JSON解析功能"""

import json
import re

def safe_json_parse(json_str: str):
    """安全解析JSON字符串，支持多种格式修复和候选选择"""
    print(f"原始JSON字符串:\n{json_str}")
    
    try:
        result = json.loads(json_str)
        print("直接json.loads成功")
        # 确保结果包含signal字段
        if isinstance(result, dict) and 'signal' in result:
            print("结果包含signal字段，返回结果")
            return result
        print("结果不包含signal字段，返回None")
        return None
    except json.JSONDecodeError as e:
        print(f"直接json.loads失败: {e}")
        try:
            # 清理markdown格式
            json_str = json_str.strip()
            print(f"清理前: {json_str}")
            if json_str.startswith('```json'):
                json_str = json_str[7:]  # Remove ```json
                print("移除了```json前缀")
            if json_str.startswith('```'):
                json_str = json_str[3:]  # Remove ```
                print("移除了```前缀")
            if json_str.endswith('```'):
                json_str = json_str[:-3]  # Remove trailing ```
                print("移除了```后缀")
            json_str = json_str.strip()
            print(f"清理后: {json_str}")
            
            # 基本格式修复
            json_str = json_str.replace("'", '"').strip()
            # 不使用有问题的正则表达式修复
            # json_str = re.sub(r'(?<!")(\w+)(?":\s*)', r'"\1"', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            print(f"格式修复后: {json_str}")
            
            def find_json_blocks(s):
                """查找字符串中的JSON块"""
                candidates = []
                depth = 0
                start = -1
                for i, c in enumerate(s):
                    if c == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0 and start != -1:
                            candidates.append(s[start:i+1])
                            start = -1
                return candidates
            
            # 查找所有可能的JSON块
            blocks = find_json_blocks(json_str)
            print(f"找到的JSON块: {blocks}")
            candidates = []
            
            for block in blocks:
                try:
                    print(f"尝试解析块: {block}")
                    cand = json.loads(block)
                    print(f"块解析成功: {cand}")
                    if isinstance(cand, dict) and 'signal' in cand:
                        candidates.append(cand)
                        print("添加候选项")
                except json.JSONDecodeError as e:
                    print(f"块解析失败: {e}")
                    pass
            
            # 如果找到候选项，选择置信度最高的
            if candidates:
                print(f"找到{len(candidates)}个候选项")
                conf_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                best = max(candidates, key=lambda x: conf_order.get(x.get('confidence', 'LOW'), 0))
                print(f"Selected highest confidence candidate: {best.get('confidence')}")
                
                # 详细诊断日志
                sl = best.get('stop_loss')
                tp = best.get('take_profit')
                print(f"Candidate details: signal={best.get('signal')}, SL={sl}, TP={tp}, confidence={best.get('confidence')}")
                
                # 检查数值类型
                if sl is None or tp is None:
                    print(f"DeepSeek返回了空值: SL={sl}, TP={tp}")
                    print(f"完整候选项: {best}")
                elif not isinstance(sl, (int, float)) or not isinstance(tp, (int, float)):
                    print(f"DeepSeek返回了非数值类型: SL={type(sl).__name__}({sl}), TP={type(tp).__name__}({tp})")
                    print(f"完整候选项: {best}")
                
                return best
            
            print("没有找到有效的候选项")
            return None
            
        except Exception as e:
            print(f"JSON parsing failed, raw content: {json_str[:200]}... Error: {e}")
            return None

# 测试JSON字符串
test_json = """```json
{
  "signal": "HOLD",
  "reason": "Mixed signals - Higher TFs show strong uptrend but current price below key EMAs (EMA21:105558, EMA55:107211) with recent bearish momentum. No clear 5m confirmation patterns or FVG stacking. Price trapped between DOB+ support (99533) and EMA resistance.",
  "stop_loss": 102913,
  "take_profit": 104991,
  "confidence": "LOW"
}
```"""

print("测试JSON解析...")
result = safe_json_parse(test_json)
print(f"解析结果: {result}")
#!/usr/bin/env python3
"""
SMC交易系统仪表盘 - Flask版本
用于实时监控交易关键指标和信号
"""

import json
import os
import time
from datetime import datetime
from flask import Flask, render_template, jsonify

# 自定义模板过滤器
def timestamp_to_datetime(timestamp):
    """将时间戳转换为datetime对象"""
    try:
        return datetime.fromtimestamp(float(timestamp))
    except:
        return datetime.now()

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    """格式化datetime对象"""
    if isinstance(value, datetime):
        return value.strftime(format)
    return value

app = Flask(__name__)

# 注册自定义过滤器
app.jinja_env.filters['timestamp_to_datetime'] = timestamp_to_datetime
app.jinja_env.filters['datetimeformat'] = datetimeformat

class TradingDashboard:
    def __init__(self):
        self.analysis_files = []
        self.signal_history_file = "signal_history.json"
        self.trading_config_file = "trading_config.json"
        self.ai_interaction_files = []
        
    def load_latest_analysis(self):
        """加载最新的分析文件"""
        analysis_dir = "./"
        files = [f for f in os.listdir(analysis_dir) if f.startswith("pure_rawdata_analysis_") and f.endswith(".json")]
        
        if not files:
            return []
            
        # 按时间戳排序，获取最新的文件
        latest_file = max(files, key=lambda x: x.split("_")[-1].split(".")[0])
        
        try:
            with open(os.path.join(analysis_dir, latest_file), 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                
                # 将字典格式转换为列表格式，适配仪表盘显示
                if "results" in analysis_data:
                    symbol_list = []
                    for symbol, result in analysis_data["results"].items():
                        # 确保smc_patterns字段存在
                        if "smc_patterns" not in result:
                            result["smc_patterns"] = {
                                "order_blocks": {
                                    "recent_ob_count": 0,
                                    "bullish_ob": {
                                        "support": 0,
                                        "resistance": 0
                                    }
                                },
                                "fair_value_gaps": {
                                    "recent_fvg_count": 0,
                                    "bullish_fvg": {
                                        "gap_top": 0,
                                        "gap_bottom": 0
                                    }
                                }
                            }
                        
                        symbol_data = {
                            "symbol": symbol,
                            "raw_data": result.get("raw_data", {}),
                            "analysis_result": result.get("ai_analysis", {}),
                            "smc_patterns": result.get("smc_patterns", {})
                        }
                        symbol_list.append(symbol_data)
                    return symbol_list
                else:
                    # 如果是旧格式，直接返回
                    return analysis_data
                    
        except Exception as e:
            print(f"加载分析文件失败: {e}")
            return []
    
    def load_signal_history(self):
        """加载信号历史记录"""
        try:
            if os.path.exists(self.signal_history_file):
                with open(self.signal_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载信号历史失败: {e}")
        return []
    
    def load_trading_config(self):
        """加载交易配置"""
        try:
            if os.path.exists(self.trading_config_file):
                with open(self.trading_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载交易配置失败: {e}")
        return {}
    
    def calculate_performance_metrics(self, history_data):
        """计算性能指标"""
        if not history_data:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit_per_trade": 0
            }
        
        total_trades = len(history_data)
        winning_trades = sum(1 for trade in history_data if trade.get("status") == "closed" and trade.get("profit", 0) > 0)
        
        return {
            "total_trades": total_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "total_profit": sum(trade.get("profit", 0) for trade in history_data if trade.get("status") == "closed"),
            "avg_profit_per_trade": sum(trade.get("profit", 0) for trade in history_data if trade.get("status") == "closed") / total_trades if total_trades > 0 else 0
        }
    
    def load_ai_interaction_data(self):
        """加载AI交互数据"""
        ai_data = {
            "current_ai_analysis": None,
            "ai_interaction_history": [],
            "submitted_data": [],
            "returned_data": []
        }
        
        # 加载最新的AI分析结果
        try:
            if os.path.exists("minimal_ai_trading_result.json"):
                with open("minimal_ai_trading_result.json", 'r', encoding='utf-8') as f:
                    ai_data["current_ai_analysis"] = json.load(f)
        except Exception as e:
            print(f"加载AI分析结果失败: {e}")
        
        # 加载AI交互历史记录
        try:
            # 查找所有AI交互相关的JSON文件
            ai_files = [f for f in os.listdir("./") if f.startswith("pure_rawdata_analysis_") and f.endswith(".json")]
            ai_files.sort(reverse=True)  # 按时间倒序排列
            
            for ai_file in ai_files[:10]:  # 只加载最近10个文件
                try:
                    with open(ai_file, 'r', encoding='utf-8') as f:
                        ai_interaction = json.load(f)
                        ai_data["ai_interaction_history"].append({
                            "file": ai_file,
                            "timestamp": ai_interaction.get("config", {}).get("timestamp", os.path.getmtime(ai_file)),
                            "symbols": ai_interaction.get("config", {}).get("symbols", []),
                            "trading_style": ai_interaction.get("config", {}).get("trading_style", "unknown")
                        })
                        
                        # 提取提交给AI的数据
                        for symbol, result in ai_interaction.get("results", {}).items():
                            if "raw_data" in result:
                                ai_data["submitted_data"].append({
                                    "symbol": symbol,
                                    "timestamp": result["raw_data"].get("timestamp", 0),
                                    "price": result["raw_data"].get("price_info", {}).get("current", 0),
                                    "file": ai_file
                                })
                except Exception as e:
                    print(f"加载AI交互文件 {ai_file} 失败: {e}")
        except Exception as e:
            print(f"加载AI交互历史失败: {e}")
        
        return ai_data

# 创建仪表盘实例
dashboard = TradingDashboard()

@app.route('/')
def index():
    """主页面"""
    # 加载数据
    analysis_data = dashboard.load_latest_analysis()
    history_data = dashboard.load_signal_history()
    config_data = dashboard.load_trading_config()
    performance_metrics = dashboard.calculate_performance_metrics(history_data)
    ai_interaction_data = dashboard.load_ai_interaction_data()
    
    # 获取第一个品种的数据用于显示
    symbol_data = analysis_data[0] if analysis_data else None
    
    return render_template('dashboard.html', 
                         analysis_data=analysis_data,
                         symbol_data=symbol_data,
                         history_data=history_data,
                         config_data=config_data,
                         performance_metrics=performance_metrics,
                         ai_interaction_data=ai_interaction_data)

@app.route('/api/data')
def api_data():
    """API接口，返回JSON格式的数据"""
    analysis_data = dashboard.load_latest_analysis()
    history_data = dashboard.load_signal_history()
    config_data = dashboard.load_trading_config()
    performance_metrics = dashboard.calculate_performance_metrics(history_data)
    ai_interaction_data = dashboard.load_ai_interaction_data()
    
    return jsonify({
        'analysis_data': analysis_data,
        'history_data': history_data,
        'config_data': config_data,
        'performance_metrics': performance_metrics,
        'ai_interaction_data': ai_interaction_data,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
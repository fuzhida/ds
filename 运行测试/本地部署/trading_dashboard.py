#!/usr/bin/env python3
"""
SMC交易系统仪表盘 - 简化版
用于实时监控交易关键指标和信号
"""

import json
import os
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 设置Streamlit页面配置
st.set_page_config(
    page_title="SMC交易系统仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        self.analysis_files = []
        self.signal_history_file = "signal_history.json"
        self.trading_config_file = "trading_config.json"
        
    def load_latest_analysis(self):
        """加载最新的分析文件"""
        analysis_dir = "./"
        files = [f for f in os.listdir(analysis_dir) if f.startswith("pure_rawdata_analysis_") and f.endswith(".json")]
        
        if not files:
            return None
            
        # 按时间戳排序，获取最新的文件
        latest_file = max(files, key=lambda x: x.split("_")[-1].split(".")[0])
        
        try:
            with open(os.path.join(analysis_dir, latest_file), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"加载分析文件失败: {e}")
            return None
    
    def load_signal_history(self):
        """加载信号历史记录"""
        try:
            if os.path.exists(self.signal_history_file):
                with open(self.signal_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"加载信号历史失败: {e}")
        return []
    
    def load_trading_config(self):
        """加载交易配置"""
        try:
            if os.path.exists(self.trading_config_file):
                with open(self.trading_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"加载交易配置失败: {e}")
        return {}
    
    def calculate_performance_metrics(self, history_data):
        """计算性能指标"""
        if not history_data:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit_per_trade": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0
            }
        
        # 简化计算，实际应根据具体数据结构调整
        total_trades = len(history_data)
        winning_trades = sum(1 for trade in history_data if trade.get("status") == "closed" and trade.get("profit", 0) > 0)
        
        return {
            "total_trades": total_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "total_profit": sum(trade.get("profit", 0) for trade in history_data if trade.get("status") == "closed"),
            "avg_profit_per_trade": sum(trade.get("profit", 0) for trade in history_data if trade.get("status") == "closed") / total_trades if total_trades > 0 else 0,
            "max_consecutive_wins": 0,  # 需要更复杂的计算
            "max_consecutive_losses": 0  # 需要更复杂的计算
        }
    
    def create_simple_metrics_display(self, symbol_data):
        """创建简单的指标显示"""
        if not symbol_data:
            return None
            
        # 显示关键指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("当前价格", f"${symbol_data['raw_data']['price_data']['current']:,.2f}")
            st.metric("决策", symbol_data["analysis_result"]["decision"])
        
        with col2:
            st.metric("置信度", f"{symbol_data['analysis_result']['confidence']*100:.1f}%")
            
            # 方向分析
            if "direction_analysis" in symbol_data["analysis_result"]:
                dir_analysis = symbol_data["analysis_result"]["direction_analysis"]
                st.metric("4小时方向", dir_analysis["4h_direction"])
        
        with col3:
            # 风险分析
            risk_data = symbol_data["analysis_result"]
            st.metric("入场价格", f"${risk_data['entry_price']:,.2f}")
            st.metric("止损价格", f"${risk_data['stop_loss']:,.2f}")
    
    def display_dashboard(self):
        """显示仪表盘"""
        st.title("🎯 SMC交易系统仪表盘")
        st.markdown("---")
        
        # 加载数据
        analysis_data = self.load_latest_analysis()
        history_data = self.load_signal_history()
        config_data = self.load_trading_config()
        performance_metrics = self.calculate_performance_metrics(history_data)
        
        # 侧边栏
        with st.sidebar:
            st.header("系统状态")
            st.metric("总交易次数", performance_metrics["total_trades"])
            st.metric("胜率", f"{performance_metrics['win_rate']:.1f}%")
            st.metric("总盈利", f"${performance_metrics['total_profit']:.2f}")
            
            st.header("配置信息")
            if config_data:
                st.text(f"最大并发交易: {config_data.get('max_concurrent_trades', 3)}")
                st.text(f"风险比例: {config_data.get('risk_percentage', 0.5)}%")
        
        # 主内容区域
        if not analysis_data:
            st.warning("未找到分析数据，请先运行交易分析")
            return
        
        # 创建标签页
        tab1, tab2 = st.tabs(["📈 实时监控", "📊 性能分析"])
        
        with tab1:
            # 实时监控标签页
            st.subheader("🔍 品种分析")
            
            # 品种选择器
            symbols = [item["symbol"] for item in analysis_data]
            selected_symbol = st.selectbox("选择品种", symbols)
            
            # 获取选中品种的数据
            symbol_data = next((item for item in analysis_data if item["symbol"] == selected_symbol), None)
            
            if symbol_data:
                self.create_simple_metrics_display(symbol_data)
                
                # 显示详细分析结果
                st.subheader("📋 详细分析")
                
                # 显示方向分析
                if "direction_analysis" in symbol_data["analysis_result"]:
                    dir_analysis = symbol_data["analysis_result"]["direction_analysis"]
                    
                    col1_dir, col2_dir, col3_dir = st.columns(3)
                    with col1_dir:
                        st.metric("4小时方向", dir_analysis["4h_direction"])
                    with col2_dir:
                        st.metric("1小时方向", dir_analysis["1h_direction"])
                    with col3_dir:
                        st.metric("当前方向", dir_analysis["current_direction"])
                    
                    # 方向一致性状态
                    consistency_status = "✅ 一致" if dir_analysis["directions_consistent"] else "❌ 不一致"
                    st.metric("方向一致性", consistency_status)
                
                # 显示技术指标
                st.subheader("📊 技术指标")
                
                ob_data = symbol_data["smc_patterns"]["order_blocks"]
                fvg_data = symbol_data["smc_patterns"]["fair_value_gaps"]
                
                col1_tech, col2_tech = st.columns(2)
                
                with col1_tech:
                    st.metric("OB数量", ob_data["recent_ob_count"])
                    st.metric("看涨支撑", f"${ob_data['bullish_ob']['support']:,.2f}")
                    st.metric("看涨阻力", f"${ob_data['bullish_ob']['resistance']:,.2f}")
                
                with col2_tech:
                    st.metric("FVG数量", fvg_data["recent_fvg_count"])
                    st.metric("看涨FVG顶部", f"${fvg_data['bullish_fvg']['gap_top']:,.2f}")
                    st.metric("看涨FVG底部", f"${fvg_data['bullish_fvg']['gap_bottom']:,.2f}")
        
        with tab2:
            # 性能分析标签页
            st.subheader("📈 交易性能")
            
            col1_perf, col2_perf, col3_perf, col4_perf = st.columns(4)
            
            with col1_perf:
                st.metric("总交易次数", performance_metrics["total_trades"])
            with col2_perf:
                st.metric("胜率", f"{performance_metrics['win_rate']:.1f}%")
            with col3_perf:
                st.metric("总盈利", f"${performance_metrics['total_profit']:.2f}")
            with col4_perf:
                st.metric("单笔平均盈利", f"${performance_metrics['avg_profit_per_trade']:.2f}")
            
            # 显示最近交易记录
            st.subheader("📋 最近交易记录")
            
            if history_data:
                for trade in history_data[-5:]:  # 显示最近5笔交易
                    with st.expander(f"交易: {trade.get('symbol', 'N/A')} - {trade.get('timestamp', 'N/A')}"):
                        st.json(trade)
            else:
                st.info("暂无交易记录")

def main():
    """主函数"""
    dashboard = TradingDashboard()
    dashboard.display_dashboard()

if __name__ == "__main__":
    main()
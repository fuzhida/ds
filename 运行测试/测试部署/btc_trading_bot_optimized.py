"""
修改版btc_trading_bot.py
集成优化后的SMC/ICT策略分析提示词
"""

# 导入优化后的提示词函数
from optimized_smc_prompt import get_optimized_smc_prompt

# 以下是需要替换的analyze_with_deepseek方法
def analyze_with_deepseek_optimized(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    使用优化后的SMC/ICT策略分析提示词进行市场分析
    """
    try:
        if deepseek_client is None:
            self.logger_system.error("DeepSeek client not available")
            return None

        # 提取市场数据
        current_price = price_data['price']
        technical_data = price_data.get('technical_data', {})
        smc_structures = price_data.get('smc_structures', {})
        mtf_analysis = price_data.get('mtf_analysis', {})
        
        # 获取时间框架数据
        higher_tf = config.higher_tf_bias_tf
        primary_tf = config.primary_timeframe
        
        # 提取多时间框架分析数据
        higher_tf_trend = mtf_analysis.get(higher_tf, {}).get('trend', 'neutral')
        higher_tf_strength = mtf_analysis.get(higher_tf, {}).get('strength', 0.5)
        primary_tf_trend = mtf_analysis.get(primary_tf, {}).get('trend', 'neutral')
        primary_tf_strength = mtf_analysis.get(primary_tf, {}).get('strength', 0.5)
        mtf_consistency = mtf_analysis.get('consistency', 0.5)
        
        # 提取SMC结构数据
        structure_score = smc_structures.get('structure_score', 0.5)
        structure_count = smc_structures.get('meaningful_count', 0)
        structure_quality = smc_structures.get('structure_quality', '中等')
        
        # 提取技术指标
        rsi = technical_data.get('rsi', 50)
        macd_line = technical_data.get('macd', 0)
        macd_signal = technical_data.get('macd_signal', 0)
        macd_histogram = macd_line - macd_signal
        
        # 计算成交量比率
        volume_ratio = 1.0
        if 'multi_tf_data' in price_data and primary_tf in price_data['multi_tf_data']:
            df = price_data['multi_tf_data'][primary_tf]
            if not df.empty and 'volume' in df.columns and len(df) > 20:
                volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                if volume_ma > 0:
                    volume_ratio = current_volume / volume_ma
        
        # 提取风险参数
        volatility = price_data.get('volatility', 2.0)
        min_rr_ratio = config.rr_min_threshold
        invalidation_point = smc_structures.get('higher_tf_choch_bos_invalidation', current_price * 0.98)
        
        # 提取关键水平
        nearest_key_level = smc_structures.get('nearest_key_level', current_price * 0.98)
        key_level_distance = smc_structures.get('key_level_distance', 0.02)
        
        # 准备市场数据字典
        market_data = {
            'current_price': current_price,
            'symbol': config.symbol,
            'higher_tf': higher_tf,
            'higher_tf_trend': higher_tf_trend,
            'higher_tf_strength': higher_tf_strength,
            'primary_tf': primary_tf,
            'primary_tf_trend': primary_tf_trend,
            'primary_tf_strength': primary_tf_strength,
            'mtf_consistency': mtf_consistency,
            'structure_score': structure_score,
            'structure_count': structure_count,
            'structure_quality': structure_quality,
            'rsi': rsi,
            'macd_histogram': macd_histogram,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'min_rr_ratio': min_rr_ratio,
            'invalidation_point': invalidation_point,
            'nearest_key_level': nearest_key_level,
            'key_level_distance': key_level_distance * 100  # 转换为百分比
        }
        
        # 生成优化后的提示词
        prompt = get_optimized_smc_prompt(market_data)
        
        # 记录提示词
        self.logger_system.info("=" * 80)
        self.logger_system.info("📤 发送给DeepSeek的优化提示词:")
        self.logger_system.info("-" * 40)
        self.logger_system.info(prompt.strip())
        self.logger_system.info("-" * 40)
        
        # 调用DeepSeek API
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=config.temperature
        )
        
        signal_text = response.choices[0].message.content.strip()
        
        # 记录DeepSeek的完整响应
        self.logger_system.info("📥 DeepSeek的完整响应:")
        self.logger_system.info("-" * 40)
        self.logger_system.info(signal_text)
        self.logger_system.info("-" * 40)
        self.logger_system.info("=" * 80)
        
        # 提取JSON部分
        start_idx = signal_text.find('{')
        end_idx = signal_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = signal_text[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
        
        # 验证信号数据完整性
        required_fields = ['signal', 'entry_price', 'stop_loss', 'take_profit', 'confidence', 'reason']
        if not all(field in signal_data for field in required_fields):
            self.logger_system.warning("Incomplete signal data, using fallback")
            signal_data = self._generate_fallback_signal(price_data, activated_level)
        
        # 验证信号值的合理性
        if signal_data['signal'] not in ['BUY', 'SELL', 'HOLD']:
            signal_data['signal'] = 'HOLD'
        
        self.logger_system.info(f"Generated optimized signal: {signal_data['signal']} at {signal_data['entry_price']:.2f}")
        return signal_data
    
    except (json.JSONDecodeError, ValueError, Exception) as e:
        self.logger_system.error(f"Optimized DeepSeek analysis failed: {e}")
        return self._generate_fallback_signal(price_data, activated_level)


# 使用说明
"""
集成步骤:
1. 将optimized_smc_prompt.py文件与btc_trading_bot.py放在同一目录
2. 在btc_trading_bot.py文件顶部添加导入语句:
   from optimized_smc_prompt import get_optimized_smc_prompt
3. 将原有的analyze_with_deepseek方法替换为analyze_with_deepseek_optimized方法
4. 或者，在原有方法中添加一个配置选项，允许选择使用原版或优化版提示词

优化点:
1. 简化了提示词结构，减少复杂变量和条件
2. 明确定义了AI专业判断权限和标准
3. 放宽了技术指标限制，提高灵活性
4. 分离了数据处理代码与提示词定义
5. 提供了更清晰的分析重点和输出要求
"""
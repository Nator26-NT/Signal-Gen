"""
Forex AI Signal Generator - AI Logic Module
Integrated with Twelve Data API
Python 3.7 compatible
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import requests
import time
import warnings
warnings.filterwarnings('ignore')

class ForexAISignalGenerator:
    def __init__(self):
        """Initialize with Twelve Data API configuration"""
        # Your API credentials
        self.api_key = "fe6aec0e85244251ab5cb28263f98bd6"
        self.base_url = "https://api.twelvedata.com/time_series"
        
        # Rate limiting
        self.rate_limit_delay = 0.5
        self.last_request_time = 0
        
        # Available pairs and timeframes
        self.available_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", 
                               "USDCAD", "USDCHF", "NZDUSD", "EURGBP", 
                               "EURJPY", "GBPJPY", "AUDJPY", "CADJPY"]
        
        self.timeframe_map = {
            "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min",
            "1h": "1h", "4h": "4h", "1day": "1day", "1week": "1week", "1month": "1month"
        }
        
        # Output size mapping
        self.outputsize_map = {
            "1min": 1440, "5min": 576, "15min": 384, "30min": 288,
            "1h": 168, "4h": 126, "1day": 500, "1week": 260, "1month": 120
        }
        
        # Fixed risk parameters (7 SL, 10 TP)
        self.risk_params = {
            "sl_pips": 7,
            "tp_pips": 10,
            "risk_reward": 1.43,
            "description": "Quick Scalping",
            "hold_period": "15-30 min"
        }
        
        # Pattern weights
        self.pattern_weights = {
            "three_line_strike": 3.0,
            "double_bottom": 2.0,
            "hammer": 2.0,
            "double_top": -2.0,
            "shooting_star": -2.0,
            "bullish_engulfing": 1.5,
            "bearish_engulfing": -1.5
        }
        
        # AI settings
        self.ai_settings = {
            "lookback_period": 50,
            "regression_window": 20,
            "min_data_points": 50,
            "prediction_horizon": 3,
            "confidence_threshold_high": 0.7,
            "confidence_threshold_medium": 0.6,
            "confidence_threshold_low": 0.5,
            "signal_score_threshold_high": 8,
            "signal_score_threshold_medium": 6,
            "signal_score_threshold_low": 4,
            "pattern_confidence_threshold": 0.7,
            "quick_trade_lookback": 30,
            "quick_trade_regression_window": 10
        }

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def fetch_twelvedata(self, symbol, interval, outputsize=100):
        """
        Fetch data from Twelve Data API
        
        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            interval: Time interval (e.g., '1min', '5min', '15min', '1h')
            outputsize: Number of data points to fetch
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self._rate_limit()
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            print(f"🌐 API Request: {symbol} {interval} ({outputsize} bars)")
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    df = pd.DataFrame(data['values'])
                    
                    # Convert and sort datetime
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    
                    # Convert columns to numeric
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"✅ API Success: {len(df)} bars for {symbol} {interval}")
                    return df
                else:
                    print(f"⚠ API Warning: No data for {symbol} {interval}")
                    return None
            else:
                print(f"❌ API Error {response.status_code}: {symbol} {interval}")
                return None
                
        except Exception as e:
            print(f"❌ API Exception: {symbol} - {str(e)}")
            return None

    def fetch_market_data(self, symbol, timeframe, use_backup=True):
        """
        Fetch forex data from Twelve Data API with fallback
        
        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '5min', '1h', '1day')
            use_backup: Whether to use backup data if API fails
        
        Returns:
            DataFrame with OHLCV data
        """
        # Get interval from timeframe
        interval = self.timeframe_map.get(timeframe, "15min")
        outputsize = self.outputsize_map.get(timeframe, 100)
        
        # Try Twelve Data API
        df = self.fetch_twelvedata(symbol, interval, outputsize)
        
        # If API failed and backup is enabled, generate synthetic data
        if df is None or len(df) < 20:
            if use_backup:
                print(f"🔄 Using synthetic data for {symbol} {timeframe}")
                df = self._generate_synthetic_data(symbol, timeframe)
            else:
                return None
        
        # Ensure we have enough data
        if len(df) >= 20:
            df = self._clean_dataframe(df)
            return df
        else:
            print(f"❌ Insufficient data: {symbol} - {len(df)} bars")
            return None

    def _clean_dataframe(self, df):
        """Clean and prepare dataframe"""
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Sort by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            df = df.set_index('datetime')
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure numeric types
        for col in required_cols + ['volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.ffill().bfill()
        
        return df

    def _generate_synthetic_data(self, symbol, timeframe):
        """
        Generate realistic synthetic forex data
        
        Args:
            symbol: Forex pair
            timeframe: Timeframe
        
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Base prices for different pairs
        base_prices = {
            "EURUSD": 1.0950, "GBPUSD": 1.2750, "USDJPY": 147.50,
            "AUDUSD": 0.6650, "USDCAD": 1.3500, "USDCHF": 0.8800,
            "NZDUSD": 0.6200, "EURGBP": 0.8600, "EURJPY": 160.50,
            "GBPJPY": 188.00, "AUDJPY": 98.00, "CADJPY": 109.00
        }
        
        clean_symbol = symbol.replace("=X", "")
        base_price = base_prices.get(clean_symbol, 1.1000)
        
        # Data points based on timeframe
        points_map = {
            "1min": 500, "5min": 400, "15min": 300, "30min": 200,
            "1h": 150, "4h": 100, "1day": 80, "1week": 40, "1month": 20
        }
        n_points = points_map.get(timeframe, 200)
        
        # Generate dates
        end_date = datetime.now()
        freq_map = {
            "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min",
            "1h": "1h", "4h": "4h", "1day": "D", "1week": "W", "1month": "M"
        }
        dates = pd.date_range(
            end=end_date,
            periods=n_points,
            freq=freq_map.get(timeframe, "15min")
        )
        
        # Generate realistic price movements
        np.random.seed(42)
        
        # Create components
        x = np.arange(n_points)
        
        # Trend component
        trend_slope = np.random.uniform(-0.00001, 0.00001)
        trend = trend_slope * x
        
        # Seasonal component
        if timeframe in ["1min", "5min", "15min", "30min", "1h"]:
            seasonal = 0.0003 * np.sin(2 * np.pi * x / 100)
        else:
            seasonal = 0.001 * np.sin(2 * np.pi * x / 20)
        
        # Noise component
        volatility = 0.0005 if timeframe in ["1min", "5min"] else 0.001
        noise = np.random.normal(0, volatility, n_points)
        
        # Combine components
        log_prices = np.log(base_price) + trend + seasonal + np.cumsum(noise)
        prices = np.exp(log_prices)
        
        # Generate OHLC data
        df = pd.DataFrame(index=dates)
        df["close"] = prices
        
        # Realistic spreads
        spread_factor = 0.01 if "JPY" in clean_symbol else 0.005
        spreads = np.random.uniform(spread_factor * 0.5, spread_factor * 1.5, n_points)
        
        df["open"] = df["close"].shift(1) * (1 + np.random.normal(0, 0.0001, n_points))
        df["open"].iloc[0] = base_price
        
        df["high"] = df["close"] + spreads * np.random.uniform(1, 3, n_points)
        df["low"] = df["close"] - spreads * np.random.uniform(1, 3, n_points)
        
        # Ensure high > low > 0
        df["high"] = df[["high", "close"]].max(axis=1)
        df["low"] = df[["low", "close"]].min(axis=1)
        df["low"] = df["low"].clip(lower=0.00001)
        
        # Generate volume
        volume_base = 1000000 if clean_symbol in ["EURUSD", "USDJPY", "GBPUSD"] else 500000
        df["volume"] = np.random.lognormal(np.log(volume_base), 0.5, n_points).astype(int)
        
        print(f"📊 Synthetic: {len(df)} bars for {clean_symbol} {timeframe}")
        return df

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        
        # Basic calculations
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Moving averages
        sma_windows = [5, 10, 20, 50]
        ema_windows = [5, 10, 20]
        
        for window in sma_windows:
            if len(df) > window:
                df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        
        for window in ema_windows:
            if len(df) > window:
                df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        bb_period = 20
        df["bb_middle"] = df["close"].rolling(window=bb_period).mean()
        bb_std = df["close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        
        # Support and Resistance
        window = 20
        df["resistance"] = df["high"].rolling(window=window).max()
        df["support"] = df["low"].rolling(window=window).min()
        
        # Clean data
        df = df.dropna()
        
        return df

    def detect_candlestick_patterns(self, df):
        """Detect candlestick patterns"""
        patterns = {
            'three_line_strike': False,
            'double_top': False,
            'double_bottom': False,
            'hammer': False,
            'shooting_star': False,
            'bullish_engulfing': False,
            'bearish_engulfing': False
        }
        
        if len(df) < 5:
            return patterns
        
        latest = df.iloc[-5:]
        
        # Three Line Strike Pattern
        if len(latest) >= 4:
            if (latest['close'].iloc[-4] < latest['open'].iloc[-4] and
                latest['close'].iloc[-3] < latest['open'].iloc[-3] and
                latest['close'].iloc[-2] < latest['open'].iloc[-2] and
                latest['close'].iloc[-1] > latest['open'].iloc[-1] and
                latest['close'].iloc[-1] > latest['open'].iloc[-4]):
                patterns['three_line_strike'] = True
        
        # Double Top/Bottom
        if len(df) >= 30:
            lookback = df.iloc[-30:-5]
            
            # Find peaks and troughs
            highs = lookback['high'].values
            lows = lookback['low'].values
            
            # Simple peak detection
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    # Found a peak
                    for j in range(i+5, len(highs)-1):
                        if highs[j] > highs[j-1] and highs[j] > highs[j+1]:
                            if abs(highs[i] - highs[j]) / highs[i] < 0.002:
                                patterns['double_top'] = True
                                break
            
            # Trough detection
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    # Found a trough
                    for j in range(i+5, len(lows)-1):
                        if lows[j] < lows[j-1] and lows[j] < lows[j+1]:
                            if abs(lows[i] - lows[j]) / lows[i] < 0.002:
                                patterns['double_bottom'] = True
                                break
        
        # Single candlestick patterns
        if len(df) >= 2:
            latest_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Calculate candle properties
            body_size = abs(latest_candle['close'] - latest_candle['open'])
            upper_wick = latest_candle['high'] - max(latest_candle['close'], latest_candle['open'])
            lower_wick = min(latest_candle['close'], latest_candle['open']) - latest_candle['low']
            total_range = latest_candle['high'] - latest_candle['low']
            
            # Hammer pattern
            if body_size > 0 and total_range > 0:
                if lower_wick > 2 * body_size and upper_wick < body_size * 0.3:
                    patterns['hammer'] = True
                
                # Shooting star pattern
                if upper_wick > 2 * body_size and lower_wick < body_size * 0.3:
                    patterns['shooting_star'] = True
            
            # Engulfing patterns
            # Bullish engulfing
            if (prev_candle['close'] < prev_candle['open'] and  # Previous bearish
                latest_candle['close'] > latest_candle['open'] and  # Current bullish
                latest_candle['open'] < prev_candle['close'] and
                latest_candle['close'] > prev_candle['open']):
                patterns['bullish_engulfing'] = True
            
            # Bearish engulfing
            if (prev_candle['close'] > prev_candle['open'] and  # Previous bullish
                latest_candle['close'] < latest_candle['open'] and  # Current bearish
                latest_candle['open'] > prev_candle['close'] and
                latest_candle['close'] < prev_candle['open']):
                patterns['bearish_engulfing'] = True
        
        return patterns

    def calculate_trend_analysis(self, df, window=20):
        """Perform linear regression trend analysis"""
        if len(df) < window:
            return {
                'slope': 0,
                'r_squared': 0,
                'trend_strength': 'neutral',
                'trend_direction': 'neutral',
                'volatility': 0
            }
        
        recent = df['close'].iloc[-window:].reset_index(drop=True)
        X = np.arange(len(recent)).reshape(-1, 1)
        y = recent.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        r_squared = r_value ** 2
        
        # Determine trend strength
        if abs(slope) > 0.001:
            trend_strength = 'strong'
        elif abs(slope) > 0.0001:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        # Determine trend direction
        if slope > 0.00005:
            trend_direction = 'up'
        elif slope < -0.00005:
            trend_direction = 'down'
        else:
            trend_direction = 'neutral'
        
        # Calculate volatility
        returns = df['returns'].iloc[-window:].dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'slope': float(slope),
            'r_squared': float(r_squared),
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'volatility': float(volatility)
        }

    def calculate_signal_score(self, df, patterns, trend_analysis):
        """Calculate comprehensive signal score"""
        latest = df.iloc[-1]
        score = 0
        breakdown = []
        
        # 1. Pattern recognition
        for pattern, detected in patterns.items():
            if detected:
                weight = self.pattern_weights.get(pattern, 0)
                score += weight
                if weight != 0:
                    sign = "+" if weight > 0 else ""
                    breakdown.append(f"{sign}{weight:.1f} {pattern.replace('_', ' ')}")
        
        # 2. Trend analysis
        if trend_analysis['trend_direction'] == 'up':
            if trend_analysis['r_squared'] > 0.7:
                score += 3
                breakdown.append("+3.0 strong uptrend")
            elif trend_analysis['r_squared'] > 0.5:
                score += 2
                breakdown.append("+2.0 moderate uptrend")
        elif trend_analysis['trend_direction'] == 'down':
            if trend_analysis['r_squared'] > 0.7:
                score -= 3
                breakdown.append("-3.0 strong downtrend")
            elif trend_analysis['r_squared'] > 0.5:
                score -= 2
                breakdown.append("-2.0 moderate downtrend")
        
        # 3. RSI analysis
        if latest['rsi'] < 30:
            score += 2
            breakdown.append("+2.0 oversold (RSI < 30)")
        elif latest['rsi'] < 40:
            score += 1
            breakdown.append("+1.0 near oversold (RSI < 40)")
        elif latest['rsi'] > 70:
            score -= 2
            breakdown.append("-2.0 overbought (RSI > 70)")
        elif latest['rsi'] > 60:
            score -= 1
            breakdown.append("-1.0 near overbought (RSI > 60)")
        
        # 4. Bollinger Band position
        if 'bb_position' in latest:
            if latest['bb_position'] < 0.2:
                score += 2
                breakdown.append("+2.0 near lower Bollinger Band")
            elif latest['bb_position'] > 0.8:
                score -= 2
                breakdown.append("-2.0 near upper Bollinger Band")
        
        # 5. MACD
        if latest['macd'] > latest['macd_signal']:
            score += 1
            breakdown.append("+1.0 bullish MACD crossover")
        elif latest['macd'] < latest['macd_signal']:
            score -= 1
            breakdown.append("-1.0 bearish MACD crossover")
        
        # 6. Support/Resistance
        if latest['close'] > latest['resistance'] * 0.999:
            score += 2
            breakdown.append("+2.0 breaking resistance")
        elif latest['close'] < latest['support'] * 1.001:
            score -= 2
            breakdown.append("-2.0 breaking support")
        
        return {
            'total_score': round(score, 2),
            'breakdown': breakdown
        }

    def generate_trading_signal(self, df, patterns, trend_analysis, signal_score):
        """Generate final trading signal"""
        latest = df.iloc[-1]
        score = signal_score['total_score']
        
        # Determine signal based on score
        if score >= 8:
            signal = 'STRONG_BUY'
            confidence = min(95, 75 + (score - 8) * 2.5)
        elif score >= 5:
            signal = 'BUY'
            confidence = min(90, 65 + (score - 5) * 3)
        elif score >= 2:
            signal = 'WEAK_BUY'
            confidence = min(75, 55 + (score - 2) * 5)
        elif score <= -8:
            signal = 'STRONG_SELL'
            confidence = min(95, 75 + (abs(score) - 8) * 2.5)
        elif score <= -5:
            signal = 'SELL'
            confidence = min(90, 65 + (abs(score) - 5) * 3)
        elif score <= -2:
            signal = 'WEAK_SELL'
            confidence = min(75, 55 + (abs(score) - 2) * 5)
        else:
            signal = 'HOLD'
            confidence = max(30, 50 - abs(score) * 2)
        
        # Generate reason text
        reasons = []
        
        # Add pattern reasons
        detected_patterns = [p for p, d in patterns.items() if d]
        if detected_patterns:
            pattern_names = [p.replace('_', ' ') for p in detected_patterns]
            reasons.append(f"Patterns: {', '.join(pattern_names)}")
        
        # Add trend reason
        if trend_analysis['trend_strength'] != 'neutral':
            reasons.append(f"{trend_analysis['trend_strength'].capitalize()} {trend_analysis['trend_direction']}trend")
        
        # Add RSI reason
        if latest['rsi'] < 35:
            reasons.append(f"Oversold (RSI: {latest['rsi']:.1f})")
        elif latest['rsi'] > 65:
            reasons.append(f"Overbought (RSI: {latest['rsi']:.1f})")
        
        # Add Bollinger Band reason
        if 'bb_position' in latest:
            if latest['bb_position'] < 0.25:
                reasons.append("Near lower Bollinger Band")
            elif latest['bb_position'] > 0.75:
                reasons.append("Near upper Bollinger Band")
        
        reason = ". ".join(reasons) if reasons else "Neutral market conditions"
        
        # Get hold period
        hold_periods = {
            '5min': '5-15 minutes',
            '15min': '15-30 minutes',
            '30min': '30-60 minutes',
            '1h': '1-2 hours',
            '4h': '4-8 hours',
            '1day': '1-2 days'
        }
        
        return {
            'signal': signal.split('_')[-1],
            'signal_strength': signal,
            'confidence': round(float(confidence), 1),
            'score': score,
            'reason': reason,
            'score_breakdown': signal_score['breakdown'],
            'patterns': patterns,
            'trend_analysis': trend_analysis,
            'current_price': round(float(latest['close']), 5),
            'pip_targets': self.risk_params.copy(),
            'hold_period': hold_periods.get('15min', '15-30 minutes')
        }

    def analyze_pair(self, symbol, timeframe, quick_mode=False):
        """Complete analysis pipeline"""
        try:
            clean_symbol = symbol.replace("=X", "").upper()
            
            # Validate
            if clean_symbol not in self.available_pairs:
                return self._get_default_signal(clean_symbol, timeframe)
            
            if timeframe not in self.timeframe_map:
                return self._get_default_signal(clean_symbol, timeframe)
            
            print(f"\n🔍 Analyzing {clean_symbol} ({timeframe})...")
            
            # Fetch data
            df = self.fetch_market_data(clean_symbol, timeframe, use_backup=True)
            
            if df is None or len(df) < 20:
                return self._get_default_signal(clean_symbol, timeframe)
            
            print(f"✅ Data: {len(df)} bars | Price: {df['close'].iloc[-1]:.5f}")
            
            # Calculate indicators
            df_features = self.calculate_technical_indicators(df)
            
            if len(df_features) < 20:
                return self._get_default_signal(clean_symbol, timeframe)
            
            # Detect patterns
            patterns = self.detect_candlestick_patterns(df_features)
            detected = [p for p, d in patterns.items() if d]
            if detected:
                print(f"✅ Patterns: {', '.join(detected[:3])}")
            
            # Trend analysis
            window = self.ai_settings["quick_trade_regression_window"] if quick_mode else self.ai_settings["regression_window"]
            trend_analysis = self.calculate_trend_analysis(df_features, window)
            print(f"✅ Trend: {trend_analysis['trend_direction']} ({trend_analysis['trend_strength']})")
            
            # Calculate signal
            signal_score = self.calculate_signal_score(df_features, patterns, trend_analysis)
            signal = self.generate_trading_signal(df_features, patterns, trend_analysis, signal_score)
            
            # Add metadata
            signal.update({
                "symbol": clean_symbol,
                "timeframe": timeframe,
                "quick_mode": quick_mode,
                "data_points": len(df_features),
                "data_source": "Twelve Data API",
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            print(f"✅ Signal: {signal['signal']} ({signal['confidence']}% confidence)")
            
            return signal
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return self._get_default_signal(symbol.replace("=X", ""), timeframe)

    def _get_default_signal(self, symbol, timeframe):
        """Return default signal"""
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': 'HOLD',
            'signal_strength': 'HOLD',
            'confidence': 50.0,
            'score': 0,
            'reason': 'Insufficient data for analysis',
            'score_breakdown': [],
            'patterns': {},
            'trend_analysis': {'trend_direction': 'neutral', 'trend_strength': 'neutral'},
            'current_price': 0.0,
            'pip_targets': self.risk_params.copy(),
            'hold_period': '15-30 minutes',
            'data_source': 'Default',
            'analysis_timestamp': datetime.now().isoformat()
        }

    def generate_quick_signals(self, count=6):
        """Generate quick trade signals"""
        quick_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        quick_timeframes = ["5min", "15min", "30min"]
        
        signals = []
        
        print(f"\n🚀 Generating {count} quick signals...")
        
        for pair in quick_pairs:
            for tf in quick_timeframes:
                try:
                    if len(signals) >= count:
                        break
                    
                    signal = self.analyze_pair(pair, tf, quick_mode=True)
                    signals.append(signal)
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    continue
            
            if len(signals) >= count:
                break
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"✅ Generated {len(signals)} signals")
        
        return signals[:count]

    def test_api_connection(self):
        """Test Twelve Data API connection"""
        print("\n🔌 Testing Twelve Data API connection...")
        test_data = self.fetch_twelvedata('EURUSD', '15min', 5)
        
        if test_data is not None and len(test_data) > 0:
            return {
                'success': True,
                'message': 'API connection successful',
                'data_points': len(test_data),
                'latest_price': float(test_data['close'].iloc[-1]) if 'close' in test_data.columns else None,
                'api_key': self.api_key
            }
        else:
            return {
                'success': False,
                'message': 'API connection failed',
                'api_key': self.api_key
            }
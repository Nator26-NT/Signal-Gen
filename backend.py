"""
Forex AI Signal Generator - Flask Backend
Using Twelve Data API
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from ai_logic import ForexAISignalGenerator
from datetime import datetime

app = Flask(__name__, template_folder='templates')
CORS(app)

# Initialize with your API
signal_gen = ForexAISignalGenerator()

# Track API usage
api_stats = {
    'total_requests': 0,
    'successful_api_calls': 0,
    'failed_api_calls': 0,
    'synthetic_data_used': 0,
    'start_time': datetime.now().isoformat()
}

@app.route('/')
def home():
    """Render main interface"""
    return render_template('index.html')

@app.route('/api/signal/<string:symbol>/<string:timeframe>', methods=['GET'])
def get_signal(symbol, timeframe):
    """Get signal using Twelve Data API"""
    api_stats['total_requests'] += 1
    
    try:
        mode = request.args.get('mode', 'standard')
        quick_mode = mode in ['quick', 'rapid']
        
        # Get signal
        signal = signal_gen.analyze_pair(symbol, timeframe, quick_mode)
        
        # Track data source
        if signal.get('data_source') == 'Twelve Data API':
            api_stats['successful_api_calls'] += 1
        elif signal.get('data_source') == 'Synthetic':
            api_stats['synthetic_data_used'] += 1
        else:
            api_stats['failed_api_calls'] += 1
        
        return jsonify({
            'success': True,
            'data': signal,
            'api_info': {
                'key_used': signal_gen.api_key[:10] + '...',
                'base_url': signal_gen.base_url,
                'data_source': signal.get('data_source', 'Unknown')
            },
            'timestamp': datetime.now().isoformat(),
            'request_id': f"req_{api_stats['total_requests']:06d}"
        })
        
    except Exception as e:
        api_stats['failed_api_calls'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'signal': 'HOLD',
            'confidence': 50,
            'api_key': signal_gen.api_key[:10] + '...'
        }), 500

@app.route('/api/signals/quick', methods=['GET'])
def get_quick_signals():
    """Get quick signals using Twelve Data API"""
    api_stats['total_requests'] += 1
    
    try:
        count = request.args.get('count', 6, type=int)
        count = min(max(count, 1), 12)
        
        signals = signal_gen.generate_quick_signals(count)
        
        # Track data sources
        for signal in signals:
            if signal.get('data_source') == 'Twelve Data API':
                api_stats['successful_api_calls'] += 1
            elif signal.get('data_source') == 'Synthetic':
                api_stats['synthetic_data_used'] += 1
        
        return jsonify({
            'success': True,
            'count': len(signals),
            'signals': signals,
            'api_info': {
                'key_used': signal_gen.api_key[:10] + '...',
                'requests_made': len(signals)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        api_stats['failed_api_calls'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'signals': [],
            'count': 0
        }), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test Twelve Data API connection"""
    api_stats['total_requests'] += 1
    
    test_result = signal_gen.test_api_connection()
    
    if test_result['success']:
        api_stats['successful_api_calls'] += 1
    else:
        api_stats['failed_api_calls'] += 1
    
    return jsonify({
        'api_test': test_result,
        'app_stats': api_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics"""
    return jsonify({
        'api_stats': api_stats,
        'api_config': {
            'key': signal_gen.api_key[:10] + '...',
            'base_url': signal_gen.base_url,
            'rate_limit': f"{signal_gen.rate_limit_delay}s",
            'available_pairs': len(signal_gen.available_pairs),
            'available_timeframes': len(signal_gen.timeframe_map)
        },
        'risk_parameters': signal_gen.risk_params,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api_key': 'configured' if signal_gen.api_key else 'missing',
        'uptime': str(datetime.now() - datetime.fromisoformat(api_stats['start_time'])),
        'requests_served': api_stats['total_requests'],
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/signal/{pair}/{timeframe}',
            '/api/signals/quick',
            '/api/test',
            '/api/stats',
            '/api/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'api_key': signal_gen.api_key[:10] + '...'
    }), 500

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        FOREX AI SIGNAL GENERATOR v2.0                    ║
    ║        Twelve Data API Integration                       ║
    ║                                                          ║
    ║        🚀 Using API Key: fe6aec0e85244251ab5cb28263f98bd6║
    ║        🌐 Base URL: https://api.twelvedata.com/time_series║
    ║        📊 Fixed Risk: 7 SL / 10 TP (1:1.43)              ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    📍 Starting server on http://127.0.0.1:5000
    📍 Press Ctrl+C to stop
    
    Endpoints:
    • GET /                    - Web interface
    • GET /api/signal/EURUSD/15min
    • GET /api/signals/quick?count=6
    • GET /api/test            - Test API connection
    • GET /api/stats           - API usage statistics
    • GET /api/health          - Health check
    
    """
    print(banner)

if __name__ == '__main__':
    print_banner()
    
    # Test API connection on startup
    print("🔌 Testing Twelve Data API connection...")
    test_result = signal_gen.test_api_connection()
    
    if test_result['success']:
        print(f"✅ API Connection Successful!")
        print(f"   Latest EURUSD: {test_result.get('latest_price', 'N/A')}")
    else:
        print("⚠ API Connection Failed - Using synthetic data fallback")
        print("   Note: Real-time data requires working API connection")
    
    print("\n" + "="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')
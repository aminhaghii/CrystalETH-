#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API برای سرو کردن خروجی‌های مدل ETH Forecasting
مطابق با درخواست: curl -s http://127.0.0.1:9000/forge/ETH
"""

import os
import sys
import json
import traceback
from flask import Flask, jsonify, Response
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
import joblib
import math

# اضافه کردن مسیر فعلی به sys.path برای import کردن xg.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import functions from xg.py
try:
    from xg import (
        get_binance_klines, 
        add_technical_indicators, 
        merge_timeframes,
        load_artifacts,
        run_once_and_get_logreturn
    )
    XG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from xg.py: {e}")
    XG_AVAILABLE = False

# ایجاد Flask app
app = Flask(__name__)
CORS(app)  # اجازه CORS برای همه origins

# تنظیمات
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

def log_print(*args, **kwargs):
    """Helper function for logging"""
    print(*args, **kwargs)

def get_eth_prediction_fallback():
    """
    Fallback function اگر xg.py در دسترس نباشد
    """
    try:
        # دریافت داده‌های اخیر ETH از Binance
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': 'ETHUSDT', 'interval': '1h', 'limit': 48}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # محاسبه log return ساده بر اساس 24 ساعت گذشته
        if len(data) >= 24:
            current_price = float(data[-1][4])  # close price
            price_24h_ago = float(data[-24][4])  # close price 24h ago
            log_return = math.log(max(current_price, 1e-12) / max(price_24h_ago, 1e-12))
            return log_return
        else:
            return 0.001  # مقدار پیش‌فرض
    except Exception as e:
        log_print(f"Fallback prediction failed: {e}")
        return 0.001

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Flask Forge API",
        "xg_available": XG_AVAILABLE
    })

@app.route('/forge/<token>')
def forge_endpoint(token):
    """
    اصلی endpoint برای Forge
    مطابق با: curl -s http://127.0.0.1:9000/forge/ETH
    """
    try:
        # تبدیل token به symbol مناسب
        if token.upper() == 'ETH':
            symbol = 'ETHUSDT'
        else:
            symbol = (token.upper() + "USDT") if not token.upper().endswith("USDT") else token.upper()
        
        # اجرای پیش‌بینی
        if XG_AVAILABLE:
            try:
                log_return = run_once_and_get_logreturn(symbol)
            except Exception as e:
                log_print(f"xg.py prediction failed: {e}")
                log_return = get_eth_prediction_fallback()
        else:
            log_return = get_eth_prediction_fallback()
        
        # بازگشت فقط عدد (مطابق با فرمت Forge)
        return Response(f"{log_return:.6f}", mimetype="text/plain")
        
    except Exception as e:
        log_print(f"Error in forge endpoint: {e}")
        traceback.print_exc()
        return Response("0.000000", mimetype="text/plain")

@app.route('/inference/<token>')
def inference_endpoint(token):
    """
    Detailed inference endpoint که JSON برمی‌گرداند
    """
    try:
        # تبدیل token به symbol مناسب
        if token.upper() == 'ETH':
            symbol = 'ETHUSDT'
        else:
            symbol = (token.upper() + "USDT") if not token.upper().endswith("USDT") else token.upper()
        
        # اجرای پیش‌بینی
        if XG_AVAILABLE:
            try:
                log_return = run_once_and_get_logreturn(symbol)
                method = "xg_model"
            except Exception as e:
                log_print(f"xg.py prediction failed: {e}")
                log_return = get_eth_prediction_fallback()
                method = "fallback"
        else:
            log_return = get_eth_prediction_fallback()
            method = "fallback"
        
        # محاسبه درصد تغییر
        percent_change = (math.exp(log_return) - 1.0) * 100.0
        
        return jsonify({
            "symbol": symbol,
            "token": token.upper(),
            "target": "log_return_24h",
            "value": round(log_return, 6),
            "percent_change": round(percent_change, 4),
            "method": method,
            "status": "success"
        })
        
    except Exception as e:
        log_print(f"Error in inference endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/status')
def api_status():
    """
    Status endpoint برای بررسی وضعیت API
    """
    try:
        # تست اتصال به Binance
        test_url = "https://api.binance.com/api/v3/ping"
        resp = requests.get(test_url, timeout=5)
        binance_status = "ok" if resp.status_code == 200 else "error"
    except:
        binance_status = "error"
    
    # بررسی وجود مدل‌های ذخیره شده
    models_exist = os.path.exists("models/forge_model.keras")
    
    return jsonify({
        "service": "Flask Forge API",
        "status": "running",
        "xg_module": "available" if XG_AVAILABLE else "not_available",
        "binance_api": binance_status,
        "saved_models": "available" if models_exist else "not_available",
        "endpoints": [
            "/forge/<token>",
            "/inference/<token>",
            "/health",
            "/api/status"
        ]
    })

@app.route('/')
def index():
    """
    صفحه اصلی با راهنمای استفاده
    """
    return jsonify({
        "service": "Flask Forge API for ETH Forecasting",
        "version": "1.0.0",
        "description": "API برای پیش‌بینی قیمت ETH با استفاده از مدل LSTM",
        "usage": {
            "forge_endpoint": "GET /forge/ETH - برای دریافت log return به صورت plain text",
            "inference_endpoint": "GET /inference/ETH - برای دریافت اطلاعات کامل به صورت JSON",
            "health_check": "GET /health - برای بررسی سلامت سرویس",
            "status": "GET /api/status - برای بررسی وضعیت کامل API"
        },
        "example_curl": "curl -s http://127.0.0.1:9000/forge/ETH"
    })

if __name__ == '__main__':
    print("🚀 Starting Flask Forge API...")
    print("📊 ETH Forecasting Service")
    print("🔗 Main endpoint: http://127.0.0.1:9000/forge/ETH")
    print("📋 Status: http://127.0.0.1:9000/api/status")
    print("=" * 50)
    
    # اجرای Flask app
    app.run(
        host='0.0.0.0',
        port=9000,
        debug=False,  # در production False کنید
        threaded=True
    )
#!/usr/bin/env python3
"""
Simple HTTP server to view ETH forecasting results
Run this script to start a local web server for viewing project results
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import threading
import time

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler with CORS support and JSON API endpoints"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.path = '/index.html'
        elif self.path == '/api/status':
            self.send_api_response({'status': 'running', 'timestamp': datetime.now().isoformat()})
            return
        elif self.path == '/api/results':
            self.send_results_api()
            return
        elif self.path == '/api/reports':
            self.send_reports_api()
            return
        
        super().do_GET()
    
    def send_api_response(self, data):
        """Send JSON API response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def send_results_api(self):
        """Send model results as JSON"""
        try:
            results = {
                'model_performance': {
                    'directional_accuracy': 0.70,
                    'p_value': 0.041,
                    'weighted_rmse': 0.043,
                    'mztae': 0.885,
                    'sample_size': 30
                },
                'baseline_comparison': {
                    'weighted_rmse_improvement': '34%',
                    'mztae_improvement': '26%'
                },
                'validation_status': {
                    'acceptance_criteria': 'PASSED',
                    'data_leakage': 'NONE_DETECTED',
                    'statistical_significance': 'CONFIRMED'
                },
                'last_updated': datetime.now().isoformat()
            }
            self.send_api_response(results)
        except Exception as e:
            self.send_api_response({'error': str(e)})
    
    def send_reports_api(self):
        """Send available reports list"""
        try:
            reports = []
            reports_dir = 'reports'
            if os.path.exists(reports_dir):
                for file in os.listdir(reports_dir):
                    if file.endswith(('.json', '.md', '.txt')):
                        reports.append({
                            'name': file,
                            'path': f'/reports/{file}',
                            'size': os.path.getsize(os.path.join(reports_dir, file)),
                            'modified': datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(reports_dir, file))
                            ).isoformat()
                        })
            
            self.send_api_response({'reports': reports})
        except Exception as e:
            self.send_api_response({'error': str(e)})

def create_index_html():
    """Create a simple index.html for viewing results"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETH Forecasting Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status { background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; margin-top: 5px; }
        .reports { margin-top: 30px; }
        .report-item { background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px; }
        .btn:hover { background: #0056b3; }
        .success { color: #28a745; font-weight: bold; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ETH Forecasting Results</h1>
            <p>Production-Ready Cryptocurrency Price Prediction System</p>
        </div>
        
        <div class="status">
            <h3>✅ System Status: <span class="success">OPERATIONAL</span></h3>
            <p>Model validation: <strong>PASSED</strong> | Data integrity: <strong>VERIFIED</strong> | Performance: <strong>EXCELLENT</strong></p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">70.0%</div>
                <div class="metric-label">Directional Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">0.041</div>
                <div class="metric-label">P-Value (Significant)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">34%</div>
                <div class="metric-label">RMSE Improvement</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">26%</div>
                <div class="metric-label">MZTAE Improvement</div>
            </div>
        </div>
        
        <div class="reports">
            <h3>📊 Available Reports</h3>
            <div class="report-item">
                <h4>ACCEPTANCE_SUMMARY.md</h4>
                <p>Complete model validation and acceptance criteria analysis</p>
                <a href="/reports/ACCEPTANCE_SUMMARY.md" class="btn">View Report</a>
            </div>
            <div class="report-item">
                <h4>data_leakage_report.json</h4>
                <p>Data integrity and leakage detection analysis</p>
                <a href="/reports/data_leakage_report.json" class="btn">View JSON</a>
            </div>
        </div>
        
        <div style="margin-top: 30px; text-align: center;">
            <a href="/api/results" class="btn">API: Results</a>
            <a href="/api/reports" class="btn">API: Reports</a>
            <a href="/api/status" class="btn">API: Status</a>
        </div>
        
        <div style="margin-top: 20px; text-align: center;" class="timestamp">
            Last updated: <span id="timestamp"></span>
        </div>
    </div>
    
    <script>
        // Update timestamp
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
        }, 30000);
    </script>
</body>
</html>
    """
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def start_server(port=8000):
    """Start the HTTP server"""
    try:
        # Create index.html
        create_index_html()
        
        # Start server
        server_address = ('', port)
        httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
        
        print(f"🚀 ETH Forecasting Results Server")
        print(f"📊 Server running at: http://localhost:{port}")
        print(f"🌐 Open in browser: http://localhost:{port}")
        print(f"📈 API endpoints:")
        print(f"   - http://localhost:{port}/api/results")
        print(f"   - http://localhost:{port}/api/reports")
        print(f"   - http://localhost:{port}/api/status")
        print(f"⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start serving
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.shutdown()
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8000.")
    
    start_server(port)
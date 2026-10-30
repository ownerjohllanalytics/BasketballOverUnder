from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Requests so Squarespace can communicate with this API

@app.route('/')
def home():
    return jsonify({"status": "API is running!"})

@app.route('/run-script', methods=['GET', 'POST'])
def run_script():
    # --- PUT YOUR PYTHON LOGIC HERE ---
    # Example: Processing input or generating a result
    result_message = "Python code executed successfully!"
    
    return jsonify({
        "status": "success",
        "result": result_message
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
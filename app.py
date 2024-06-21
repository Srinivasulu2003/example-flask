from flask import Flask, request, jsonify
from gradio_client import Client

# Initialize Flask app
app = Flask(__name__)

# Initialize Gradio client
client = Client("srinuksv/SRUNU")

# Define endpoint for webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Get the JSON data from the POST request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Extract query from JSON data
        query = data.get('queryResult', {}).get('queryText', '')
        
        if not query:
            return jsonify({'error': 'No query found in request'}), 400

        # Perform prediction using Gradio client
        result = client.predict(
            query=query,
            api_name="/predict"
        )

        # Return the result as JSON response
        return jsonify({'fulfillmentText': result})
    
    except Exception as e:
        # Log the error
        app.logger.error(f"Error processing webhook: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Define a simple route to test if the Flask app is running
@app.route('/')
def index():
    return "Flask app is running!"

# Run the Flask app

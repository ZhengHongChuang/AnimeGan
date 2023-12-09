import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from service.styleTransfer import styleTransfer
from PIL import Image  
import numpy as np
app = Flask(__name__)
CORS(app)  

@app.route('/styleTransfer', methods=['POST'])
def style_transfer_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        image = Image.open(image_file)
        numpy_array = np.array(image)
        stylized_image = styleTransfer(numpy_array)
        pil_image = Image.fromarray(stylized_image)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        pil_image.save(temp_filename)
        return send_file(
            temp_filename,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='stylized_image.jpg'
        )
    except Exception as e:
        print(f"Error during style transfer: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
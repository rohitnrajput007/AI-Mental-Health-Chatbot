from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration


app = Flask(__name__)

# Initialize the Blenderbot model and tokenizer
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = AutoTokenizer.from_pretrained(mname)

def chat(user_input):
    # Generate a response from the user's input
    inputs = tokenizer([user_input], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return reply

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['user_message']
    reply = chat(user_message)
    # SpeakText(reply)
    return jsonify({'bot_response': reply})

if __name__ == '__main__':
    app.run(debug=True)

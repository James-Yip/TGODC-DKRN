from flask import Flask,  request, make_response
import json
from target_chat_for_server import init_target_chat

Chinese_target_chat_instance = init_target_chat('neural_dkr', 'CWC')  #初始化
target_chat_instance = init_target_chat('neural_dkr', 'TGPC')  #初始化
app = Flask(__name__)



@app.route('/Chinese_chatbot_api/', methods=["POST"])
def Chinese_chatbot_api():
    data = {}
    user_input = request.form.get('userIn')
    user_input_list = json.loads(user_input)
    data['modelOut'] =Chinese_target_chat_instance.chat(user_input_list)
    data['state'] = "success"
    model_output = json.dumps(data)
    rst = make_response(model_output)
    rst.headers['Access-Control-Allow-Origin'] = '*'
    rst.headers['Access-Control-Allow-Methods'] = 'POST'# 响应POST
    return rst



@app.route('/English_chatbot_api/', methods=["POST"])
def English_chatbot_api():
    data = {}
    user_input = request.form.get('userIn')
    user_input_list = json.loads(user_input)
    data['modelOut'] =target_chat_instance.chat(user_input_list)
    data['state'] = "success"
    model_output = json.dumps(data)
    rst = make_response(model_output)
    rst.headers['Access-Control-Allow-Origin'] = '*'
    rst.headers['Access-Control-Allow-Methods'] = 'POST'# 响应POST
    return rst

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")

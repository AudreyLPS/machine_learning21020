from flask import Flask, render_template, request
from tools import model
from tools import nettoyage
from tools import entrainement
import json


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')

@app.route("/result",methods=['POST'])
def retour():
    user_text = request.form.get('input_text')
    print(user_text)
    print('user_text')
    return json.dumps({'text_user':user_text})

# renvoi une prediction en json associé au texte envoyé en paramètre d’une requête POST
@app.route("/prediction",methods=['POST'])
def prediction():
    user_text = request.form.get('input_text')
    prediction=model(user_text)
    print(prediction)
    return json.dumps({'text_user':prediction})

@app.route("/entrainement",methods=['POST'])
def route_entrainement():
    traine=entrainement()
    return json.dumps({'text_user':traine})


if __name__ == "__main__":
    app.run(debug=True)
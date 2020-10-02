from flask import Flask, render_template, request
from tools import model
from tools import nettoyage
from tools import entrainement
import json


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['POST'])
def retour():
    return render_template("prediction.html")


# renvoi une prediction en json associé au texte envoyé en paramètre d’une requête POST
@app.route("/reponse",methods=['POST'])
def prediction():
    user_text = request.form.get('input_text')
    prediction=model(user_text)
    print(prediction)
    #return json.dumps({'text_user':prediction})
    return render_template("reponse.html", input_text=user_text,prediction=prediction)


@app.route("/entrainement",methods=['POST'])
def route_entrainement():
    traine=entrainement()
    #return json.dumps({'text_user':traine})
    return render_template("trainning.html", traine=traine)


if __name__ == "__main__":
    app.run(debug=True)
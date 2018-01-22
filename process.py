from flask import Flask, render_template, request, jsonify
from withfucntions_255 import startinghere
app = Flask(__name__)

@app.route('/', methods= ["GET", "POST"])
def homepage():

    return render_template('form.html')

@app.route('/output',methods=["POST"])
def output():
    
    start=request.form['starting_value']
    review = startinghere(start)
    if review:
	return render_template('output1.html',result=review)
    else:
	return render_template('output1.html',result="Input needed")



if __name__ == "__main__":
    app.debug=True
    app.run(threaded=True)

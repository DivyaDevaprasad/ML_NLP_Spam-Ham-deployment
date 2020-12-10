from flask import Flask,render_template,url_for,request

from sklearn.naive_bayes import MultinomialNB
#import sklearn.external.joblib as extjoblib
import joblib
#from sklearn.externals import joblib
import pickle

# load the model from disk
model = pickle.load(open("nlp_model.pkl", 'rb'))
cv = pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)


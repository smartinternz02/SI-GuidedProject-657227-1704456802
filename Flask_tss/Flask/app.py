import numpy as np
import os
import tensorflow 
import keras.models
import keras.preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
from PIL import Image

app=Flask(__name__)

model=load_model("animal.h5")
#model1 = load_model("TSSDR.h5")
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        #img = Image.open(filepath)
        #img=image.load_img(filepath,target_size=(64,64))
        img1 = keras.preprocessing.image.image_utils.load_img(filepath,target_size=(64,64))
        # Convert the image to a NumPy array
        x = keras.preprocessing.image.image_utils.img_to_array(img1)
        
        #x=Image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['Diabetic','Normal']
        text="The Classified Retina is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()
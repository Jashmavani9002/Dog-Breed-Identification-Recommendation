
import numpy as np
from importlib.metadata import requires
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.shortcuts import render
from keras import models
from keras import layers

from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pyrebase
import gc
gc.collect()
import pandas as pd

import os
from itertools import chain
import random
import tensorflow
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
import pickle
import requests
from PIL import Image
import shutil
import urllib
from django.contrib import auth
import urllib.request
from PIL import Image
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
firebaseConfig = {
    "apiKey": "AIzaSyBRme34tSvxsXXGCV8oa3FqxdS-rTYYk5Q",
    "authDomain": "dogbreed-e90b2.firebaseapp.com",
    "databaseURL": "https://dogbreed-e90b2-default-rtdb.firebaseio.com",
    "projectId": "dogbreed-e90b2",
    "storageBucket": "dogbreed-e90b2.appspot.com",
    "messagingSenderId": "843710481760",
    "appId": "1:843710481760:web:1fe1f6f2aedb3fe41c4a05",
    "measurementId": "G-MZR7MM05NN"
}
firebase = pyrebase.initialize_app(firebaseConfig)

authe = firebase.auth()
database = firebase.database()



user={}
city = ""

# go back to index main page from about 
def back(request):
    global user
    uid = user['localId']
    name = database.child("users").child(uid).child("details").child('name').get().val()
    session_id = user["idToken"]
    request.session["uid"]=str(session_id
                               )
    return render(request,'index.html',{"e":name})

# login process  
def index(request):
    email= request.POST.get("uname")
    passw = request.POST.get("psw") 
    
    try:
        global user
        user = authe.sign_in_with_email_and_password(email,passw)
        print(passw)
        uid = user['localId']
        name = database.child("users").child(uid).child("details").child('name').get().val()
        print(name)
        global city
        city = database.child("users").child(uid).child("details").child('city').get().val()
        print(city)
    except:
        message = "Invalid Credentials"
        return render(request,'login.html',{"messg":message})

    session_id = user["idToken"]
    request.session["uid"]=str(session_id)
    return render(request,'index.html',{"e":name})

# sign up process 
def index_2(request):
    
    email = request.POST.get("uname")
    passw = request.POST.get("psw")
    city = request.POST.get("city")
    name = request.POST.get("name") 

    try:
        user = authe.create_user_with_email_and_password(email,passw)
        uid = user['localId']
        
        data={"Id":email,"name":name,"status":"1","city":city,"dp":"https://firebasestorage.googleapis.com/v0/b/mydogbreed-6ae22.appspot.com/o/profile.png?alt=media&token=1942831a-82fe-4c4c-b932-e2d7b7374e81"}
        database.child("users").child(uid).child("details").set(data)
        print(database)
        message = "Account Created Successfully"
    except:
        message_1 = "Account already exist or Invalid password"
        return render(request,'signup.html',{"messg":message_1})

    
    return render(request,'login.html',{"messg":message})    
    
# calling login page 
def login(request):
    return render(request,'login.html')

# logout button 
def loggedout(request):
    return render(request,'loggedout.html')

def index_3(request):
    return render(request,'index.html')

#about page 
def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')



# sign up  
@csrf_exempt
def signup(request):
    return render(request,'signup.html')



def postsign(request):
    return render(request,'index.html')

# logout 
def logout(request):
    auth.logout(request)
    return render(request,'loggedout.html')



#nearbt profile page
def nearby(request):
    global city
    global user
    uid = user['localId']
    all_user = list(database.child('users').shallow().get().val())
    i = 0
    near_user = []
    while i < len(all_user):
        if all_user[i] != uid:
            Local_city = database.child("users").child(all_user[i]).child("details").child('city').get().val()
        
            if Local_city == city:
                near_user.append(all_user[i])
                # Local_name = database.child("users").child(all_user[i]).child("details").child('name').get().val()
        i = i +1
    local_names = []
    local_emails = []
    dps = []
    for i in near_user:
        Local_name = database.child("users").child(i).child("details").child('name').get().val()
        local_names.append(Local_name)
        Local_email = database.child("users").child(i).child("details").child('Id').get().val()
        local_emails.append(Local_email)
        Local_dp = database.child("users").child(i).child("details").child('dp').get().val()
        dps.append(Local_dp)
    comb_lis = zip(local_emails,local_names,dps)

    return render(request,'nearby_profiles.html',{'comb_lis':comb_lis})


#my profile upload photos 
def myprofile(request):
    return render(request,'profile.html')

#for dp
def mydp(request):
    return render(request,'setdp.html')

def geturl(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    import random
    try:
        all_img = list(database.child('users').child(uid).child("files").shallow().get().val())
        # img_num = len(all_img) + 1
        
        
        
        x = random.randint(0,100)
        while(x in all_img):
            x = random.randint(0,100)
    except:
        print("himlo mc")
        x = random.randint(0,100)

    database.child("users").child(uid).child("files").child(x).set(url)
    return render(request,'index.html')

def myphotos(request):
    global user
    uid = user['localId']
    img_lis = []
    dp = None  # Default value in case dp is not found

    try:
        all_img = list(database.child('users').child(uid).child("files").shallow().get().val())
        
        for i in all_img:
            img = database.child('users').child(uid).child('files').child(i).get().val()
            img_lis.append(img)
        
        # dp = database.child('users').child(uid).child("details").child('dp').get().val()
        
    except TypeError as e:
        # Handle the 'NoneType' object issue
        print(f"Error: {e}")

    return render(request, 'my_photos.html', {'img_lis': img_lis})
   


#to set dp 
def setdp(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    database.child("users").child(uid).child("details").child('dp').set(url)
    return render(request,'index.html')

def classi(request):
    return render(request,'classi.html')

#recommendation page 
def recommend(request):
    return render(request,'recommend.html')


def setclassi(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    import random
    

    # import tensorflow
    from sklearn.preprocessing import LabelEncoder
    # from tensorflow.keras.utils import to_categorical

    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    # import pickle
    urllib.request.urlretrieve(url, "temp.png")
    img = "temp.png"

    # model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
    img_data = np.array([img_to_array(load_img(img, target_size = (299,299,3)))])

    # print(img_data)

    from sklearn.preprocessing import LabelEncoder

    # List of all dog breed labels
    class_labels = [
        'Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier',
        'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog',
        'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull',
        'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan',
        'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman',
        'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher',
        'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer',
        'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog',
        'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel',
        'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier',
        'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog',
        'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound',
        'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian',
        'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed',
        'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog',
        'Shih-Tzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
        'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner',
        'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier',
        'affenpinscher', 'basenji', 'basset', 'beagle', 'black-and-tan_coonhound',
        'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn',
        'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dhole',
        'dingo', 'flat-coated_retriever', 'giant_schnauzer', 'golden_retriever', 'groenendael',
        'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois',
        'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound',
        'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer', 'toy_poodle',
        'toy_terrier', 'vizsla', 'whippet', 'wire-haired_fox_terrier'
    ]

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit and transform the class labels
    encoded_labels = le.fit_transform(class_labels)

    # # Print the mapping between original class labels and encoded labels
    # label_mapping = dict(zip(class_labels, encoded_labels))
    # print(label_mapping)

    # test_predictions = model.predict(x_test1)
    # predictions = le.classes_[np.argmax(test_predictions, axis=1)]
    from tensorflow.keras.models import load_model

    # Load the model from the file
    loaded_model = load_model(r'Model\your_model.h5')
    x_test1 = img_data / 255.
    test_predictions = loaded_model.predict(x_test1)
    # print(test_predictions)
    # Assuming you have a LabelEncoder instance named le
    predictions = le.classes_[np.argmax(test_predictions, axis=1)]

    # print(predictions)
    name = predictions[0].upper().replace("_"," ")

    csv_file_path = 'labels.csv'  # Replace with the actual path to your CSV file
    dog_data = pd.read_csv(csv_file_path)
    predicted_dog_data = dog_data[dog_data['Name'] == predictions[0].lower()]
    link = "https://en.wikipedia.org/wiki/" + predictions[0]

    try:
        all_img = list(database.child('users').child(uid).child("files").shallow().get().val())

        x = random.randint(0, 100)
        while x in all_img:
            x = random.randint(0, 100)
    except:
        print("An error occurred while fetching data from the database.")
        x = random.randint(0, 100)

    # Save the image URL and predicted breed name in the database
    database.child("users").child(uid).child("files").child(x).set({"url": url, "predicted_breed": name})


    return render(request, 'predict.html', {
        'name': name,
        'dg': url,
        'lk': link,
        'dog_data': predicted_dog_data.to_dict('records'),
      # Add this line to pass the recommendations to the template
    })

    # return render(request,'predict.html',{'url':url}

def setrecommend(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    import random
    try:
        all_img = list(database.child('users').child(uid).child("files").shallow().get().val())
        
        x = random.randint(0,100)
        while(x in all_img):
            x = random.randint(0,100)
    except:
        print("himlo mc")
        x = random.randint(0,100)

    database.child("users").child(uid).child("files").child(x).set(url)


    # import tensorflow
    from sklearn.preprocessing import LabelEncoder
    # from tensorflow.keras.utils import to_categorical

    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    # import pickle
    urllib.request.urlretrieve(url, "temp.png")
    img = "temp.png"

    # model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
    img_data = np.array([img_to_array(load_img(img, target_size = (299,299,3)))])

    # print(img_data)

    from sklearn.preprocessing import LabelEncoder

    # List of all dog breed labels
    class_labels = [
        'Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier',
        'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog',
        'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull',
        'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan',
        'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman',
        'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher',
        'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer',
        'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog',
        'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel',
        'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier',
        'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog',
        'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound',
        'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian',
        'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed',
        'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog',
        'Shih-Tzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
        'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner',
        'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier',
        'affenpinscher', 'basenji', 'basset', 'beagle', 'black-and-tan_coonhound',
        'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn',
        'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dhole',
        'dingo', 'flat-coated_retriever', 'giant_schnauzer', 'golden_retriever', 'groenendael',
        'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois',
        'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound',
        'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer', 'toy_poodle',
        'toy_terrier', 'vizsla', 'whippet', 'wire-haired_fox_terrier'
    ]

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit and transform the class labels
    encoded_labels = le.fit_transform(class_labels)

    # # Print the mapping between original class labels and encoded labels
    # label_mapping = dict(zip(class_labels, encoded_labels))
    # print(label_mapping)

    # test_predictions = model.predict(x_test1)
    # predictions = le.classes_[np.argmax(test_predictions, axis=1)]
    from tensorflow.keras.models import load_model

    # Load the model from the file
    loaded_model = load_model(r'Model\your_model.h5')
    x_test1 = img_data / 255.
    test_predictions = loaded_model.predict(x_test1)
    # print(test_predictions)
    # Assuming you have a LabelEncoder instance named le
    predictions = le.classes_[np.argmax(test_predictions, axis=1)]

    # print(predictions)
    name = predictions[0].upper().replace("_"," ")


    df = pd.read_csv("akc-data-latest.csv")

    df.columns.tolist()

    df['group'].unique().tolist()

    df = df.rename(columns={'Unnamed: 0': 'breed'})

    df = df.dropna()
    df['group'].unique().tolist()

    for col in [col for col in df.columns if 'value' in col]:
        df[('high_'+col).replace('_value','')] = df[col].apply(lambda x: x >= .8)
        df[('medium_'+col).replace('_value','')] = df[col].apply(lambda x: .4 <= x <= .8)
        df[('low_'+col).replace('_value','')] = df[col].apply(lambda x: x <= .4)

        df[col] = df[col].apply(lambda x: [x,0])

    df['shedding_category']

    for col in ['height','weight','expectancy']:
        df[col] = (df['max_'+col] + df['min_'+col])/2

    for col in ['height','weight','expectancy']:
        temp = df[col].describe(percentiles=[.2,.33,.4,.6,.67,.8])
        df['high_'+col] = df[col].apply(lambda x: x > temp['67%'])
        df['medium_'+col] = df[col].apply(lambda x: temp['33%'] < x < temp['67%'])
        df['low_'+col] = df[col].apply(lambda x: x < temp['33%'])

        df[col+'_value'] = df[col].apply(lambda x: '1' if x >= temp['80%'] else x)
        df[col+'_value'] = df[col+'_value'].apply(lambda x: '.8' if ((type(x)!=str) and (x >= temp['60%']) and (x < temp['80%'])) else x)
        df[col+'_value'] = df[col+'_value'].apply(lambda x: '.6' if ((type(x)!=str) and (x >= temp['40%']) and (x < temp['60%'])) else x)
        df[col+'_value'] = df[col+'_value'].apply(lambda x: '.4' if ((type(x)!=str) and (x >= temp['20%']) and (x < temp['40%'])) else x)
        df[col+'_value'] = df[col+'_value'].apply(lambda x: '.2' if ((type(x)!=str) and (x < temp['20%'])) else x)
        df[col+'_value'] = df[col+'_value'].apply(lambda x: [float(x),0])

    df['max_weight']

    output_cols =['group','temperment'] + [col for col in df.columns if any ([substr in col for substr in ['min_','max_','category']])]

    df

    df['temperament list'] = df['temperament'].apply(lambda x: x.split(',') if type(x)==str else [])
    temperament = []
    for i in df['temperament list']:
        temperament.extend(i)
    temperament_no_repeats = set(temperament)
    df['one-hot temperament'] = df['temperament list'].apply(lambda x: [int(temperament in x) for temperament in temperament_no_repeats])

    group_no_repeats = df['group'].unique()
    df['one-hot group'] = df['group'].apply(lambda x: [int(group in x) for group in group_no_repeats])

    from sklearn.metrics.pairwise import euclidean_distances
    from  sklearn.metrics.pairwise import cosine_similarity
    def recommend_similar_dogs(breed,group=[],low=[],medium=[],high=[],ignore=[],important=[]):
        if type(group) == str:
            group = [group]
        if type(low) == str:
            low = [low]
        if type(medium) == str:
            medium = [medium]
        if type(high) == str:
            high = [high]
        if type(ignore) == str:
            ignore = [ignore]

        temp_cols = list(set(df.columns) - set(ignore))
        temp = df[temp_cols]
        if len(group) > 0:
            temp = temp[(temp['breed']==breed)|(temp['group'].isin(group))]
        if len(low) > 0:
            for col in low:
                temp = temp[(temp['breed']==breed)|(temp['low_'+col])]
        if len(medium) > 0:
            for col in medium:
                temp = temp[(temp['breed']==breed)|(temp['medium_'+col])]
        if len(high) > 0:
            for col in high:
                temp = temp[(temp['breed']==breed)|(temp['high_'+col])]
        temp = temp.reset_index(drop=True)

        sims = np.zeros([len(temp),len(temp)])
        for col in [col for col in temp.columns if 'value' in col]:
            if col in important:
                sims += 5*(1-np.array(euclidean_distances(temp[col].tolist(),temp[col].tolist())))
            else:
                sims += (1-np.array(euclidean_distances(temp[col].tolist(),temp[col].tolist())))

        for col in ['one-hot temperament','one-hot group']:
            if col in important:
                sims += 5*np.array(cosine_similarity(temp[col].tolist(),temp[col].tolist()))
            else:
                sims += np.array(cosine_similarity(temp[col].tolist(),temp[col].tolist()))

        idx = temp[temp['breed']==breed].index
        sims = list(enumerate(sims[idx][0]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        num_dogs = min(10,len(temp))
        sims = sims[:num_dogs+1]
        breed_indices = [i[0] for i in sims]

        n = 0
        for i in breed_indices:
            if n == 0:
                print('Selected:'.format(n),temp['breed'].iloc[i])
            else:
                print('{}.'.format(n),temp['breed'].iloc[i])
            n += 1

        breed_names = []
        image_urls = []

        for i in breed_indices:
            breed_name = temp['breed'].iloc[i]
            breed_names.append(breed_name)
            image_url = df[df['breed'] == breed_name]['Image'].values[0]
            image_urls.append(image_url)

        return list(zip(breed_names, image_urls))


    name = name.title()
    name = name.strip()
    recommendations = recommend_similar_dogs(name,high = ['demeanor','trainability'], important = ['group','height','weight'])

    

    return render(request, 'recommendation.html', {
        'name': name,
        'recommendations': recommendations
      # Add this line to pass the recommendations to the template
    })


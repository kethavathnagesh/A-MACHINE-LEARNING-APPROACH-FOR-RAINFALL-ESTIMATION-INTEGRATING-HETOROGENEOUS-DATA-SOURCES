
from django.shortcuts import render, redirect, get_object_or_404

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,rainfall_estimation,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Rainfall_Estimate_Prediction_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Date= request.POST.get('Date')
            Location= request.POST.get('Location')
            MinTemp= request.POST.get('MinTemp')
            MaxTemp= request.POST.get('MaxTemp')
            Rainfall= request.POST.get('Rainfall')
            Evaporation= request.POST.get('Evaporation')
            Sunshine= request.POST.get('Sunshine')
            WindGustDir= request.POST.get('WindGustDir')
            WindGustSpeed= request.POST.get('WindGustSpeed')
            WindDir= request.POST.get('WindDir')
            WindSpeed= request.POST.get('WindSpeed')
            Humidity= request.POST.get('Humidity')
            Pressure= request.POST.get('Pressure')
            Cloud= request.POST.get('Cloud')
            Temp= request.POST.get('Temp')
            idnumber= request.POST.get('idnumber')



        data = pd.read_csv("Raifall_Datasets.csv", encoding='latin-1')

        # Creating a mapping for sentiments
        def apply_results(results):
            if (results == 'No'):
                return 0
            elif (results == 'Yes'):
                return 1

        data['Results'] = data['RainTomorrow'].apply(apply_results)

        x = data['idnumber']
        y = data['Results']
        # cv = CountVectorizer()
        # x = cv.fit_transform(x)

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        x = cv.fit_transform(data['idnumber'].apply(lambda x: np.str_(x)))
        #x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")
        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))


        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))


        print("Logistic Regression")
        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        print("SGD Classifier")
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
        sgd_clf.fit(X_train, y_train)
        sgdpredict = sgd_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, sgdpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, sgdpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, sgdpredict))
        models.append(('SGDClassifier', sgd_clf))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        idnumber = [idnumber]
        vector1 = cv.transform(idnumber).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Rainfall'
        elif prediction == 1:
            val = 'Heavy Rainfall'

        print(prediction)
        print(val)

        rainfall_estimation.objects.create(Date1=Date,
        Location=Location,
        MinTemp=MinTemp,
        MaxTemp=MaxTemp,
        Rainfall=Rainfall,
        Evaporation=Evaporation,
        Sunshine=Sunshine,
        WindGustDir=WindGustDir,
        WindGustSpeed=WindGustSpeed,
        WindDir=WindDir,
        WindSpeed=WindSpeed,
        Humidity=Humidity,
        Pressure=Pressure,
        Cloud=Cloud,
        Temp=Temp,
        idnumber=idnumber,
        Prediction=val)

        return render(request, 'RUser/Rainfall_Estimate_Prediction_Type.html',{'objs': val})
    return render(request, 'RUser/Rainfall_Estimate_Prediction_Type.html')




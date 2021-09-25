# Django Imports
from django.shortcuts import render
from django.views import View

# Internal Imports
from . import mongodb as mdb
from .my_decorator import AdminStaffRequiredMixin

# Python Package Imports
import pickle
import pandas as pd
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from bson.binary import Binary
from base64 import b64encode
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({})
db_data = mdb.access()


# Home Module Starts
class Home(AdminStaffRequiredMixin, View):

    template_name = 'admin_classifier/home.html'
    context = {}

    def get(self, request):
        return render(request, self.template_name, self.context)
# Home Module Ends


# Simple Factory Pattern Starts:
class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass

    @staticmethod
    def description_update(algo_desc, ds_desc, algo_name):
        update_data = {'algo_desc': str(algo_desc), 'ds_desc': str(ds_desc)}
        update_message = mdb.update(db_data, algo_name, update_data, "Descriptions Updated",
                                    "Unexpected error while updating descriptions")
        return update_message

    @staticmethod
    def pkl_upload(upload_file, algo_name):
        if upload_file.content_type == 'application/octet-stream':
            pkl_obj = pickle.dumps(upload_file)
            mongo_data = {'pkl_data': Binary(pkl_obj), 'upload_method': 'pkl'}
            pkl_message = mdb.update(db_data, algo_name, mongo_data, "File Uploaded",
                                     "Unexpected error while uploading pickle file")
        else:
            pkl_message = "Invalid File Type"
        return pkl_message

    @staticmethod
    def pkl_change(pkl_features_temp, algo_name, pkl_label_notes_temp=None):
        pkl_features = []
        for feature in pkl_features_temp:
            pkl_features.append(feature.strip())
        if algo_name != "MLR":
            pkl_label_notes = {}
            for label in pkl_label_notes_temp:
                temp = label.split("=")
                pkl_label_notes[temp[0].strip()] = temp[1].strip()
            mongo_data = {'label_notes': pkl_label_notes, 'training_features': pkl_features}
        else:
            mongo_data = {'training_features': pkl_features}
        pkl_change_message = mdb.update(db_data, algo_name, mongo_data, "Success", "Error")
        return pkl_change_message

    @staticmethod
    def graph_upload(image_file, algo_name):
        if image_file.content_type == 'image/png' or image_file.content_type == 'image/jpeg' or image_file.content_type == 'image/jpg':
            image_data = image_file.read()
            encoded_image = str(b64encode(image_data))[2:-1]
            mime = str(image_file.content_type)
            mime = mime + ';' if mime else ';'
            graph_image = "data:%sbase64,%s" % (mime, encoded_image)
            mongo_data = {'graph_image': graph_image}
            graph_message = mdb.update(db_data, algo_name, mongo_data, "Success", "Error")
        else:
            graph_message = "Invalid Image Type"
        return graph_message

    @staticmethod
    def csv_upload(upload_file):
        global data
        feature_list = []
        if upload_file.content_type == 'application/vnd.ms-excel' and upload_file.size < 50000:
            string_data = StringIO(upload_file.read().decode('utf-8'))
            data = pd.read_csv(string_data)
            feature_list = list(data.columns)
            csv_message = "File Uploaded"
        else:
            if upload_file.content_type != 'application/vnd.ms-excel':
                csv_message = "Invalid File Type"
            else:
                csv_message = "Maximum size of the CSV file should be less than 50kb"
        return feature_list, csv_message

    @staticmethod
    def scale(X_train, X_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return pickle.dumps(sc), X_train, X_test


class Classification(AdminStaffRequiredMixin, Algorithm):

    template_name = 'admin_classifier/classification.html'
    context = {}
    feature_list = []
    submit_button = None
    message = None
    pkl_message = None
    pkl_change_message = None
    csv_message = None
    update_message = None
    accuracy = None
    f1_score = None
    algo_desc = None
    ds_desc = None

    def get(self, request):

        descriptions = mdb.find(db_data, "KNN")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']
        self.context = {'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}
        return render(request, self.template_name, self.context)

    def post(self, request):

        global data
        descriptions = mdb.find(db_data, "KNN")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']

        if 'update' in request.POST:
            self.algo_desc = request.POST.get('algo_desc')
            self.ds_desc = request.POST.get('ds_desc')
            self.update_message = Algorithm.description_update(self.algo_desc, self.ds_desc, 'KNN')
            op_data = mdb.find(db_data, 'OP')
            subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
            subject_obj.notify("Classification Description")
            self.context = {'update_message': self.update_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            self.pkl_message = Algorithm.pkl_upload(upload_file, 'KNN')
            self.context = {'pkl_message': self.pkl_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl_change' in request.POST:
            pkl_features_temp = str(request.POST.get('pkl_features')).split(',')
            pkl_label_notes_temp = str(request.POST.get("pkl_label_notes")).split("\r\n")
            image_file = request.FILES['graph_image']
            pkl_change_message_temp = Algorithm.pkl_change(pkl_features_temp, "KNN", pkl_label_notes_temp)
            graph_message = Algorithm.graph_upload(image_file, "KNN")
            if pkl_change_message_temp == "Success" and graph_message == "Success":
                op_data = mdb.find(db_data, 'OP')
                subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                subject_obj.notify("Classification Pickle File")
                self.pkl_change_message = "Changes Saved Successfully"
            else:
                if graph_message != "Success":
                    self.pkl_change_message = "Invalid Image Type"
                else:
                    self.pkl_change_message = "Unexpected error while saving pickle changes"
            self.context = {'pkl_change_message': self.pkl_change_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            self.feature_list, self.csv_message = Algorithm.csv_upload(upload_file)
            self.context = {'csv_message': self.csv_message, 'features': self.feature_list, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            image_file = request.FILES['csv_image']
            graph_message = Algorithm.graph_upload(image_file, "KNN")
            if graph_message == "Success":
                n_neighbors = int(request.POST.get("neighbors"))
                leaf_size = int(request.POST.get("leaf"))
                weights = str(request.POST.get("weights"))
                algorithm = str(request.POST.get("algorithm"))
                training_features = list(request.POST.getlist("training_features"))
                training_label = str(request.POST.get("training_label"))
                csv_label_notes_temp = str(request.POST.get("csv_label_notes")).split("\r\n")
                try:
                    csv_label_notes = {}
                    for label in csv_label_notes_temp:
                        temp = label.split("=")
                        csv_label_notes[temp[0].strip()] = temp[1].strip()

                    X = data.loc[:, training_features].values
                    y = data.loc[:, training_label].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
                    classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                      leaf_size=leaf_size,
                                                      weights=weights,
                                                      algorithm=algorithm)
                    sc, X_train, X_test = Algorithm.scale(X_train, X_test)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    self.accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                    self.f1_score = round(f1_score(y_test, y_pred) * 100, 2)
                    if self.accuracy >= 90 and self.f1_score >= 80:
                        pkl_obj = pickle.dumps(classifier)
                        mongo_data = {'pkl_data': Binary(pkl_obj), 'training_features': training_features,
                                      'label_notes': csv_label_notes, 'upload_method': 'csv',
                                      'testing_accuracy': self.accuracy, 'f1_score': self.f1_score,
                                      'scaling_obj': sc}
                        self.message = mdb.update(db_data, "KNN", mongo_data, "Model Successfully Trained",
                                                  "Unexpected error while training the model")
                        op_data = mdb.find(db_data, 'OP')
                        subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                        subject_obj.notify("Classification CSV File")
                    else:
                        self.message = "Model Training Failed. The accuracy of the model should be > 90% and " \
                                       "the F1-Score should be > 80%. However, this model's accuracy " \
                                       "is " + str(self.accuracy) + "% and the F1-Score " \
                                        "is " + str(self.f1_score) + "%."
                except:
                    self.message = "Unexpected error while training the model"
            else:
                self.message = "Invalid Image Type"

            self.context = {'submitbutton': self.submit_button, 'pkl_message': self.pkl_message,
                            'csv_message': self.csv_message, 'accuracy': self.accuracy, 'message': self.message,
                            'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc, 'f1_score': self.f1_score}

        return render(request, self.template_name, self.context)


class Regression(AdminStaffRequiredMixin, Algorithm):

    template_name = 'admin_classifier/regression.html'
    context = {}
    feature_list = []
    submit_button = None
    message = None
    pkl_message = None
    pkl_change_message = None
    csv_message = None
    update_message = None
    rmse = None
    algo_desc = None
    ds_desc = None

    def get(self, request):

        descriptions = mdb.find(db_data, "MLR")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']
        self.context = {'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}
        return render(request, self.template_name, self.context)

    def post(self, request):

        global data
        descriptions = mdb.find(db_data, "MLR")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']

        if 'update' in request.POST:
            self.algo_desc = request.POST.get('algo_desc')
            self.ds_desc = request.POST.get('ds_desc')
            self.update_message = Algorithm.description_update(self.algo_desc, self.ds_desc, 'MLR')
            op_data = mdb.find(db_data, 'OP')
            subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
            subject_obj.notify("Regression Description")
            self.context = {'update_message': self.update_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            self.pkl_message = Algorithm.pkl_upload(upload_file, 'MLR')
            self.context = {'pkl_message': self.pkl_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl_change' in request.POST:
            pkl_features_temp = str(request.POST.get('pkl_features')).split(',')
            image_file = request.FILES['graph_image']
            pkl_change_message_temp = Algorithm.pkl_change(pkl_features_temp, "MLR")
            graph_message = Algorithm.graph_upload(image_file, "MLR")
            if pkl_change_message_temp == "Success" and graph_message == "Success":
                op_data = mdb.find(db_data, 'OP')
                subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                subject_obj.notify("Regression Pickle File")
                self.pkl_change_message = "Changes Saved Successfully"
            else:
                if graph_message != "Success":
                    self.pkl_change_message = "Invalid Image Type"
                else:
                    self.pkl_change_message = "Unexpected error while saving pickle changes"
            self.context = {'pkl_change_message': self.pkl_change_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            self.feature_list, self.csv_message = Algorithm.csv_upload(upload_file)
            self.context = {'csv_message': self.csv_message, 'features': self.feature_list, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            image_file = request.FILES['csv_image']
            graph_message = Algorithm.graph_upload(image_file, "MLR")
            if graph_message == "Success":
                fit_intercept = str(request.POST.get("fit_intercept"))
                fit_intercept = True if fit_intercept == "True" else False
                normalize = str(request.POST.get("normalize"))
                normalize = True if normalize == "True" else False
                training_features = list(request.POST.getlist("training_features"))
                training_label = str(request.POST.get("training_label"))

                try:
                    X = data.loc[:, training_features].values
                    y = data.loc[:, training_label].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
                    sc, X_train, X_test = Algorithm.scale(X_train, X_test)
                    regressor = LinearRegression(fit_intercept=fit_intercept,
                                                 normalize=normalize)
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    self.rmse = round((mean_squared_error(y_test, y_pred))**0.5, 2)
                    print(self.rmse, y_test.mean()*0.1)
                    if self.rmse <= (y_test.mean()*0.1):
                        pkl_obj = pickle.dumps(regressor)
                        mongo_data = {'pkl_data': Binary(pkl_obj), 'training_features': training_features,
                                      'rmse': self.rmse, 'scaling_obj': sc, 'upload_method': 'csv'}
                        self.message = mdb.update(db_data, "MLR", mongo_data, "Model Successfully Trained",
                                                  "Unexpected error while training the model")
                        op_data = mdb.find(db_data, 'OP')
                        subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                        subject_obj.notify("Regression CSV File")
                    else:
                        self.message = "Model Training Failed. The RMSE values of the model should be " \
                                       "less than 10% of the actual values. For this model, 10% of the actual value " \
                                       "is " + str(round(y_test.mean()*0.1, 2)) + " and the RMSE value " \
                                        "is " + str(self.rmse)
                except:
                    self.message = "Unexpected error while training the model"
            else:
                self.message = "Invalid Image Type"

            self.context = {'submitbutton': self.submit_button, 'pkl_message': self.pkl_message,
                            'csv_message': self.csv_message, 'rmse': self.rmse, 'message': self.message,
                            'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}

        return render(request, self.template_name, self.context)


class Clustering(AdminStaffRequiredMixin, Algorithm):

    template_name = 'admin_classifier/clustering.html'
    context = {}
    feature_list = []
    submit_button = None
    message = None
    pkl_message = None
    pkl_change_message = None
    csv_message = None
    update_message = None
    algo_desc = None
    ds_desc = None

    def get(self, request):

        descriptions = mdb.find(db_data, "KM")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']
        self.context = {'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}
        return render(request, self.template_name, self.context)

    def post(self, request):

        global data
        descriptions = mdb.find(db_data, "KM")
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']

        if 'update' in request.POST:
            self.algo_desc = request.POST.get('algo_desc')
            self.ds_desc = request.POST.get('ds_desc')
            self.update_message = Algorithm.description_update(self.algo_desc, self.ds_desc, 'KM')
            op_data = mdb.find(db_data, 'OP')
            subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
            subject_obj.notify("Clustering Description")
            self.context = {'update_message': self.update_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            self.pkl_message = Algorithm.pkl_upload(upload_file, 'KM')
            self.context = {'pkl_message': self.pkl_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl_change' in request.POST:
            pkl_features_temp = str(request.POST.get('pkl_features')).split(',')
            pkl_label_notes_temp = str(request.POST.get("pkl_label_notes")).split("\r\n")
            image_file = request.FILES['graph_image']
            pkl_change_message_temp = Algorithm.pkl_change(pkl_features_temp, "KM", pkl_label_notes_temp)
            graph_message = Algorithm.graph_upload(image_file, "KM")
            if pkl_change_message_temp == "Success" and graph_message == "Success":
                op_data = mdb.find(db_data, 'OP')
                subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                subject_obj.notify("Clustering Pickle File")
                self.pkl_change_message = "Changes Saved Successfully"
            else:
                if graph_message != "Success":
                    self.pkl_change_message = "Invalid Image Type"
                else:
                    self.pkl_change_message = "Unexpected error while saving pickle changes"
            self.context = {'pkl_change_message': self.pkl_change_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            self.feature_list, self.csv_message = Algorithm.csv_upload(upload_file)
            self.context = {'csv_message': self.csv_message, 'features': self.feature_list, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            image_file = request.FILES['csv_image']
            graph_message = Algorithm.graph_upload(image_file, "KM")
            if graph_message == "Success":
                n_clusters = int(request.POST.get("n_clusters"))
                init = str(request.POST.get("init"))
                n_init = int(request.POST.get("n_init"))
                max_iter = int(request.POST.get("max_iter"))
                training_features = list(request.POST.getlist("training_features"))

                X = data.loc[:, training_features].values
                kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter)
                try:
                    kmeans.fit(X)
                    csv_label_notes_temp = str(request.POST.get("csv_label_notes")).split("\r\n")
                    csv_label_notes = {}
                    for label in csv_label_notes_temp:
                        temp = label.split("=")
                        values = (temp[0].strip()).split(",")
                        values = list(map(int, values))
                        key = kmeans.predict([values])[0]
                        csv_label_notes[str(key)] = temp[1].strip()
                    pkl_obj = pickle.dumps(kmeans)
                    mongo_data = {'pkl_data': Binary(pkl_obj), 'training_features': training_features,
                                  'label_notes': csv_label_notes, 'upload_method': 'csv'}
                    self.message = mdb.update(db_data, "KM", mongo_data, "Model Successfully Trained",
                                              "Unexpected error while training the model")
                    op_data = mdb.find(db_data, 'OP')
                    subject_obj = ConcreteSubject(op_data['observer_list'], op_data['update_message_list'])
                    subject_obj.notify("Clustering CSV File")
                except:
                    self.message = "Unexpected error while training the model"
            else:
                self.message = "Invalid Image Type"

            self.context = {'submitbutton': self.submit_button, 'pkl_message': self.pkl_message,
                            'csv_message': self.csv_message, 'message': self.message,
                            'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}

        return render(request, self.template_name, self.context)
# Simple Factory Pattern Ends


# Observer Pattern Starts
class Subject(ABC):

    @abstractmethod
    def subscribe(self, learner):
        pass

    @abstractmethod
    def unsubscribe(self, learner):
        pass

    @abstractmethod
    def notify(self, update):
        pass


class ConcreteSubject(Subject):

    def __init__(self, learners, messages):

        self._learners = learners
        self._messages = messages

    def subscribe(self, learner):

        self._learners.append(learner)
        mongo_data = {'observer_list': self._learners}
        temp = mdb.update(db_data, "OP", mongo_data, "Success", "Error")

    def unsubscribe(self, learner):

        self._learners.remove(learner)
        mongo_data = {'observer_list': self._learners}
        temp = mdb.update(db_data, "OP", mongo_data, "Success", "Error")

    def notify(self, update):

        self._messages.insert(0, "Admin Has Updated The " + str(update))
        if len(self._messages) > 5:
            temp = self._messages.pop()
        mongo_data = {'update_message_list': self._messages}
        temp = mdb.update(db_data, "OP", mongo_data, "Success", "Error")


class Learner(ABC):

    @abstractmethod
    def update(self):
        pass


class ConcreteLearner(Learner):

    def update(self):
        mongo_data = mdb.find(db_data, "OP")
        return mongo_data
# Observer Pattern Ends

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class NativeBayesClassifier:
    def train(self, features, labels):
        self.unique_labels = set(labels) #Y
        self.parameters = {}  #Store parameters for each label (mean, variance, priori) 

        for label in self.unique_labels:
            features_label = np.array([features[i] for i in range(len(labels)) if labels[i] == label])  #Select samples for the current label
            self.parameters[label] = {
                'mean': np.mean(features_label, axis=0),  #Mean of features 
                'variance': np.var(features_label, axis=0),  #Variance of features 
                'prior': len(features_label) / len(features)  #Prior probability of the label P(Y)
            }

    def normal_distribution(self, feature, mean, variance):
        #Calculating P(X|Y)
        epsilon = 1e-6
        variance += epsilon
        coeff = 1 / np.sqrt(2 * np.pi * variance)
        exponent = np.exp(-((feature - mean) ** 2) / (2 * variance))
        return coeff * exponent

    def calculate_posterior(self, feature):
        #Calculating P(Y|X) = P(Y) * P(X|Y) 
        posteriors = {}
        for label, params in self.parameters.items():
            prior = np.log(params['prior'])  #Logarithm of prior probability, because of possibility small values
            likelihoods = np.sum(np.log(self.normal_distribution(feature, params['mean'], params['variance'])))
            posteriors[label] = prior + likelihoods 

        return posteriors

    def predict(self, features):
        #label = argmax(P(Y|X))
        predictions = []

        for feature in features:
            posteriors = self.calculate_posterior(feature)
            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)

#Load Iris data 
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
df = pd.read_csv("./iris/iris.data", header=None, names=column_names)

#Labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

#Training and test data set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state=42)

#Training
model = NativeBayesClassifier()
model.train(features_train, labels_train)

#Prediction
labels_pred = model.predict(features_test)
accuracy = accuracy_score(labels_test, labels_pred)

print(f"Model accuracy: {accuracy * 100}%")

#Examples
df_predictions = pd.DataFrame({
    'Label': labels_test,
    'Predicted Label': labels_pred
})
print(df_predictions.head())

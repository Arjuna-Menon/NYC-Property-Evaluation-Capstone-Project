
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


def generate_loss_plot(history, filename=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if (filename!=None):
        plt.savefig(filename)
    plt.show()


def generate_report(history, Y_train, Y_test, epochs):
    report_train = pd.DataFrame()
    report_train['mse'] = history.history['mse']
    report_train['RMSE'] = np.sqrt(pd.DataFrame(history.history['mse']))
    report_train['R2'] = history.history['r_square']
    report_train['actual'] = Y_train[:epochs]
    
    report_test = pd.DataFrame()
    report_test['mse'] = history.history['val_mse']
    report_test['RMSE'] = np.sqrt(pd.DataFrame(history.history['val_mse']))
    report_test['R2'] = history.history['val_r_square']
    report_test['actual'] = Y_test[:epochs]
    return report_train, report_test

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def get_data(): 
    df = pd.read_csv('pluto5_stddum.csv')
    df.drop(['assessland'], axis=1, inplace=True)
    
    X = df[df.columns]
    X.drop('assesstot', axis=1, inplace=True)
    predictors = X.columns
    X = X.values
    Y = df['assesstot'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test, predictors

def fit_model(model, x_train, x_test, y_train, y_test, epochs):
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', r_square])
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    generate_loss_plot(history, filename=None)
    report_train, report_test = generate_report(history, y_train, y_test, epochs)
    return model, report_train, report_test


def predict(model, x, y, filename=None):
    y_pred = model.predict(x)
    fig, ax = plt.subplots()
    ax.plot(y, color = 'blue')
    ax.plot(y_pred, color = 'red')
    ax.legend(['Real', 'Predicted'])
    if (filename!=None):
        fig.savefig(filename)
    plt.show()
    return y_pred
    
def print_errors(report, y_pred, y_actual, epochs):
    report['error'] = report['actual'] - pd.DataFrame(y_pred[:epochs])[0]
    print(report[['mse','RMSE','R2', 'error']].mean())
    #TODO: plt.hist((y-y_actual), bins='auto')
    return report

def get_means(reports, metric):
    means = []
    for report in reports:
        means.append(report[[metric]].mean()[0])
    return means
        
def calculate_best(reports):
    rmse_values = get_means(reports, 'RMSE')
    print('index=', rmse_values.index(min(rmse_values)), 'value=', min(rmse_values))
    r2_values = get_means(reports, 'R2')
    print('index=', r2_values.index(min(r2_values)), 'value=', max(r2_values))

x_train, x_test, y_train, y_test, predictors = get_data()
input_nodes = len(predictors)
epochs = 20
reports = []

def run_model(hidden_nodes, x_train, x_test, y_train, y_test, epochs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_nodes, tf.keras.activations.linear))
    model.add(tf.keras.layers.Dense(hidden_nodes, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, tf.keras.activations.linear))
    model, report_train, report_test = fit_model(model, x_train, x_test, y_train, y_test, epochs)
    y_pred = predict(model, x_test, y_test)
    report_test = print_errors(report_test, y_pred, y_test, epochs)
    return report_train, report_test

#choose the amount of nodes in hidden layers: http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-10.html
#NN0: 1 hidden layer with (Number of inputs + outputs) * (2/3) nodes: overfitting
report_train, report_test = run_model(int((input_nodes+1)*(2/3)), x_train, x_test, y_train, y_test, epochs)
reports.append(report_test)


#NN1: # A typical recommendation is that
#the number of weights should be no more than 1/30 of the number of training cases: underfitting
report_train, report_test = run_model(int(len(x_train)/(30*2)), x_train, x_test, y_train, y_test, epochs)
reports.append(report_test)


#NN2: reduce amount of nodes hidden layer: underfitting
report_train, report_test = run_model(int(len(x_train)/(30*4)), x_train, x_test, y_train, y_test, epochs)
reports.append(report_test)


#NN3: reduce amount of nodes hidden layer: underfitting
report_train, report_test = run_model(int(len(x_train)/(30*6)), x_train, x_test, y_train, y_test, epochs)
reports.append(report_test)

#NN4: reduce amount of nodes hidden layer: underfitting
report_train, report_test = run_model(int(len(x_train)/(30*8)), x_train, x_test, y_train, y_test, epochs)
reports.append(report_test)


#choose the best
calculate_best(reports)


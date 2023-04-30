#Abd Elelah Arafah
#400197623
#Assignment #5

import numpy
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("\n ______________________ ")
print("|-----Code Initiated---|")
print("|------Abd Elelah------|")
print("|------Assignment5-----|")
print("|______________________|")

#Hard coded global vars
num_of_runs = 5
epochs_value = 1000 #1000 runs through the entire dataset
test_hidden_sizes = numpy.arange(2,7) #Testing from sizes 2 to 6 for each hidden layer
data_file_name = 'data_banknote_authentitcation.txt'
figure_count = 0 #counter to assign each plot a number

#A fxn to calc the misclas. Rate
def Misclas_Rate_Fxn(model, X, t):
    test_values = model.NN_FWD_PASS_Fxn(X.T)["l3_out"].T
    return numpy.sum(numpy.rint(test_values) != t) / t.shape[0]

#A fxn to return the ReLU activation function: max(0, z)
def Recti_Lin_Unit_Fxn(val):
    val[val < 0] = 0
    return val


#Define a class for the Neural Networks to contain all the calculations and contains all the functions
class NN_Class():
    #define init vals
    def __init__(self, X_Data, t_Label_Data, h1_size_of_layer, h2_size_of_layer, learning_rate):
        self.learning_rate = learning_rate
        self.X_Data = X_Data
        self.t_Label_Data = t_Label_Data
        self.numFeatures = X_Data.shape[1]
        self.param_w1 = numpy.random.randn(h1_size_of_layer, self.numFeatures)
        self.param_w2 = numpy.random.randn(h2_size_of_layer, h1_size_of_layer)
        self.param_w3 = numpy.random.randn(t_Label_Data.shape[1], h2_size_of_layer)
        self.h1_size_of_layer = h1_size_of_layer
        self.h2_size_of_layer = h2_size_of_layer

    #train fxn
    def NN_train_Fxn(self):
        for _ in range(epochs_value):
            self.Update_w_param_Fxn(self.Calc_Grad_Fxn(
                self.NN_FWD_PASS_Fxn(self.X_Data.T)))

    #A fxn to compute forward propg.
    def NN_FWD_PASS_Fxn(self, X):
        store = {}
        store["l1_z"] = numpy.dot(self.param_w1, X)
        store["l1_out"] = Recti_Lin_Unit_Fxn(store["l1_z"])
        store["l2_z"] = numpy.dot(self.param_w2, store["l1_out"])
        store["l2_out"] = Recti_Lin_Unit_Fxn(store["l2_z"])
        store["l3_z"] = numpy.dot(self.param_w3, store["l2_out"])
        store["l3_out"] = Recti_Lin_Unit_Fxn(store["l3_z"])
        return store

    #Computing the Cost using the cross-entropy loss function
    def Calc_Cost_Fxn(self, y_array, t_array):
        cost_comp = 0
        size = len(t_array)
        for i in range(size):
            cost_comp += t_array[i][0] * numpy.logaddexp(0, -y_array[i][0]) + \
                (1-t_array[i][0]) * numpy.logaddexp(0, y_array[i][0])
        return cost_comp / size
        
    def Compute_Err_Fxn(self, X_data_Arr, t_Data_Arr):
        FwdPropComp = self.NN_FWD_PASS_Fxn(X_data_Arr.T)
        return self.Calc_Cost_Fxn(FwdPropComp["l3_out"].T, t_Data_Arr)

    #A fxn to compute backward propg.
    def Calc_Grad_Fxn(self, forwardPassResults):
        stored_vals = {}
        size_inverse = 1 / len(self.t_Label_Data)
        label_data = self.t_Label_Data.T
        X_data = self.X_Data.T

        #For parameter w3
        differential_A = size_inverse * (forwardPassResults["l3_out"] - label_data)
        differential_Z = differential_A
        stored_vals["w3"] = size_inverse * numpy.dot(differential_Z, forwardPassResults['l2_out'].T)

        #For parameter w2
        differential_A = numpy.dot(self.param_w3.T, differential_Z)
        differential_Z = numpy.multiply(differential_A, numpy.where(forwardPassResults["l2_out"] > 0, 1, 0))
        stored_vals["w2"] = size_inverse * numpy.dot(differential_Z, forwardPassResults['l1_out'].T)

        #For parameter w1
        differential_A = numpy.dot(self.param_w2.T, differential_Z)
        differential_Z = numpy.multiply(differential_A, numpy.where(forwardPassResults["l1_out"] > 0, 1, 0))
        stored_vals["w1"] = size_inverse * numpy.dot(differential_Z, X_data.T)

        return stored_vals

    #gradient descent from lectures
    def Update_w_param_Fxn(self, grad_val):
        self.param_w1 -= self.learning_rate * grad_val["w1"]
        self.param_w2 -= self.learning_rate * grad_val["w2"]
        self.param_w3 -= self.learning_rate * grad_val["w3"]



#A fxn to plot the errors
def Plot_Data_Fxn(title_of_the_plot, FNN_Model, X_Tr_Data, t_Tr_Data, X_validation_Data, t_validation_Data, X_Ts_Data, t_Ts_Data, fig_cnt):
    #Store all errors
    store_train_errs = []
    store_valid_errs = []
    store_misclas_rate = []
    #Loop through the data to compute the NN computations with 1000 epochs
    for z in range(epochs_value):
        store_misclas_rate.append(Misclas_Rate_Fxn(FNN_Model, X_Ts_Data, t_Ts_Data))
        compute_errs = FNN_Model.Compute_Err_Fxn(X_validation_Data, t_validation_Data)
        store_valid_errs.append(compute_errs)
        train_values = FNN_Model.NN_FWD_PASS_Fxn(X_Tr_Data.T)
        compute_errs = FNN_Model.Compute_Err_Fxn(X_Tr_Data, t_Tr_Data)
        store_train_errs.append(compute_errs)
        compute_grad = FNN_Model.Calc_Grad_Fxn(train_values)
        FNN_Model.Update_w_param_Fxn(compute_grad)
    #Plotting all the figures
    plt.figure(fig_cnt)
    plt.plot(store_train_errs, label='Train Errors', color="red")
    plt.plot(store_valid_errs, label='Valid Errors', color="blue")
    plt.plot(store_misclas_rate, label='Misclas. Rate', color="green")
    plt.title(title_of_the_plot)
    plt.legend(loc="center right")
    plt.xlabel("Epochs")
    plt.ylabel("Error Values")



#main block of code to perform all the computations and plot the data
if __name__ == '__main__':
    #Importing the data from local machine using numpy library
    banknote_db = numpy.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
    #randomize using student ID
    numpy.random.seed(7623)
    numpy.random.shuffle(banknote_db)
    #Split the data into y and X sets
    x_data = banknote_db[:, :-1]
    y_data = banknote_db[:, -1]
    #Splitting the data into 60% train, 20% test, 20% valid
    X_train_set, temp, t_train_set, t_Temp = train_test_split(x_data, y_data, test_size=0.3, train_size=0.7, random_state=7623)
    X_valid_set, X_test_set, t_valid_set, t_test_set = train_test_split(temp, t_Temp, test_size = 0.5, train_size = 0.5, random_state=7623)
    #Scaling
    sc = StandardScaler()  
    X_train_set = sc.fit_transform(X_train_set)
    X_valid_set = sc.transform(X_valid_set)
    X_test_set = sc.transform(X_test_set)
    t_train_set = numpy.expand_dims(t_train_set, axis=1)
    t_test_set = numpy.expand_dims(t_test_set, axis=1)
    t_valid_set = numpy.expand_dims(t_valid_set, axis=1)
    #Comparing features from 2 -> 4
    feat_num = numpy.arange(2,5)
    #Best found error and model
    best_found_error_value = math.inf
    best_found_model = None

    #Loop and print the error values for each combination
    for feat_number in feat_num:
        error_store_val = math.inf
        object_NN = None
        print("\nNum. of features = ", feat_number)
        for h1_1sthidden_layer_size in test_hidden_sizes:
            for h2_2ndhidden_layer_size in test_hidden_sizes:
                error_holder = 0
                for count in range(num_of_runs):
                    Current_NN_Model = NN_Class(X_train_set[:, :feat_number], t_train_set, h1_1sthidden_layer_size, h2_2ndhidden_layer_size, 0.005)
                    Current_NN_Model.NN_train_Fxn()
                    error_holder += Current_NN_Model.Compute_Err_Fxn(X_valid_set[:, :feat_number], t_valid_set)/num_of_runs
                #Compare the error values for each size of the hidden layer
                if error_holder < error_store_val:
                    error_store_val = error_holder
                    object_NN = Current_NN_Model
                if error_holder < best_found_error_value:
                    best_found_error_value = error_holder
                    best_found_model = Current_NN_Model
                print("h1 size:\t" + str(h1_1sthidden_layer_size),", h2 size:\t" + str(h2_2ndhidden_layer_size),", CE ERR:\t" + str(error_holder.round(4)),", MR:\t" + str(Misclas_Rate_Fxn(Current_NN_Model, X_train_set[:, :feat_number], t_train_set).round(4)))
        #Print each of the data
        Plot_Data_Fxn(str("Num Feat. =" + str(feat_number) + ", h1 Size, h2 Size = " + str(object_NN.h1_size_of_layer) + ', ' + str(object_NN.h2_size_of_layer)), object_NN, X_train_set[:, :feat_number], t_train_set, X_valid_set[:, :feat_number], t_valid_set, X_test_set[:, :feat_number], t_test_set, figure_count)
        figure_count = figure_count + 1

    
    #Print the Results for the best model
    print("\n----------------------------------------------")
    print("Results as discovered by running the code:")
    print("No. of features =", best_found_model.numFeatures)
    print("Size of Hidden Layer h1 =", best_found_model.h1_size_of_layer)
    print("Size of Hidden Layer h2 =", best_found_model.h2_size_of_layer)
    print("The minimized Test Error =", best_found_model.Compute_Err_Fxn(X_test_set[:, :best_found_model.numFeatures], t_test_set))
    print("The minimized Valid Error =", best_found_error_value)
    print("----------------------------------------------")
    plt.show()
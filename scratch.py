import numpy as np
from scipy import signal
import gzip
import _pickle as pkl 
from im2col import im2col_indices


def tanh(x):
	return np.tanh(x)

def d_tanh(x):
	return 1 - np.tanh(x) ** 2                                

def log(x):
	return 1/(1 + np.exp(-1*x))

def d_log(x):
	return log(x) * (1 - log(x))

np.random.seed(598765)

x1 = np.array([[0,0,0], [0,0,0], [0,0,0]])
x2 = np.array([[1,1,1], [0,0,0], [0,0,0]])
x3 = np.array([[0,0,0], [1,1,1], [1,1,1]])
x4 = np.array([[1,1,1], [1,1,1], [1,1,1]])


X = [x1,x2,x3,x4]




Y = np.array([[0.53],
			  [0.77],
			  [0.88],
			  [1.1],
			  ])


f = gzip.open('mnist.pkl.gz', 'rb')


train_data, val_data, test_data=pkl.load(f, encoding='latin1')




X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data



X_train = X_train.reshape(-1, 1, 28, 28)
X_val = X_val.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)




m = y_train.shape[0]
y = np.zeros((m, 10), dtype="int32")
for i in range(m):
    idx = y_train[i]
    y[i][idx] = 1

w1 = np.random.randn(2,2) * 4
w2 = np.random.randn(729, 10) * 4


num_epochs = 1000
learning_rate = 0.7


cost_before_trian = 0
cost_after_train=0



#grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2.T)
#grad_1_part_2 = d_tanh(layer_1)
#grad_1_part_3 = X[i]
#grad_1_part_1_reshape = np.reshape(grad_1_part_1,(2,2))
#grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
#grad_1 = signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1,2),'valid')





final_out, start_out = np.array([[]]), np.array([[]])

for it in range(num_epochs):

	for i,X_ in enumerate(X_train):

		layer_1 = signal.convolve2d(X_[0], w1, 'same')
		layer_1_act = tanh(layer_1)
		layer_1_axis1 = np.expand_dims(layer_1_act, axis=0)
		layer_1_act = np.expand_dims(layer_1_axis1, axis=1)
		layer_1_col = im2col_indices(layer_1_act, 2, 2, 0, 1)

		max_pool_layer_1 = np.argmax(layer_1_col, axis=0)
				
		layer_2 = max_pool_layer_1.dot(w2)

		layer_2_act = log(layer_2)

		cost = np.square(layer_2_act - Y[i]).sum() * 0.5
		
		if i % 100 == 0:
			print("Current iteration", it, "current_train", i, "current_cost:", cost,end="/n")


		grad_2_part_1 = layer_2_act - Y[i]
		grad_2_part_2 = d_log(layer_2)
		grad_2_part_3 = max_pool_layer_1
		
		grad_2_part_temp = grad_2_part_1*grad_2_part_2
		print()
		# print(grad_2_part_3.shape)
		# print(grad_2_part_temp.shape)
		grad_2 = np.reshape(grad_2_part_3,(1,-1)).T.dot(np.reshape(grad_2_part_temp,(1, -1)))
		

		
		grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2.T)
		grad_1_part_2 = d_tanh(layer_1)
		grad_1_part_3 = X_[0]
		print(grad_1_part_1.shape,grad_1_part_2.shape,grad_1_part_3.shape)
		
		# grad_1_part_1_reshape = np.reshape(grad_1_part_1, (27,27))
		grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
		
		grad_1  = np.rot90(signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2),'valid'),2)


		w2 = w2 - grad_2 * learning_rate
		w1 = w1 - grad_1 * learning_rate


for i in range(len(X)):

	layer_1 = signal.convolve2d(X[i], w1, 'valid')
	layer_1_act = tanh(layer_1)
	layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
	layer_2 = layer_1_act_vec.dot(w2)
	layer_2_act = log(layer_2)
	cost = np.square(layer_2_act - Y[i]).sum() * 0.5
	cost_after_train = cost_after_train + cost
	final_out = np.append(final_out, layer_2_act)


print("\nW1:", w1, "\nW2:", w2)
print("------------------------")

print("cost before training",cost_before_trian)
print("cost_after_train", cost_after_train)

print("-------------------------")

print("Sart Out put:", start_out) 
print("final_out:", final_out)
print("Ground Truth:", Y.T)








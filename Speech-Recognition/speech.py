from __future__ import absolute_import
from __future__ import division
import time
import pdb
import tensorflow as tf
import numpy as np
from prep2 import assign_files_and_labels,process_input,pad_sequences,sparse_tuple_from,process_text,modify_text
import os
import csv


rootdir = "/home/praveen/Downloads/data%2Fspeech_commands_v0.01"
if os.path.isfile('trainfull.npy'):
    train = np.load('trainfull.npy')
    test = np.load('testfull.npy')
    validate = np.load('validatefull.npy')
    train_targets = np.load('trainfull_targets.npy')
    test_targets = np.load('testfull_targets.npy')
    validate_targets = np.load('validatefull_targets.npy')
else:
    input_files = assign_files_and_labels(rootdir)
    input_files = process_input(input_files)
    
    for j in range(len(input_files[0])):
       for i,e in enumerate(input_files[0][j]):
           if e.shape[1] != 44:
               l = [0]*(20*(44-e.shape[1]))
               input_files[0][j][i] = np.append(e,l).reshape(20,44)
    
    train,test,validate = input_files[0]    
    train_targets,test_targets,validate_targets = input_files[1]
    


def subset(points):
    global train,train_targets,test,test_targets,validate,validate_targets
    train = train[:points]
    train_targets = train_targets[:points]
    validate = validate[:int(points*0.1)]
    validate_targets = validate_targets[:int(points*0.1)]
    test = test[:int(points*0.1)]
    test_targets = test_targets[:int(points*0.1)]
    

num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 6000
num_hidden = 512
num_layers = 1
batch_size = 128
initial_learning_rate = 0.0004
momentum = 0.02
num_examples = len(train)
num_batches_per_epoch = int(num_examples/batch_size)
epsilon = 1e-3
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
div = 1
csv_file = "512_rnn_1.csv"

def shuffle_and_save(inp,targets,category):
    vect = (zip(inp,targets))
    i,t= zip(*np.random.permutation(vect))
    #np.save(category+'.npy',i)
    #np.save(category+'_targets.npy',t)
    return i,t


shuffle_and_save(train,train_targets,'trainfull')
shuffle_and_save(test,test_targets,'testfull')
shuffle_and_save(validate,validate_targets,'validatefull')

def nextbatch(arr,batch,size):
    first_index = batch*size
    last_index = (batch+1)*size    
    return arr[first_index:last_index]


# THE MAIN CODE!
with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():

        inputs = tf.placeholder(tf.float32,[None,20,None])

        input_layer = tf.reshape(inputs,[-1,tf.shape(inputs)[2],20])
        
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs_rnn = input_layer
        
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])
        #seq_len = tf.fill([batch_size,],54)
        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell 
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(num_hidden)
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob = 0.5)
            cells.append(cell)
        stack = tf.contrib.rnn.MultiRNNCell(cells)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs_rnn, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs_rnn)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # A global step to save checkpoint numbers
        global_step = tf.Variable(0,dtype = tf.int32,trainable = False,name = 'global_step')

        # A variable to save total time taken
        timing = tf.Variable(0,dtype="float",trainable = False)
        
        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        
        batch_mean2, batch_var2 = tf.nn.moments(logits,[0])
        scale2 = tf.Variable(tf.ones([num_classes]))
        beta2 = tf.Variable(tf.zeros([num_classes]))
        logits = tf.nn.batch_normalization(logits,batch_mean2,batch_var2,beta2,scale2,epsilon)
        
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        
        loss = tf.nn.ctc_loss(targets,logits, seq_len)#,ignore_longer_outputs_than_inputs=True)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(cost)
        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        #ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              ##targets))

        edist = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets,
                           normalize=False)
        ler = tf.reduce_mean(tf.cast(tf.equal(edist, 0), tf.float32))
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()


        ckpt_dir = "./512rnn1checkpoint_dir"
        if not os.path.exists(ckpt_dir):
            print "Created dir"
            os.makedirs(ckpt_dir)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print "Loading From Checkpoint ",ckpt.model_checkpoint_path
            saver.restore(session, ckpt.model_checkpoint_path) # restore all variables

        
        s = global_step.eval() # get last global_step
        print "Starting From Epoch:", s+1
        total_time = timing.eval()
        print "Starting time",total_time
        #pdb.set_trace()
        for curr_epoch in range(s,num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
               
                batch_train = nextbatch(train,batch,batch_size)
               
                tim = int(44)/div
                
                batch_len = [tim for i in range((batch_size))]
                batch_targets = nextbatch(train_targets,batch,batch_size)
                batch_targets = sparse_tuple_from(batch_targets)
                feed = {inputs: batch_train,
                        targets: batch_targets,
                        seq_len: batch_len}
                
                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size

            train_cost /= num_examples
            train_ler /= num_examples
            log = "Epoch {}/{}, train_cost = {:.3f}, train_acc = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))
            total_time+= time.time()-start
            timing.assign(total_time).eval()

            
            #validation_input,validation_len = pad_sequences(validate)
            validation_input = validate
            tim = int(44)/div
            validation_len = [tim for i in range(len(validate))]
            validation_target = sparse_tuple_from(validate_targets)
            val_ler = session.run(ler,feed_dict = {inputs:validation_input,targets:validation_target,seq_len:validation_len})
            print " Validation Acc :",val_ler
            
            train,train_targets = shuffle_and_save(train,train_targets,'train')
            test,test_targets = shuffle_and_save(test,test_targets,'test')
            global_step.assign(curr_epoch+1).eval()
           
            test_ler = ''
            if (curr_epoch+1)%5 == 0:
                saver.save(session, ckpt_dir + "/model.ckpt", global_step=global_step)
            if (curr_epoch)%10 == 0:
                #test_input,test_len = pad_sequences(test)
                tim = int(44)/div
                test_input = test
                test_len = [tim for i in range(len(test))]
                test_target = sparse_tuple_from(test_targets)
                feed = {inputs:test_input,targets:test_target,seq_len: test_len}
                test_ler = session.run(ler,feed_dict = feed)
                print " Test ACC :",test_ler
                t = test
                tim = int(44)/div
                batch_len = [tim for i in range(len(test))]
                feed = {inputs:t,seq_len:batch_len}

                d = session.run(decoded[0], feed_dict=feed)

                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

                for i, seq in enumerate(dense_decoded):

                    org = [chr(c+FIRST_INDEX) for c in test_targets[i]]
                    print "Original:","".join(org)
                    seq = [s for s in seq if s != -1]
                    s = [chr(c+FIRST_INDEX) for c in seq]
                    print "Predicted:","".join(s)
                    if i > 10 :
                        break
            fields=[curr_epoch+1,train_ler,val_ler,test_ler]
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            
            

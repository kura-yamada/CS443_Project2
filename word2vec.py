'''word2vec.py
Skip-gram Word2vec neural network implemented in TensorFlow
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 2: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import numpy as np
import tensorflow as tf


class Skipgram:
    '''A Skip-gram neural network (word2vec approach) that takes target words as data samples and learns to predict the
    surrounding context words ("classes"). The network learns emergent numerical word embedding vectors for the target
    words based on word context.

    The network has an MLP archeciture:
    Input -> Dense (linear act) -> Dense (softmax act)
    '''
    def __init__(self, num_feats, num_hidden, num_classes, wt_stdev=0.1):
        '''Constructor to intialize the network weights and bias. Skip-gram is a 2 layer MLP network (not counting the
        input layer).

        Parameters:
        -----------
        num_feats: int. Num input features (M)
        num_hidden: int. Num of units in the hidden layer (H)
        num_classes: int. Num data classes (C)
        wt_stdev: float. Standard deviation of the Gaussian-distributed weights and bias

        TODO:
        - Create Gaussian-distributed wts and bias tensors for each layer of the network. Remember to wrap your weights
        and bias as tf.Variables for gradient tracking!
        '''
        self.M = num_feats
        self.H = num_hidden
        self.C = num_classes
        self.set_wts(tf.Variable(tf.random.normal(mean = 0, stddev = wt_stdev, shape = (self.M, self.H))),
        tf.Variable(tf.random.normal(mean = 0, stddev = wt_stdev, shape = (self.H,self.C))))
        self.set_b(tf.Variable(tf.random.normal(mean = 0, stddev=wt_stdev, shape=(self.H,))),
                    tf.Variable(tf.random.normal(mean = 0, stddev=wt_stdev, shape=(self.C,))))

    def set_wts(self, y_wts, z_wts):
        '''Replaces the network weights with those passed in as parameters. Used by test code.

        Parameters:
        -----------
        y_wts: tf.Variable. shape=(M, H). New input-to-hidden layer weights.
        z_wts: tf.Variable. shape=(H, C). New hidden-to-output layer weights.
        '''
        self.y_wts, self.z_wts = y_wts, z_wts

    def set_b(self, y_b, z_b):
        '''Replaces the network biases with those passed in as parameters. Used by test code.

        Parameters:
        -----------
        y_b: tf.Variable. shape=(H,). New hidden layer bias.
        z_b: tf.Variable. shape=(C,). New output layer bias.
        '''
        self.y_b, self.z_b = y_b, z_b

    def one_hot(self, x, C):
        '''One-hot codes the vector of class labels `y`

        Parameters:
        -----------
        y: ndarray. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.

        Copy-and-paste from Hebbian Learning project.
        '''
        return tf.cast(tf.one_hot(tf.cast(x,dtype=tf.int32), C, off_value = 0), dtype=tf.float32)

    def multi_hot(self, y_ind_list, num_classes):
        '''Multi-hot codes the vector of class labels `y_ind_list`

        Parameters:
        -----------
        y_ind_list: ndarray of ndarrays. len(outer ndarray)=B, len(each inner ndarray)= = Si.
            int-coded class assignments for each sample i in the training mini-batch
            (i.e.) indices of the context words associated with each target word i in the mini-batch.
            NOTE: The number of context words (Si) is NOT THE SAME for each sample (target word) i!
        num_classes int. Number of unique output classes total (i.e. vocab size)

        Returns:
        -----------
        y_multi_h: tf.constant. tf.float32. shape=(B, C) Multi-hot coded class assignments.

        NOTE:
        - Because of the uneven/jagged number of context words for the samples in the mini-batch, it is difficult to use
        only vectorization to implement this. Using one loop here is totally fine!
        - Because you are building a constant tensor, it might be easier to build the mini-batch of multi-hot vectors in
        numpy and then convert to a TensorFlow tensor at the end.
        '''
        y_multi_h = np.zeros((len(y_ind_list), num_classes))
        for i, y  in enumerate(y_ind_list):
            y_multi_h[i,:] = np.sum(self.one_hot(y, num_classes), axis = 0)
        y_multi_h = np.where(y_multi_h >= 1, 1, 0)

        return tf.cast(y_multi_h, dtype=tf.float32)

    def forward(self, x):
        '''Performs the forward pass through the decoder network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_in: tf.constant. shape=(B, C). net_in in the output layer to every sample in the mini-batch.

        NOTE:
        - You are returning NET_IN rather than the usual NET_ACT here because the Skip-gram loss involves the output
        layer net_in instead of the output layer net_act.
        '''
        y = x @ self.y_wts + self.y_b

        return tf.cast(y @ self.z_wts + self.z_b, dtype = tf.float32)

    def loss(self, z_net_in, y_multi_h):
        '''Computes the Skip-gram loss on the current mini-batch using the multi-hot coded class labels `y_multi_h`.

        Parameters:
        -----------
        z_net_in: tf.constant. shape=(B, C). Net input to every sample in the mini-batch in the output layer.
        y_multi_h: tf.constant. tf.float32. shape=(B, C). Multi-hot coded class assignments (i.e. 1s at context word
            indices, 0s elsewhere for each sample).
            For example:
            y_multi_h = [[0 1 0 0]
                         [0 1 0 1]]
            means that sample 0 in the mini-batch has one context word that has index 1 in the vocab. Sample 1 in the
            mini-batch has two context words that have indices 1 and 3 in the vocab.

        Returns:
        -----------
        loss: float. Skip-gram loss averaged over the mini-batch.

        Hints:
        - It may be a good idea to break down the loss equation and implement each component on separate lines of code.
        - Check out https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp
        - To implement the right-most term in the loss equation, it may be helpful to notice that y_multi_h has the same
        shape as z_net_in and y_multi_h has 1s at all the context word positions (yc in the equation) and 0s elsewhere.
        Thus, you can use y_multi_h as a "gate" or a "filter" to extract z_net_in values that are at the appropriate
        context words and at the same time nulling out the remaining z_net_in values that are not desired.
        '''
        right_side = tf.reduce_sum(y_multi_h * z_net_in, axis = 1)
        left_side = tf.reduce_logsumexp(z_net_in, axis = 1) * tf.reduce_sum(y_multi_h, axis = 1)
        return tf.reduce_sum(left_side - right_side) / float(tf.shape(y_multi_h)[0])


    def extract_at_indices(self, x, indices):
        '''Returns the samples in `x` that have indices `indices` to form a mini-batch.

        Parameters:
        -----------
        x: tf.constant. shape=(N, ...). Data samples or labels
        indices: tf.constant. tf.int32, shape=(B,), Indices of samples to return from `x` to form a mini-batch.
            Indices may be in any order, there may be duplicate indices, and B may be less than N (i.e. a mini-batch).
            For example indices could be [0, 1, 2], [2, 2, 1], or [2, 1].
            In the case of [2, 1] this method would samples with index 2 and index 1 (in that order).

        Returns:
        -----------
        tf.constant. shape=(B, ...). Value extracted from `x` whose sample indices are `indices`.

        Copy-and-paste from Hebbian Learning project.
        '''
        return tf.gather(x, indices)

    def fit(self, x, y, mini_batch_sz=512, lr=1e-2, n_epochs=400, print_every=100, verbose=True):
        '''Trains the Skip-gram network on the training samples `x` (int-coded target words) and associated labels `y`
        (int-coded context words). Uses the Adam optimizer.

        Parameters:
        -----------
        x: ndarray. ints. shape=(N,). Int-coded data samples in the corpus (target words).
        y: ndarray of ndarrays. len(outer ndarray)=N, len(each inner ndarray)= = Si.
            int-coded class assignments for each sample i in the corpus
            (i.e.) indices of the context words associated with each target word i
            NOTE: The number of context words (Si) is NOT NECESSARILY THE SAME for each sample (target word) i!
        mini_batch_sz: int. Number of samples to include in each mini-batch.
        lr: float. Learning rate used with Adam optimizer.
        n_epochs: int. Train the network for this many epochs
        print_every: int. How often (in epoches) to print out the current epoch and training loss.
        verbose: bool. If set to `False`, there should be no print outs during training. Messages indicating start and
        end of training are fine.


        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.

        TODO:
        Go through the usual motions:
        - Set up Adam optimizer and training loss history tracking container.
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - For each mini-batch, one-hot code target words and multi-hot code context words.
        - Compute forward pass and loss for each mini-batch. Have your Adam optimizer apply the gradients to update each
        wts and bias.
        - Record the average training loss values across all mini-batches.
        '''
        N = x.shape[0]
        n_minibatches = N // mini_batch_sz if N % mini_batch_sz == 0 else (N // mini_batch_sz) + 1

        # Define loss tracking containers
        train_loss_hist = []
        

        #Adam Optimizer
        opt = tf.optimizers.Adam(learning_rate=lr)

        #Multi-hot coding
        y = self.multi_hot(y, self.C)
        #one hot coding
        x = self.one_hot(x, self.M)

  

        print(f"Starting training with a max of {n_epochs}...")
        for num_epochs in range(n_epochs):
            #Running all the minibatches, keeping track of loss average
            loss_avg = 0
            for i in range(n_minibatches):
                #Getting random sample
                idx = tf.random.uniform(shape=[mini_batch_sz], minval=0, maxval=N, dtype=tf.int32)
                x_mini = self.extract_at_indices(x, idx)
                y_mini = self.extract_at_indices(y, idx)
                #Forward and Backwards pass
                with tf.GradientTape() as tape:
                    net_act = self.forward(x_mini)
                    loss = self.loss(net_act, y_mini)
                grads = tape.gradient(loss, (self.y_wts, self.y_b, self.z_wts, self.z_b))
                #Apply optimizer
                opt.apply_gradients(zip(grads, (self.y_wts, self.y_b, self.z_wts, self.z_b)))

                loss_avg += float(loss)

            loss_avg /= n_minibatches
            train_loss_hist.append(loss_avg)

            if verbose and num_epochs % print_every == 0:
                print(f"Completed epoch {num_epochs}; training loss: {loss_avg:.5f}")


        return train_loss_hist

    def get_word_vector(self, word2ind, word):
        '''Get a single word embedding vector from a trained network

           Parameters:
           -----------
           word2ind: Dictionary. Maps word strings -> int code indices in vocab
           word: Word for which we want to return its word embedding vector

           Returns:
           -----------
           ndarray. Word embedding vector. shape=(embedding_sz,)
           This is the wt vector from the 1st net layer at the index specified by the word's int-code.
        '''
        if word not in word2ind:
            print(f'{word} not in word dictionary!')
        
        input = np.array([word2ind[word]])
        input = self.one_hot(input, self.C)
    
        return np.squeeze(input @ self.y_wts + self.y_b)
        

    def get_all_word_vectors(self, word2ind, wordList):
        '''Get all word embedding vectors for the list of words `wordList` from the trained network

           Parameters:
           -----------
           word2ind: Dictionary. Maps word strings -> int code indices in vocab
           wordList: List of strings. Words for which we want to return their word embedding vectors

           Returns:
           -----------
           ndarray. Word embedding vectors. shape=(len(wordList), embedding_sz)
            This is the wt vectors from the 1st net layer at the index specified by each word's int-code.
        '''

        input =  []
        for word in wordList:
            if word not in word2ind:
                wordList.remove(word)
                print(f"'{word}' not in dictionary")
            else:
                input.append(word2ind[word])
        input = self.one_hot(input, self.C)
    
        return np.squeeze(input @ self.y_wts + self.y_b)

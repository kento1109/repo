class CNN_attention(Chain):
    def __init__(self, vocab_size, embedding_size, input_channel, output_channel_1, output_channel_2, output_channel_3, k1size, k2size, k3size, pooling_units, atten_size=20, output_size=args.classtype, train=True):
        super(CNN_attention, self).__init__(
            w2e = L.EmbedID(vocab_size, embedding_size),
            conv1 = L.Convolution2D(input_channel, output_channel_1, (k1size, embedding_size)),
            conv2 = L.Convolution2D(input_channel, output_channel_2, (k2size, embedding_size)),
            conv3 = L.Convolution2D(input_channel, output_channel_3, (k3size, embedding_size)),

            l1 = L.Linear(pooling_units, output_size),
            #Attention
            a1 = L.Linear(1, atten_size),
            a2 = L.Linear(atten_size, 1),
        )
        self.output_size = output_size
        self.train = train
        self.embedding_size = embedding_size
        self.ignore_label = 0
        self.w2e.W.data[self.ignore_label] = 0
        self.w2e.W.data[1] = 0  # 非文字
        self.input_channel = input_channel

    def initialize_embeddings(self, word2id):
        #w_vector = word2vec.Word2Vec.load_word2vec_format('./vector/glove.840B.300d.txt', binary=False)  # GloVe
        w_vector = word2vec.Word2Vec.load_word2vec_format('./vector/GoogleNews-vectors-negative300.bin', binary=True)  # word2vec
        for word, id in sorted(word2id.items(), key=lambda x:x[1])[1:]:
            if word in w_vector:
                self.w2e.W.data[id] = w_vector[word]
            else:
                self.w2e.W.data[id] = np.reshape(np.random.uniform(-0.25,0.25,self.embedding_size),(self.embedding_size,))

    def __call__(self, x):
        h_list = list()
        ox = copy.copy(x)
        if args.gpu != -1:
            ox.to_gpu()

        x = xp.array(x.data)
        x = F.tanh(self.w2e(x))
        b, max_len, w = x.shape  # batch_size, max_len, embedding_size
        x = F.reshape(x, (b, self.input_channel, max_len, w))

        c1 = self.conv1(x)
        b, outputC, fixed_len, _ = c1.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h1 = self.attention_pooling(F.relu(c1), b, outputC, fixed_len, tf)
        h1 = F.reshape(h1, (b, outputC))
        h_list.append(h1)

        c2 = self.conv2(x)
        b, outputC, fixed_len, _ = c2.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h2 = self.attention_pooling(F.relu(c2), b, outputC, fixed_len, tf)
        h2 = F.reshape(h2, (b, outputC))
        h_list.append(h2)

        c3 = self.conv3(x)
        b, outputC, fixed_len, _ = c3.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h3 = self.attention_pooling(F.relu(c3), b, outputC, fixed_len, tf)
        h3 = F.reshape(h3, (b, outputC))
        h_list.append(h3)

        h4 = F.concat(h_list)
        y = self.l1(F.dropout(h4, train=self.train))
        return y

    def set_tfs(self, x, b, outputC, fixed_len):
        TF = Variable(x[:,:fixed_len].data != 0, volatile='auto')
        TF = F.reshape(TF, (b, 1, fixed_len, 1))
        TF = F.broadcast_to(TF, (b, outputC, fixed_len, 1))
        return TF

    def attention_pooling(self, c, b, outputC, fixed_len, tf):
        reshaped_c = F.reshape(c, (b*outputC*fixed_len, 1))
        scala = self.a2(F.tanh(self.a1(reshaped_c)))
        reshaped_scala = F.reshape(scala, (b, outputC, fixed_len, 1)) 
        reshaped_scala = F.where(tf, reshaped_scala, Variable(-10*xp.ones((b, outputC, fixed_len, 1)).astype(xp.float32), volatile='auto'))  
        rereshaped_scala = F.reshape(reshaped_scala, (b*outputC, fixed_len))  # reshape for F.softmax
        softmax_scala = F.softmax(rereshaped_scala)
        atten = F.reshape(softmax_scala, (b*outputC*fixed_len, 1))
        a_h = F.scale(reshaped_c, atten, axis=0)
        reshaped_a_h = F.reshape(a_h, (b, outputC, fixed_len, 1))
        p = F.sum(reshaped_a_h, axis=2)
        return p

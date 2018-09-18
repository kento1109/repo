import numpy as np
from keras.layers import Input, Dense, Reshape, concatenate, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from utils import draw_sin_wave, draw_mnist


class RGAN():
    def __init__(self,
                 batch_size,
                 input_dim,
                 latent_dim,
                 hidden_dim,
                 sequence,
                 summary=False,
                 save_image=False,
                 save_model=False,
                 dataset='sin_wave'):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence = sequence
        self.batch_size = batch_size
        self.summary = summary
        self.save_image = save_image
        self.save_model = save_model
        self.dataset = dataset

        if dataset == 'sin_wave':
            self.draw = draw_sin_wave
        elif dataset == 'mnist':
            self.draw = draw_mnist
        else:
            raise ValueError

        # optimizer = RMSprop()
        g_optimizer = Adam(lr=0.1)
        d_optimizer = SGD(lr=0.1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.sequence, self.latent_dim))
        gen = self.generator(z)

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False
        self.set_trainable(self.discriminator, trainable=False)

        # The discriminator takes generated data as input and validate
        validity = self.discriminator(gen)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)

        # initialize MMD Class
        self.my_mmd = MMD(self.sequence, self.input_dim, self.latent_dim)

    def build_generator(self):

        model = Sequential()

        model.add(LSTM(units=self.hidden_dim,
                       batch_input_shape=(
                           None, self.sequence, self.latent_dim),
                       return_sequences=True))
        model.add(Dense(self.input_dim, activation='tanh'))
        if self.summary:
            print('generator model')
            model.summary()

        noise = Input(shape=(self.sequence, self.latent_dim),
                      name='g_noise_input')
        gen_x = model(noise)

        return Model(noise, gen_x)

    def build_discriminator(self):

        model = Sequential()

        model.add(LSTM(units=self.hidden_dim,
                       batch_input_shape=(None, self.sequence, self.input_dim),
                       return_sequences=True))
        model.add(Dense(self.input_dim, activation='sigmoid'))
        if self.summary:
            print('discriminator model')
            model.summary()

        inputs = Input(shape=(self.sequence, self.input_dim), name='d_input')
        validity = model(inputs)

        return Model(inputs, validity)

    def save_models(self):
        model_json_str = self.generator.to_json()
        open('models/' + self.dataset + '_generator_model.json', 'w') \
            .write(model_json_str)
        self.generator.save_weights(self.dataset + '_generator_weight.h5')
        model_json_str = self.discriminator.to_json()
        open('models/' + self.dataset + '_discriminator_model.json', 'w') \
            .write(model_json_str)
        self.generator.save_weights(self.dataset + '_discriminator_weight.h5')

    def set_trainable(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    def train(self, n_epochs, x_train, x_eval):

        if self.save_image:
            # plot real samples
            show_samples = 16
            real_samples = x_train[:show_samples]
            self.draw(real_samples, None, show_samples, -1)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, self.sequence, self.input_dim))
        fake = np.zeros((self.batch_size, self.sequence, self.input_dim))

        for epoch in range(n_epochs):

            g_loss = []
            d_loss = []
            acc = []
            best_mmd2 = 999

            np.random.shuffle(x_train)

            for i in range(int(x_train.shape[0] / self.batch_size)):
                # Select a random batch of
                x = x_train[i * self.batch_size: (i + 1) * self.batch_size]

                noise = np.random.normal(0, 1, (
                    self.batch_size, self.sequence, self.latent_dim))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of new images
                gen_x = self.generator.predict(noise)
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(x, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_x, fake)
                _d_loss, _acc = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss.append(_d_loss)
                acc.append(_acc)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (
                    self.batch_size, self.sequence, self.latent_dim))

                # Train the generator
                _g_loss = self.combined.train_on_batch(noise, valid)
                g_loss.append(_g_loss)

            # calculate maximum mean discrepancy
            eval_size = x_eval.shape[0]
            noise = np.random.normal(0, 1,
                                     (eval_size, self.sequence,
                                      self.latent_dim))
            eval_gan = self.generator.predict(noise)
            mmd2, that_np = self.my_mmd.calc_mmd(x_eval, eval_gan)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [mmd: %f]" % (
                epoch + 1,
                np.mean(d_loss),
                100 * np.mean(acc),
                np.mean(g_loss),
                mmd2))

            if self.save_image:
                show_samples = 16
                noise = np.random.normal(0, 1, (
                    show_samples, self.sequence, self.latent_dim))
                gen_samples = self.generator.predict(noise)
                self.draw(gen_samples, None, show_samples, epoch, title='gen')

            if self.save_model:
                if epoch > 10 and mmd2 < best_mmd2:
                    self.save_models()
                    best_mmd2 = mmd2


class RCGAN():
    def __init__(self,
                 batch_size,
                 input_dim,
                 latent_dim,
                 hidden_dim,
                 sequence,
                 labels,
                 summary=False,
                 save_image=False,
                 save_model=False,
                 gen_data=False,
                 dataset='sin_wave'):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence = sequence
        self.batch_size = batch_size
        self.labels = labels
        self.summary = summary
        self.save_image = save_image
        self.save_model = save_model
        self.gen_data = gen_data
        self.dataset = dataset

        if dataset == 'sin_wave':
            self.draw = draw_sin_wave
        elif dataset == 'mnist':
            self.draw = draw_mnist
        else:
            raise ValueError

        # optimizer = RMSprop()
        g_optimizer = RMSprop()
        d_optimizer = RMSprop()
        # g_optimizer = Adam(lr=0.1)
        # d_optimizer = SGD(lr=0.1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates data
        z = Input(shape=(self.sequence, self.latent_dim))
        # conditional input
        cond = Input(shape=(self.sequence, self.labels))
        gen = self.generator([z, cond])

        # For the combined model we will only train the generator
        self.set_trainable(self.discriminator, trainable=False)

        # The discriminator takes generated data as input and validate
        validity = self.discriminator([gen, cond])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, cond], validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)

        # initialize MMD Class
        self.my_mmd = MMD(self.sequence, self.input_dim, self.latent_dim)

    def build_generator(self):

        model = Sequential()

        model.add(LSTM(units=self.hidden_dim,
                       batch_input_shape=(
                           None, self.sequence, self.latent_dim + self.labels),
                       return_sequences=True))
        model.add(Dense(self.input_dim, activation='tanh'))
        if self.summary:
            print('generator model')
            model.summary()

        noise = Input(shape=(self.sequence, self.latent_dim),
                      name='g_noise_input')
        # conditional input
        cond = Input(shape=(self.sequence, self.labels), name='g_cond_input')
        cond_input = concatenate([noise, cond], axis=-1)
        gen_x = model(cond_input)

        return Model([noise, cond], gen_x)

    def build_discriminator(self):

        model = Sequential()

        model.add(LSTM(units=self.hidden_dim,
                       batch_input_shape=(None, self.sequence,
                                          self.input_dim + self.labels),
                       return_sequences=True))
        model.add(Dense(self.input_dim, activation='sigmoid'))
        if self.summary:
            print('discriminator model')
            model.summary()

        inputs = Input(shape=(self.sequence, self.input_dim), name='d_input')
        cond = Input(shape=(self.sequence, self.labels), name='d_cond_input')
        cond_input = concatenate([inputs, cond], axis=-1)
        validity = model(cond_input)

        return Model([inputs, cond], validity)

    def set_trainable(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    def save_models(self):
        model_json_str = self.generator.to_json()
        open('models/' + self.dataset + '_generator_model.json', 'w') \
            .write(model_json_str)
        self.generator.save_weights('models/' + self.dataset + '_generator_weight.h5')
        model_json_str = self.discriminator.to_json()
        open('models/' + self.dataset + '_discriminator_model.json', 'w') \
            .write(model_json_str)
        self.generator.save_weights('models/' + self.dataset + '_discriminator_weight.h5')

    def train(self, n_epochs, x_train, y_train, x_eval, y_eval):

        if self.save_image:
            # plot real samples
            show_samples = 16
            real_samples = x_train[:show_samples]
            real_labels = y_train[:show_samples]
            real_labels = np.argmax(real_labels, axis=1)
            self.draw(real_samples, real_labels, show_samples, -1)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, self.sequence, self.input_dim))
        fake = np.zeros((self.batch_size, self.sequence, self.input_dim))

        # set initial sigma for MMD
        self.my_mmd.set_sigma(x_eval)

        for epoch in range(n_epochs):

            g_loss = []
            d_loss = []
            acc = []
            best_mmd2 = 999

            # shuffle
            rnd = np.random.randint(999)
            for l in [x_train, y_train]:
                np.random.seed(rnd)
                np.random.shuffle(l)

            for i in range(int(x_train.shape[0] / self.batch_size)):
                # Select a random batch of
                x = x_train[i * self.batch_size: (i + 1) * self.batch_size]
                y = y_train[i * self.batch_size: (i + 1) * self.batch_size]
                # Repeat cond by sequence
                y = np.stack([y] * self.sequence, axis=1)

                noise = np.random.normal(0, 1, (
                    self.batch_size, self.sequence, self.latent_dim))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of new images
                gen_x = self.generator.predict([noise, y])
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([x, y], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_x, y],
                                                                fake)
                _d_loss, _acc = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss.append(_d_loss)
                acc.append(_acc)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (
                    self.batch_size, self.sequence, self.latent_dim))

                # # Condition on random labels
                # sampled_cond = np.random.randint(0,
                #                                  self.labels,
                #                                  (self.batch_size,
                #                                   self.labels))
                # sampled_cond = np.stack([sampled_cond] * self.sequence, axis=1)

                # Train the generator
                _g_loss = self.combined.train_on_batch([noise, y],
                                                       valid)
                g_loss.append(_g_loss)

            # calculate maximum mean discrepancy
            eval_size = x_eval.shape[0]
            noise = np.random.normal(0, 1,
                                     (eval_size, self.sequence,
                                      self.latent_dim))
            # sampled_cond = np.random.randint(0,
            #                                  self.labels,
            #                                  (eval_size,
            #                                   self.labels))
            # Repeat cond by sequence
            # sampled_cond = np.stack([sampled_cond] * self.sequence, axis=1)
            _y_eval = np.stack([y_eval] * self.sequence, axis=1)
            eval_gan = self.generator.predict([noise, _y_eval])
            mmd2, that_np = self.my_mmd.calc_mmd(x_eval, eval_gan)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [mmd: %f]" % (
                epoch + 1,
                np.mean(d_loss),
                100 * np.mean(acc),
                np.mean(g_loss),
                mmd2))

            if self.save_image:
                show_samples = 16
                noise = np.random.normal(0, 1, (
                    show_samples, self.sequence, self.latent_dim))
                sampled_cond = np.random.randint(0,
                                                 self.labels,
                                                 (show_samples,
                                                  self.labels))
                sampled_cond = np.stack([sampled_cond] * self.sequence, axis=1)
                gen_samples = self.generator.predict([noise, sampled_cond])
                predicted_label = np.argmax(sampled_cond[:, 0, :], axis=1)
                self.draw(gen_samples, predicted_label, show_samples, epoch,
                          title='gen')

            if epoch > 10 and mmd2 < best_mmd2:
                if self.save_model:
                    self.save_models()
                if self.gen_data:
                    # generate data for TSTR
                    TSTR_samples = 10000
                    noise = np.random.normal(0, 1, (
                        TSTR_samples, self.sequence, self.latent_dim))
                    sampled_cond = np.random.randint(0,
                                                     self.labels,
                                                     (TSTR_samples,
                                                      self.labels))
                    sampled_cond = np.stack([sampled_cond] * self.sequence,
                                            axis=1)
                    gen_samples = self.generator.predict([noise, sampled_cond])
                    predicted_label = sampled_cond[:, 0, :]
                    # predicted_label = np.argmax(sampled_cond[:, 0, :], axis=1)
                    print('generate synthetic data for TSTR...')
                    print('train shape:', gen_samples.shape)
                    np.savez('synthetic_data/data.npz', x=gen_samples,
                             y=predicted_label)
                best_mmd2 = mmd2

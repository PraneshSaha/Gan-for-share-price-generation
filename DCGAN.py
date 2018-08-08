class DCGAN():
    def __init__(self, data_len=1, prop_no=4,channel=1):
       
        self.data_len = data_len
        self.prop_no = prop_no
        self.data_channel=channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.6
        # In: 1 x 4 x 1, depth = 1
        # Out: 1 x 2 x 1, depth=64*8
        input_shape = (self.data_len, self.prop_no, self.data_channel)
        self.D.add(Conv2D(depth*1,(1,2), strides=1, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU())
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, (1,3), strides=2, padding='same'))
        self.D.add(LeakyReLU())
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4,(1,2), strides=1, padding='same'))
        self.D.add(LeakyReLU())
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, (1,2), strides=2, padding='same'))
        self.D.add(LeakyReLU())
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D


    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.6
        depth = 64*8
        dim = 1
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(Reshape((dim,dim,depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D(size=(1,2)))
        self.G.add(Conv2DTranspose(int(depth/2),(1,2), padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU())

        self.G.add(UpSampling2D(size=(1,2)))
        self.G.add(Conv2DTranspose(int(depth/4),(1,2), padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU())

        self.G.add(Conv2DTranspose(int(depth/8),(1,2), padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU())

    
        self.G.add(Conv2DTranspose(1,(1,1)))
        self.G.add(Activation('tanh'))
        self.G.summary()
        return self.G
    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class Share_DCGAN():
    def __init__(self,x_train):
        self.data_rows =1
        self.data_cols = 4
        self.x_train = x_train

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=250,file_name='C:/Users/Pranesh/Desktop/share/dat_1.txt'):
        noise_input = None
        f=open(file_name,'w')
        for i in range(train_steps):
            u=self.x_train.shape[0]-batch_size
            t=np.random.randint(0,u)
            data_train = self.x_train[t:t+batch_size,:,:,:]
            noise = np.random.normal(-1., 1.0, size=[batch_size, 100])
            data_fake = self.generator.predict(noise)
            print(data_train.shape[:],data_fake.shape[:])
            x = np.concatenate((data_train, data_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.normal(-1., 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            out=Out_put_vis()
            if i%1000==0:
                out.plot_fig(data_train,data_fake,str(i))
            for i in range(batch_size):
                s=str(data_fake[i,0,0,0])+","+str(data_fake[i,0,1,0])+","+str(data_fake[i,0,2,0])+","+str(data_fake[i,0,3,0])
                f.write(s+'\n')
        f.close() 
         

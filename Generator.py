from keras.models import load_model
class Data_gen():
    def gen(self,model,n,file_name):
        model = load_model(model)
        f=open(file_name,'w')
        for i in range(n):
            noise = np.random.normal(-1., 1.0, size=[1, 100])
            data_gen =model.predict(noise)
            s=str(data_gen[0,0,0,0])+","+str(data_gen[0,0,1,0])+","+str(data_gen[0,0,2,0])+","+str(data_gen[0,0,3,0])
            f.write(s+'\n')
        f.close()

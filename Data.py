class data(): 
    def get_imlist(self,path): #extract list of files in folder
        return [os.path.join(f) for f in os.listdir(path) if f.endswith('.txt')]
    def get_data(self,name_company,path): #get data from txt file
        lineData=list()
        with open(path+'/'+name_company,'rt') as f:
            reader = csv.reader(f,delimiter=' ',skipinitialspace=True)    
            cols=next(reader)
           
            for col in cols:
                lineData.append(list())
        
            for line in reader:
                for i in range(0,len(lineData)):
                    lineData[i].append(line[i])
        return lineData
    def extract_data(self,lineData): #extract data from lineData for training
        ses=lineData[0]
        l=len(ses)
        data=np.zeros((l,1,4))
        for i in range(l):
            date,op,mx,mn,cls,vol=ses[i].split(',')
            data[i,0,0]=float(op)
            data[i,0,1]=float(mx)
            data[i,0,2]=float(mn)
            data[i,0,3]=float(cls)
           
        return data
    def extract_data1(self,lineData): #extract data from generated data
        ses=lineData[0]
        l=len(ses)
        data=np.zeros((l,1,4,1))
        for i in range(l):
            op,mx,mn,cls=ses[i].split(',')
            data[i,0,0,0]=float(op)
            data[i,0,1,0]=float(mx)
            data[i,0,2,0]=float(mn)
            data[i,0,3,0]=float(cls)
           
        return data
    def train_test_split(self,data): #split into training and testing
        inc=data.shape[0]
        inc=int(inc*.7)
        train=data[:inc,:,:]
        test=data[inc:,:,:]
        return train

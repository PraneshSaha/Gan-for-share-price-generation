class Scale():
    def __init__(self):
        self.t2=1000
    def rescle_props(self,in_data):
        input_data=in_data        
       
        input_data[:,:,0]=input_data[:,:,0]/self.t2
        input_data[:,:,1]=input_data[:,:,1]/self.t2
        input_data[:,:,2]=input_data[:,:,2]/self.t2
        input_data[:,:,3]=input_data[:,:,3]/self.t2
        return input_data 
    def descle_props(self,pred_data):
        pred_data[:,:,0]=pred_data[:,:,0]*self.t2
        pred_data[:,:,1]=pred_data[:,:,1]*self.t2
        pred_data[:,:,2]=pred_data[:,:,2]*self.t2
        pred_data[:,:,3]=pred_data[:,:,3]*self.t2
        return pred_data
    

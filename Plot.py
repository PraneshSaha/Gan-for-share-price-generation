class Out_put_vis():  
    def plot_fig(self,input1,pred,n):
        plt.figure(1)
        t=np.arange(1,10000,1)
        p=input1.shape[0]
        l=min(t.shape[0],pred.shape[0])
        plt.plot(t[:p],input1[:p,0,0],'b--',t[:l],pred[:l,0,0], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/open_1'+n+'.png')
        plt.clf()
        
        plt.plot(t[:p],input1[:p,0,1],'b--',t[:l],pred[:l,0,1], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/high_1'+n+'.png')
        plt.clf()
        
        plt.plot(t[:p],input1[:p,0,2],'b--',t[:l],pred[:l,0,2], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/low_1'+n+'.png')
        plt.clf()
        
        plt.plot(t[:p],input1[:p,0,3],'b--',t[:l],pred[:l,0,3], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/close_1'+n+'.png')
        plt.clf()
    def plot_gen(self,pred):
        l=pred.shape[0]
        t=np.arange(1,l+1,1)
        plt.plot(t[:l],pred[:l,0,0], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/openp_1.png')
        plt.clf()
        
        plt.plot(t[:l],pred[:l,0,1], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/highp_1.png')
        plt.clf()
        
        plt.plot(t[:l],pred[:l,0,2], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/lowp_1.png')
        plt.clf()
        
        plt.plot(t[:l],pred[:l,0,3], 'r--')
        plt.savefig('C:/Users/Pranesh/Desktop/share/closep_1.png')
        plt.clf()

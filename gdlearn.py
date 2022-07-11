class SLinearRegression():
    def fit(self,X,Y):
        ln= len(X)
        def mean(list, ln):
            sum= 0
            for i in list:
                sum= sum + i
            mean1= sum/ln
            return mean1
        def uiu(list,meanvalue):
            diff= []
            sum=0
            for i in list:
                dif= i - meanvalue
                difsq= dif * dif
                diff.append(difsq)
            for j in diff:
                sum= sum + j
            return sum
        def uiui(list1,list2,ln,xmean,ymean):
            ls1= []
            ls2= []
            ls3= []
            sum= 0
            for i in list1:
                dif1= i - xmean
                ls1.append(dif1)
            for j in list2:
                dif2= j - ymean
                ls2.append(dif2)
            for u in range(ln):
                prod= ls1[u]*ls2[u]
                ls3.append(prod)
            for r in ls3:
                sum= sum + r
            return sum
        self.Xmean = mean(X,ln)
        self.Ymean= mean(Y,ln)
        self.C= uiu(X,self.Xmean) #C is  ∑(Xi-Xbar)sq
        self.D= uiu(Y,self.Ymean) #D is  ∑(Yi-Ybar)sq
        self.E= uiui(X,Y,ln,self.Xmean,self.Ymean) #E is  ∑((Xi-Xbar) * (Yi-Ybar))
        self.b1= (self.E)/(self.C) 
        self.b0= self.Ymean - (self.b1*self.Xmean)

    def predict(self,nx):
        Yhat = ((self.b1)*nx) + self.b0 
        print(Yhat)
    
    def test(self, test_data,test_label,method='MSE'):
        ylr = []
        ls = 0
        for i in test_data:
            yht = (self.b1 * i) + self.b0
            ylr.append(yht)
        if method == 'MSE':
            for i in range(len(test_label)):
                sq = ((test_label[i]-ylr[i])**2)
                ls = ls + sq
                loss = ls/(len(test_label))
        if method == 'RMSE':
            for i in range(len(test_label)):
                sq = ((test_label[i]-ylr[i])**2)
                ls = ls + sq
                loss = (ls/(len(test_label)))**0.5
        if method == 'MAE':
            for i in range(len(test_label)):
                sq = abs(((test_label[i]-ylr[i])**2))
                ls = ls + sq
                loss = ls/(len(test_label))
        print(str(method) + ' is: ' + str(loss))


class LogisticRegression():
    def fit(self,X,Y):
        ln= len(X)
        def mean(list, ln):
            sum= 0
            for i in list:
                sum= sum + i
            mean1= sum/ln
            return mean1
        def uiu(list,meanvalue):
            diff= []
            sum=0
            for i in list:
                dif= i - meanvalue
                difsq= dif * dif
                diff.append(difsq)
            for j in diff:
                sum= sum + j
            return sum
        def uiui(list1,list2,ln,xmean,ymean):
            ls1= []
            ls2= []
            ls3= []
            sum= 0
            for i in list1:
                dif1= i - xmean
                ls1.append(dif1)
            for j in list2:
                dif2= j - ymean
                ls2.append(dif2)
            for u in range(ln):
                prod= ls1[u]*ls2[u]
                ls3.append(prod)
            for r in ls3:
                sum= sum + r
            return sum
        self.Xmean = mean(X,ln)
        self.Ymean= mean(Y,ln)
        self.C= uiu(X,self.Xmean) #C is  ∑(Xi-Xbar)sq
        self.D= uiu(Y,self.Ymean) #D is  ∑(Yi-Ybar)sq
        self.E= uiui(X,Y,ln,self.Xmean,self.Ymean) #E is  ∑((Xi-Xbar) * (Yi-Ybar))
        self.b1= (self.E)/(self.C) 
        self.b0= self.Ymean - (self.b1*self.Xmean)

    def predict(self,nx,threshold = 0.5):
        def sigmoid(u):
            sig = 1 / (1 + ((2.718281828) ** -(u)))
            return sig
        Yhat = ((self.b1)*nx) + self.b0 
        if (sigmoid(Yhat)>threshold):
            print('[1]')
        else:
            print('[0]')
    
    def test(self, test_data,test_label,threshold):
        def sigmoid(u):
            sig = 1 / (1 + ((2.718281828) ** -(u)))
            return sig
        ylr = []
        for i in test_data:
            yht = sigmoid(((self.b1 * i) + self.b0))
            if yht > threshold:
                ylr.append(1)
            else:
                ylr.append(0)
        p = 0
        n = 0
        for i in range(len(test_label)):
            if (test_label[i]==ylr[i]):
                p = p + 1
            else:
                n = n + 1
        score = ((p * 100)/len(test_label))
        print('Test Score is: ' + str(round(score,2)) + ' %')


class KNNClassifier():
    def fit(self,data, data_label):
        ln = len(data)
        self.data = data.to_numpy()
        unk = list(data_label.unique())
        self.data_label = data_label.to_numpy()
    def predict(self,lst,k):
        def dist(lst):
            fsqr = []
            for i in range(len(self.data)):
                sq = 0
                for j in range(len((self.data[0]))):
                               sq = sq + (lst[j] - (self.data[i][j]))**2
                sqr = sq**0.5
                fsqr.append(sqr)
            return fsqr 
        A = dist(lst)
        B = A.copy()
        B.sort()
        ls1 = []
        pp = set(self.data_label)
        t = self.data_label
        for r in range(k):
            ind = A.index(B[r])
            A[ind] = 'o'
            ls1.append(ind)
        ls2 = []
        for p in range(k):
            el = t[ls1[p]]
            ls2.append(el)
        rs = list(pp)
        d = {}
        for i in rs:
            ctr = 0
            for j in ls2: #
                if i==j:
                    ctr = ctr + 1
            d.update({i:ctr})
        ltk = list(d.keys()) 
        ltv = list(d.values())
        indd = ltv.index(max(ltv))
        return ltk[indd]
    def test(self,test_data,test_label,k):
        self.test_data = test_data.to_numpy()
        self.test_label = test_label.to_numpy()
        ykn = []
        for i in range(len(test_data)):
            self.k = k
            a = self.predict(self.test_data[i],self.k)
            ykn.append(a)
        p = 0
        n = 0
        for w in range(len(self.test_label)):
            if (self.test_label[w]==ykn[w]):
                p = p + 1
            else:
                n = n + 1
        score = ((p * 100)/len(self.test_label))
        print('Test Score is: ' + str(round(score,5)) + ' %')


class NaiveBayes():
    def fit(self, data, data_label):
        self.data = data
        self.data_label = data_label
        stdl = set(self.data_label)
        self.ot = list(stdl)
        self.cl = self.data.columns.tolist()
        unval = []
        unlist = []
        problist = []
        probcl = []
        fproblist = []
        for i in range(len(self.cl)):
            unva = data[self.cl[i]]
            unval = unva.unique().tolist()
            unlist.append(unval)
        self.unlist = unlist
        self.data = data.to_numpy()
        self.data_label= data_label.to_numpy()
        for i in range(len(self.ot)):
            problist = []
            for j in range(len(self.cl)):
                prob = []
                for l in range(len(self.unlist[j])):   
                    ctrc = 0
                    ctri = 0
                    for k in range(len(self.data_label)):
                        if (self.data[k][j]) == (self.unlist[j][l]) and (self.data_label[k]) == (self.ot[i]):
                            ctrc = ctrc + 1
                        if self.data_label[k] == self.ot[i]:
                            ctri = ctri + 1
                    cond = ctrc / ctri
                    prob.append(cond)
                problist.append(prob)
            fproblist.append(problist)
            probcl.append(ctri/len(self.data_label))
        self.fproblist = fproblist
        self.probcl = probcl
    def predict(self,Xdata):
        self.Xdata = Xdata
        rs = []
        for i in range(len(self.ot)):
            prodn = 1
            for j in range(len(self.Xdata)):
                if self.Xdata[j] not in self.unlist[j]:
                    prd = 0
                else:
                    prd = self.fproblist[i][j][self.unlist[j].index(self.Xdata[j])]
                prodn = prodn * prd
            fprod = prodn * self.probcl[i]
            rs.append(fprod)
        return (self.ot[rs.index(max(rs))])
    def test(self, test_data,test_label):
        ylr = []
        test_data = test_data.to_numpy()
        test_label = test_label.to_numpy()
        for y in range(len(test_data)):
            A = self.predict(test_data[y])
            ylr.append(A)
        p = 0
        n = 0
        for o in range(len(test_label)):
            if (test_label[o]==ylr[o]):
                p = p + 1
            else:
                n = n + 1
        score = ((p * 100)/len(test_label))
        print('Test Score is: ' + str(round(score,2)) + ' %')
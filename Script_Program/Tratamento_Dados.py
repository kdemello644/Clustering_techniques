from sklearn.preprocessing import StandardScaler
class Tratamento_Dados:
    def __init__(self,data,target):
        self.data = data
        self.target = target
    @property
    def tratamento_dados(self)->list:
        df = self.data
        y = df.pop(self.target).values
        X = df
        scaler = StandardScaler()
        scaler.fit(X)
        scaler.fit(y.reshape(-1,1))
        return X, y
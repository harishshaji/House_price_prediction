import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello():
    result=''
    # if request.method=='POST':
    #     print('post')
    #     score1=request.form.get('score1')
    #     score2=request.form.get('score2')
    #     failure=request.form.get('failure')
    #     sex=request.form.get('sex')
    #     return render_template ("index.html",result=[score1,score2])
    return render_template("index.html",**locals())

##############################################################################################################
@app.route('/predict',methods=['POST','GET'])
def predict():
    MSSubClas =(request.form.get('n1'))
    MSZoning=(request.form.get('n2'))
    LotFrontage=(request.form.get('n3'))
    YearBuild=(request.form.get('n6'))
    LotShape=(request.form.get('n7'))
    FirstFloor=(request.form.get('n8'))
    secondFloor=(request.form.get('n9'))
    LotArea=(request.form.get('n4'))
    Street=(request.form.get('n5'))

######################################################################################################################

    df=pd.read_csv('houseprice.csv',usecols=["SalePrice","MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                             "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()
    list_row={}
    list_row = {"MSSubClass":80,"MSZoning":"RL","LotFrontage":70,"LotArea":7000,"Street":"pahe","YearBuilt":2000,"LotShape":"Reg","1stFlrSF":756,"2ndFlrSF":754}
    df = df.append(list_row, ignore_index=True)
    import datetime
    datetime.datetime.now().year
    df['Total Years']=datetime.datetime.now().year-df['YearBuilt']
    df.drop("YearBuilt",axis=1,inplace=True)
    cat_features=["MSSubClass", "MSZoning", "Street", "LotShape"]
    out_feature="SalePrice"
    from sklearn.preprocessing import LabelEncoder
    lbl_encoders={}
    for feature in cat_features:
        lbl_encoders[feature]=LabelEncoder()
        df[feature]=lbl_encoders[feature].fit_transform(df[feature])
    import numpy as np

    cat_features=np.stack([df['MSSubClass'],df['MSZoning'],df['Street'],df['LotShape']],1)

    cat_features

    import torch
    cat_features=torch.tensor(cat_features,dtype=torch.int64)
    cat_features

    cont_features=[]
    for i in df.columns:
        if i in ["MSSubClass", "MSZoning", "Street", "LotShape","SalePrice"]:
            pass
        else:
            cont_features.append(i)
    cont_values=np.stack([df[i].values for i in cont_features],axis=1)
    cont_values=torch.tensor(cont_values,dtype=torch.float)
    y=torch.tensor(df['SalePrice'].values,dtype=torch.float).reshape(-1,1)
    cat_dims=[len(df[col].unique()) for col in ["MSSubClass", "MSZoning", "Street", "LotShape"]]

    embedding_dim= [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    embed_representation=nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
    cat_featuresz=cat_features[:4]
    pd.set_option('display.max_rows', 500)
    embedding_val=[]
    for i,e in enumerate(embed_representation):
        embedding_val.append(e(cat_features[:,i]))
    z = torch.cat(embedding_val, 1)
    droput=nn.Dropout(.4)
    final_embed=droput(z)
    batch_size=1200
    test_size=int(batch_size*0.15)
    train_categorical=cat_features[:batch_size-test_size]
    test_categorical=cat_features[batch_size-test_size:batch_size]
    train_cont=cont_values[:batch_size-test_size]
    test_cont=cont_values[batch_size-test_size:batch_size]
    y_train=y[:batch_size-test_size]
    y_test=y[batch_size-test_size:batch_size]
    pca = test_categorical[-2]
    pco = test_cont[-2]
    pca = pca.resize_(2, 4)
    pco = pco.resize_(2, 5)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class FeedForwardNN(nn.Module):

        def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.5):
            super().__init__()
            self.embeds = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embedding_dim])
            self.emb_drop = nn.Dropout(p)
            self.bn_cont = nn.BatchNorm1d(n_cont)

            layerlist = []
            n_emb = sum((out for inp, out in embedding_dim))
            n_in = n_emb + n_cont

            for i in layers:
                layerlist.append(nn.Linear(n_in, i))
                layerlist.append(nn.ReLU(inplace=True))
                layerlist.append(nn.BatchNorm1d(i))
                layerlist.append(nn.Dropout(p))
                n_in = i
            layerlist.append(nn.Linear(layers[-1], out_sz))

            self.layers = nn.Sequential(*layerlist)

        def forward(self, x_cat, x_cont):
            embeddings = []
            for i, e in enumerate(self.embeds):
                embeddings.append(e(x_cat[:, i]))
            x = torch.cat(embeddings, 1)
            x = self.emb_drop(x)

            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1)
            x = self.layers(x)
            return x


    ### Loading the saved Model
    embs_size=[(15, 8), (5, 3), (2, 1), (4, 2)]
    model1=FeedForwardNN(embs_size,5,1,[100,50],p=0.4)


    model1.load_state_dict(torch.load('HouseWeights.pt'))
    ypredict = model1(pca,pco)
    # print(ypredict[1])
    # print(pca,pco)
    data_predicted=pd.DataFrame(ypredict.tolist(),columns=["Prediction"])
    l=data_predicted.iloc[1]
    pred1=l["Prediction"]
    # print(data_predicted)
    # print(type(data_predicted))

    return render_template("index.html", result=pred1)
if __name__ == "__main__":
    app.run(debug=True)





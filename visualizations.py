import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import math

#From https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow

if __name__ == "__main__":
    df = pd.read_csv('./creditcard.csv')
    df['Time'] = df['Time'].apply(lambda x: (float(x) % 86400) / 86400.0)
    df['Amount'] = df['Amount'].apply(lambda x: math.log10(max([float(x), 1])))

    start = 0
    end = 30

    for i, cn in enumerate(df[df.ix[:, start:end].columns]):
        plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        sns.distplot(df[cn][df.Class == 0], bins=50)
        sns.distplot(df[cn][df.Class == 1], bins=50)
        ax.set_xlabel('')
        ax.set_title('Histogram for feature ' + str(cn))
        plt.savefig(str(cn) + ".png")
        plt.close()


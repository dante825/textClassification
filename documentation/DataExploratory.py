import pandas as pd
import matplotlib.pyplot as plt


def input_data_explore():
    input_df = pd.read_csv('../output/20newsGroup18828.csv')

    print(input_df.shape)

    fig, ax = plt.subplots(figsize=(17, 7))
    agg_df = input_df.groupby('category').count().sort_values('category')
    print(agg_df)
    fig_plot = agg_df.plot(kind='bar', ax=ax, title='The count of each category of the dataset',
                legend=False)
    fig_plot.set_xlabel('Categories')
    fig_plot.set_ylabel('Count')

    plt.tight_layout()
    # plt.autoscale()
    # plt.show()
    plt.savefig('../output/count.png', format='png')


def tf_dimension_effect():
    tf_data = {
        'KNN': [0.28, 0.29, 0.31, 0.39, 0.45, 0.49, 0.52, 0.52, 0.54, 0.55, 0.53, 0.3],
        'SVM': [0.81, 0.81, 0.8, 0.78, 0.77, 0.77, 0.74, 0.71, 0.68, 0.64, 0.56, 0.27],
        'NN': [0.84, 0.8, 0.8, 0.79, 0.78, 0.76, 0.73, 0.72, 0.7, 0.69, 0.63, 0.34]
            }

    df = pd.DataFrame(tf_data, index=[8000, 6000, 4000, 2000, 1000, 500, 250, 200, 150, 100, 50, 10])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(17, 7))
    fig_plot = df.plot(kind='line', ax=ax, title='Effect of dimension reduction on accuracy (term frequency with SVD)',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfsvd.png", format='png')
    plt.show()


def tfidf_dimension_effect():
    tfidf_data = {
        'KNN': [0.77, 0.77, 0.77, 0.58, 0.55, 0.55, 0.61, 0.65, 0.67, 0.69, 0.69, 0.55],
        'SVM': [0.87, 0.87, 0.87, 0.86, 0.84, 0.83, 0.8, 0.8, 0.78, 0.76, 0.74, 0.5],
        'NN': [0.88, 0.86, 0.85, 0.84, 0.82, 0.8, 0.8, 0.81, 0.8, 0.8, 0.77, 0.59]
            }

    df = pd.DataFrame(tfidf_data, index=[8000, 6000, 4000, 2000, 1000, 500, 250, 200, 150, 100, 50, 10])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(17, 7))
    fig_plot = df.plot(kind='line', ax=ax, title='Effect of dimension reduction on accuracy (TF-IDF with SVD)',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfidfsvd.png", format='png')
    plt.show()


def main():
    # input_data_explore()
    # tf_dimension_effect()
    tfidf_dimension_effect()


if __name__ == '__main__':
    main()


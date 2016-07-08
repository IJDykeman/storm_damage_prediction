import pandas
import numpy as np
def load_dataset(path):
    print("loading...")
    data = pandas.read_hdf(path, '/df')
    df = pandas.DataFrame(data)
    data_dict = {}
    for label in set(df._get_numeric_data().columns).union({'hcad'}):
        # union hcad to ensure that hcad col comes in even if not considered numerical
        # if label != 'hcad':
        data_dict[label] = df[label].astype(float)
        # df[label][df[label] > 1] = 1.0

    # df['hcad'] = df['hcad'].astype(float)
    result = pandas.DataFrame.from_dict(data_dict)

    result = result.replace([np.inf, -np.inf], 1)
    
    return result.sort(['hcad']).fillna(0)


hcad = load_dataset("../Dropbox/data_for_brian/y_df.hd")
hcad.to_csv("./y_df.csv")
print "done to csv"
# META = load_dataset("~/Dropbox/data_for_brian/meta/df_meta.hd")
# y_data = load_data("~/Dropbox/data_for_brian/y_df.hd")

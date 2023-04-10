import pandas as pd

def load_yelp_orig_data():
    review_path = './dataset/yelp_dataset/yelp_academic_dataset_review.json'
    output_path = './dataset/output'

    # read the entire file into a python array
    with open(review_path, 'r') as f:
        data = f.readlines()

    data = data[:3500000]
    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)

    data_df.head(100000).to_csv(output_path + '/output_reviews_top.csv')

load_yelp_orig_data()
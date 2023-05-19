"""Script to convert csv data set of track to json format"""

import argparse
import json
import pandas as pd

def parse_to_json(file_path):
    """Reads the csv data, convert it to json and dumps in to seperate file
    
    :param file_path: Location of csv file. This param is required
    :return: writes the json data into a seperate file. The json includes a list each track data
            is stored in seperate index having keys:
            * artist_name
            * track_name
            * genre
            * lyrics
    """
    
    # read the csv data
    csv_data = pd.read_csv(file_path, usecols=["artist_name", "track_name", "genre", "lyrics"])

    # convert the csv data to json
    json_data = json.loads(csv_data.to_json(orient="records"))

    with open("tracks.json", 'w') as json_file:
        json.dump(json_data, json_file)


if __name__== "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert csv file to json",
        prog="parser_csv"
    )

    parser.add_argument(
        "--file",
        help="Path of csv file",
        required=True
    )

    args = vars(parser.parse_args())

    parse_to_json(args["file"])
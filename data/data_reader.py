import pandas as pd
import requests

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def hu_data_loader(address='http://humap2.proteincomplexes.org/static/downloads/humap2/humap2_complexes_20200809.txt'):
  file_url = 'http://humap2.proteincomplexes.org/static/downloads/humap2/humap2_complexes_20200809.txt'
  destination_path = file_url.split('/')[-1]
  download_file(file_url,destination_path)


  data=pd.read_csv(destination_path)
  return data

